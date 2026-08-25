from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pickle
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_learning.config import ARTIFACTS_DIR


def _index_path(dataset: str, splits: list[str], out: Path | None) -> Path:
    if out is not None:
        return out
    if len(splits) == 1:
        return ARTIFACTS_DIR / "indexes" / f"bm25_index_{dataset}_{splits[0]}.pkl"
    # Multi-split combined: name reflects all splits joined by '+' so the
    # ordering of --split args doesn't change the filename.
    suffix = "+".join(splits)
    return ARTIFACTS_DIR / "indexes" / f"bm25_index_{dataset}_{suffix}.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a BM25 / dense / hybrid index from one or more dataset splits."
    )
    parser.add_argument("--dataset", default="hotpotqa", help="Dataset: hotpotqa, triviaqa, ...")
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        help="Dataset split(s). Repeat the flag to combine multiple splits into one index "
             "(e.g. --split train --split validation). Default: validation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of examples to index PER SPLIT. Default: all available.",
    )
    parser.add_argument(
        "--retrieval-mode",
        default="hybrid",
        choices=["bm25", "dense", "hybrid"],
        help="Index type to build. 'hybrid' = BM25 + dense + RRF (recommended).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output index path. Auto-detected from dataset/splits if not set.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent embedding batches to dispatch to Ollama. "
             "Each worker holds one /api/embed HTTP request in flight. "
             "Default 4 matches a typical local Ollama OLLAMA_NUM_PARALLEL. "
             "Use --workers 1 to run serially.",
    )
    parser.add_argument(
        "--bm25-workers",
        type=int,
        default=1,
        help="Number of processes for parallel BM25 tokenization. Tokenize "
             "holds the GIL (regex), so threads would not help — only "
             "processes. Default 1 (single-process) is fastest for the "
             "typical <10k-paragraph HotpotQA corpus; raise this for larger "
             "corpora (e.g. Simple English Wikipedia behind PopQA).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild from scratch even if the final .pkl already exists. "
             "Sidecars are also regenerated.",
    )
    parser.add_argument(
        "--extend",
        action="store_true",
        help="Extend an existing index by appending new paragraphs instead of "
             "rebuilding from scratch. Reuses the BM25 tokenized corpus and "
             "embeddings for paragraphs whose IDs match the prefix of the "
             "new dedup'd list (i.e. the same --split, larger --limit). Only "
             "the newly-added paragraphs are tokenized and embedded. The "
             "final BM25Okapi is rebuilt from the concatenated corpus; the "
             "embedding mmap is resized in place. If the sidecar's doc IDs "
             "don't match the new prefix, this errors out (use --force to "
             "rebuild instead).",
    )
    return parser.parse_args()


def _dedupe_documents(documents: list) -> list:
    """Drop exact-duplicate paragraphs (same title + text). BM25 tokenization
    and embedding are both O(N), so deduping saves real time on the
    train+validation combined build where some paragraphs repeat."""
    seen: set[tuple[str, str]] = set()
    out: list = []
    for doc in documents:
        key = (doc.title, doc.text)
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)
    return out


# ---------------------------------------------------------------------------
# Incremental sidecars — let a killed build resume from where it stopped.
# Sidecars are written next to the final .pkl with the same base name:
#   <base>.docs.pkl      deduped Document list
#   <base>.bm25.pkl      tokenized corpus + pickled BM25Okapi object
#   <base>.emb.npy       mmap'd dense embeddings, (N, dim) float32
#   <base>.emb_count.txt number of fully-embedded rows (resume marker)
# A stale sidecar (different N or different dim) is detected and rebuilt.
# ---------------------------------------------------------------------------

def _sidecar_paths(output: Path) -> dict[str, Path]:
    base = output.with_suffix("")  # /path/foo.pkl -> /path/foo
    return {
        "docs": Path(f"{base}.docs.pkl"),
        "bm25": Path(f"{base}.bm25.pkl"),
        "emb": Path(f"{base}.emb.npy"),
        "emb_count": Path(f"{base}.emb_count.txt"),
    }


def _save_docs_sidecar(path: Path, documents: list) -> None:
    from dataclasses import asdict
    with path.open("wb") as f:
        pickle.dump([asdict(d) for d in documents], f)


def _load_docs_sidecar(path: Path, expected_n: int | None) -> list | None:
    """Return the deduped documents from `path`, or None if missing/stale.

    Pass ``expected_n=None`` to skip the N check (lenient load — used by
    ``--extend`` to inspect the sidecar regardless of its size).
    """
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            payload = pickle.load(f)
    except Exception:
        return None
    if not isinstance(payload, list):
        return None
    if expected_n is not None and len(payload) != expected_n:
        return None
    from rag_learning.data import Document
    try:
        return [Document(**item) for item in payload]
    except Exception:
        return None


def _save_bm25_sidecar(path: Path, tokenized_corpus: list, bm25_obj) -> None:
    payload = {
        "tokenized_corpus": tokenized_corpus,
        "bm25_object_bytes": pickle.dumps(bm25_obj),
        "n": len(tokenized_corpus),
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def _load_bm25_sidecar(path: Path, expected_n: int):
    """Return (tokenized_corpus, bm25_obj) or None if missing/stale."""
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            payload = pickle.load(f)
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("n") != expected_n:
        return None
    try:
        return payload["tokenized_corpus"], pickle.loads(payload["bm25_object_bytes"])
    except Exception:
        return None


def _read_emb_count(count_path: Path) -> int:
    if not count_path.exists():
        return 0
    try:
        return max(0, int(count_path.read_text().strip()))
    except Exception:
        return 0


def _write_emb_count(count_path: Path, count: int) -> None:
    """Write ``count`` to the sidecar file, but only if it advances the
    existing value. Parallel embedding batches can complete out of order;
    a later batch's start index might be higher than an earlier batch's,
    so we want monotonically non-decreasing progress.
    """
    try:
        prev = int(count_path.read_text().strip())
    except Exception:
        prev = -1
    if count <= prev:
        return
    count_path.write_text(str(count))


def _open_emb_mmap(
    path: Path, count_path: Path, n: int, dim: int
) -> tuple[np.ndarray, int]:
    """Open (or create) a memory-mapped float32 array of shape (n, dim)
    for incrementally-written embeddings.

    Returns (mmap_array, start_count). start_count is the number of
    fully-written rows from a previous run; the caller should resume
    embedding from that index. If the existing file's shape doesn't
    match (N or dim changed), it's deleted and recreated.

    The returned mmap is read-write so callers can fill in remaining rows.
    """
    if path.exists():
        try:
            # r+ = read+write on existing file. (w+ would truncate.)
            existing = np.load(path, mmap_mode="r+")
        except Exception:
            existing = None
        if existing is not None and existing.shape == (n, dim):
            count = min(_read_emb_count(count_path), n)
            return existing, count
        # Stale: shape doesn't match (N changed, dim changed, or file corrupt).
        path.unlink()
        if count_path.exists():
            count_path.unlink()
    arr = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(n, dim)
    )
    arr[:] = 0.0
    arr.flush()
    return arr, 0


def _prefix_match_len(old_docs: list, new_docs: list) -> int:
    """How many docs at the start of ``new_docs`` have the same ``id`` as
    ``old_docs`` (and in the same order)?

    The match is strict and contiguous from index 0. Returns 0 if either
    list is empty or the first id differs. Used by ``--extend`` to decide
    how many existing rows we can reuse.
    """
    if not old_docs or not new_docs:
        return 0
    n = min(len(old_docs), len(new_docs))
    for i in range(n):
        if old_docs[i].id != new_docs[i].id:
            return i
    return n


def _extend_emb_mmap(
    path: Path, count_path: Path, n_old: int, n_new: int, dim: int
) -> tuple[np.ndarray, int]:
    """Resize an existing mmap from ``(n_old, dim)`` to ``(n_new, dim)`` while
    preserving the first ``n_old`` rows. Returns ``(new_mmap, n_old)`` so
    the caller can continue embedding from row ``n_old``.

    The resize is destructive on disk (the old file is deleted and a new
    one created). If a write fails mid-resize, the caller should ``rm`` the
    emb sidecars and re-run; the BM25 sidecar is unaffected.
    """
    if not path.exists():
        # Nothing to extend; behave like a fresh open.
        arr = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32, shape=(n_new, dim)
        )
        arr[:] = 0.0
        arr.flush()
        _write_emb_count(count_path, 0)
        return arr, 0
    old = np.load(path, mmap_mode="r")
    if old.shape != (n_old, dim):
        # Stale (someone wrote a different shape); drop and start fresh.
        old = None
        path.unlink()
        if count_path.exists():
            count_path.unlink()
    if old is None:
        arr = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32, shape=(n_new, dim)
        )
        arr[:] = 0.0
        arr.flush()
        _write_emb_count(count_path, 0)
        return arr, 0
    # Resize: read the existing rows into memory, delete the file, recreate
    # with the new shape, copy rows back. ``n_new`` can be > or == ``n_old``;
    # if ``n_new <= n_old`` the caller has effectively shrunk the corpus
    # (rare), in which case the trailing rows are dropped.
    old_data = np.array(old[:n_old], dtype=np.float32, copy=True)
    del old
    path.unlink()
    if count_path.exists():
        count_path.unlink()
    new = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(n_new, dim)
    )
    new[:n_old] = old_data
    new.flush()
    # Reset count to ``n_old`` so the next run resumes from there.
    _write_emb_count(count_path, n_old)
    return new, n_old


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _embed_to_mmap(
    truncated: list[str],
    emb_arr: np.ndarray,
    count_path: Path,
    start_count: int,
    batch_size: int = 64,
    max_workers: int = 4,
    embedder=None,
) -> int:
    """Embed ``truncated[start_count:]`` and write rows into ``emb_arr``.

    After each successfully-completed batch, the corresponding rows are
    flushed to disk and the new count is written to ``count_path``. If the
    process is killed, the on-disk mmap is valid up to the last flushed
    batch, and ``count_path`` marks the resume point for the next run.

    Returns the final count (== len(truncated) if all embedded, or the
    last completed batch boundary if interrupted).
    """
    if embedder is None:
        from rag_learning.embed import Embedder
        embedder = Embedder()

    n = len(truncated)
    pending = list(range(start_count, n, batch_size))
    if not pending:
        return n

    def _embed_with_diagnose(batch_idx: int, batch: list[str]) -> np.ndarray:
        try:
            return embedder.embed(batch)
        except Exception as exc:
            print(f"\n  [embed] batch {batch_idx}-{batch_idx + len(batch)} failed: {exc!r}")
            if hasattr(exc, "response") and exc.response is not None:
                try:
                    body = exc.response.text
                    print(f"  [embed] Ollama response body: {body[:500]}")
                except Exception:
                    pass
            for j, t in enumerate(batch):
                if len(t) > 20000:
                    print(f"  [embed]   batch[{j}] is {len(t)} chars "
                          f"(first 200): {t[:200]!r}")
            raise

    def _report(done: int, start: float) -> None:
        elapsed = time.time() - start
        rate = done / max(elapsed, 0.1)
        eta = (n - done) / max(rate, 0.1)
        print(f"  embedded {done}/{n} ({100 * done / n:.0f}%) "
              f"in {elapsed:.0f}s — {rate:.0f} docs/s, ETA {eta:.0f}s")

    start = time.time()
    completed = start_count

    if max_workers <= 1 or len(pending) <= 1:
        for i in pending:
            batch = truncated[i : i + batch_size]
            vecs = _embed_with_diagnose(i, batch)
            emb_arr[i : i + len(batch)] = vecs
            emb_arr.flush()
            completed = i + len(batch)
            _write_emb_count(count_path, completed)
            _report(completed, start)
    else:
        workers = min(max_workers, len(pending))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(_embed_with_diagnose, i, truncated[i : i + batch_size]): i
                for i in pending
            }
            for fut in as_completed(futures):
                i = futures[fut]
                vecs = fut.result()
                emb_arr[i : i + len(vecs)] = vecs
                emb_arr.flush()
                completed = i + len(vecs)
                _write_emb_count(count_path, completed)
                _report(completed, start)
    return completed


def main() -> None:
    from rag_learning.data import iter_context_paragraphs, load_examples
    from rag_learning.embed import Embedder
    from rag_learning.index import BM25Index, DenseIndex, HybridIndex, tokenize_corpus

    args = parse_args()
    if args.split is None or len(args.split) == 0:
        args.split = ["validation"]
    if args.output is None:
        args.output = _index_path(args.dataset, args.split, None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sidecars = _sidecar_paths(args.output)

    # ---- Final .pkl already exists? Done, exit early (unless --force or --extend). ----
    # --extend is allowed past this check because the .pkl is rewritten with
    # the larger corpus at the end of the run.
    if args.output.exists() and not args.force and not args.extend:
        print(f"Index already exists at {args.output}.")
        print("  Use --force to rebuild from scratch.")
        return

    # ---- Load + dedupe paragraphs (with sidecar resume) ----
    # We always have to call load_examples + iter_context_paragraphs once
    # to know the would-be dedup'd N — that's what tells us whether the
    # existing sidecars are still valid. The dataset itself is also small
    # (just a metadata load on cached HF datasets) so this isn't a real
    # cost; the savings are on the BM25 + embedding work.
    documents: list = []
    n_examples_per_split: dict[str, int] = {}
    for split in args.split:
        examples = load_examples(split=split, limit=args.limit, dataset=args.dataset)
        n_examples_per_split[split] = len(examples)
        documents.extend(iter_context_paragraphs(examples, dataset=args.dataset))
    expected_n = len(_dedupe_documents(documents))  # O(N); just for N check

    loaded = _load_docs_sidecar(sidecars["docs"], expected_n=expected_n)
    if loaded is not None:
        # Even when N matches, verify the first id is the same corpus — a
        # config change (--dataset, --split, dataset version) can keep N
        # constant while silently changing the contents. If the first id
        # differs, the sidecar is stale and must not be reused.
        first_new_id = documents[0].id if documents else None
        first_old_id = loaded[0].id if loaded else None
        if first_new_id != first_old_id:
            print(
                f"  Refusing: sidecar {sidecars['docs']} has the same N "
                f"({len(loaded)}) but a different first id than the new "
                f"corpus (old={first_old_id!r}, new={first_new_id!r}). "
                f"The sidecar is stale (likely a config change). Use "
                f"--force to rebuild from scratch."
            )
            sys.exit(1)
        documents = loaded
        print(f"  Resumed {len(documents)} dedup'd paragraphs from {sidecars['docs']}.")
        extend_n = 0
    else:
        # Either this is a fresh build, or --extend is set and the new N
        # differs from the existing sidecar. Decide which.
        documents = _dedupe_documents(documents)
        extend_n = 0
        if args.extend and sidecars["docs"].exists():
            # Load the existing sidecar WITHOUT the N check (we know it
            # doesn't match — that's the whole point of --extend) and see
            # how many rows we can reuse by ID prefix.
            old_docs = _load_docs_sidecar(sidecars["docs"], expected_n=None)
            # _load_docs_sidecar with expected_n=None validates type only;
            # if it's still None here, the file is corrupt and we just
            # fall through to a fresh build.
            if old_docs:
                n_prefix = _prefix_match_len(old_docs, documents)
                if n_prefix == 0:
                    print(
                        f"  --extend: sidecar {sidecars['docs']} has "
                        f"{len(old_docs)} paragraphs but the first id does "
                        f"not match the new dedup'd list. Refusing to "
                        f"extend — use --force to rebuild from scratch."
                    )
                    sys.exit(1)
                if n_prefix < len(old_docs):
                    # Sidecar has stale IDs in the middle; this is unusual
                    # and would corrupt the mmap if we naively reused the
                    # prefix. Refuse to extend.
                    print(
                        f"  --extend: sidecar ids match for the first "
                        f"{n_prefix} paragraphs, then diverge at index "
                        f"{n_prefix} (old={old_docs[n_prefix].id!r}, "
                        f"new={documents[n_prefix].id!r}). Refusing to "
                        f"extend — use --force to rebuild."
                    )
                    sys.exit(1)
                # OK: the sidecar is a strict prefix of the new docs.
                if n_prefix == len(documents):
                    # New N == old N and IDs all match → identical dataset.
                    # Treat as a normal resume (no extension needed).
                    extend_n = 0
                else:
                    extend_n = n_prefix
                    print(
                        f"  --extend: reusing {n_prefix} of "
                        f"{len(documents)} paragraphs from "
                        f"{sidecars['docs']} (the rest are new)."
                    )
        if extend_n == 0 and not (args.extend and sidecars["docs"].exists()):
            # Plain fresh-build announcement.
            print(
                f"Loaded {n_examples_per_split} examples from {args.dataset} "
                f"(splits: {args.split}); {len(documents)} unique paragraphs "
                f"after dedup."
            )
        _save_docs_sidecar(sidecars["docs"], documents)
        if extend_n == 0 and not (args.extend and sidecars["docs"].exists()):
            print(f"  Saved dedup'd paragraphs to {sidecars['docs']}.")

    # ---- BM25 tokenize (with sidecar resume / extend) ----
    bm25_data = _load_bm25_sidecar(sidecars["bm25"], expected_n=len(documents))
    if bm25_data is not None:
        tokenized_corpus, bm25_obj = bm25_data
        print(f"  Resumed BM25 from {sidecars['bm25']}.")
    else:
        # Either no sidecar (fresh), or --extend and the sidecar holds only
        # the first ``extend_n`` docs.
        if extend_n > 0:
            old_bm25 = _load_bm25_sidecar(sidecars["bm25"], expected_n=extend_n)
        else:
            old_bm25 = None
        if old_bm25 is not None:
            old_tokenized, _ = old_bm25
            new_only = documents[extend_n:]
            n_new = len(new_only)
            if args.bm25_workers > 1:
                print(
                    f"Tokenizing {n_new} new paragraphs with "
                    f"{args.bm25_workers} processes (skipping {extend_n} "
                    f"already-tokenized)..."
                )
            else:
                print(
                    f"Tokenizing {n_new} new paragraphs "
                    f"(skipping {extend_n} already-tokenized)..."
                )
            new_tokenized = tokenize_corpus(new_only, n_workers=args.bm25_workers)
            tokenized_corpus = old_tokenized + new_tokenized
            # BM25Okapi stats have to be recomputed for the full corpus.
            from rank_bm25 import BM25Okapi as _BM25Okapi
            print(f"  Rebuilding BM25Okapi over {len(tokenized_corpus)} docs...")
            bm25_obj = _BM25Okapi(tokenized_corpus)
            _save_bm25_sidecar(sidecars["bm25"], tokenized_corpus, bm25_obj)
            print(f"  Saved BM25 to {sidecars['bm25']}.")
        else:
            if args.bm25_workers > 1:
                print(
                    f"Tokenizing {len(documents)} paragraphs with "
                    f"{args.bm25_workers} processes..."
                )
            else:
                print(f"Tokenizing {len(documents)} paragraphs...")
            bm25 = BM25Index(documents, n_workers=args.bm25_workers)
            tokenized_corpus = bm25.tokenized_corpus
            bm25_obj = bm25.bm25
            _save_bm25_sidecar(sidecars["bm25"], tokenized_corpus, bm25_obj)
            print(f"  Saved BM25 to {sidecars['bm25']}.")
            del bm25  # release the BM25Okapi object; we already pickled it

    # ---- Embeddings (with mmap resume / extend) ----
    # Skipped for bm25-only builds (the sidecar files are still created if
    # a previous hybrid/dense build was run, so an upgrade to dense or
    # hybrid is just a re-run with the new --retrieval-mode).
    if args.retrieval_mode == "bm25":
        emb_arr = None
        print("  Skipping embedding (bm25-only build).")
    else:
        # Build the embed-text list (truncated) before opening the mmap so we
        # know N. Truncation is deterministic so a re-run produces the same
        # strings, which means the mmap rows are stable across resumes.
        MAX_CHARS = 8000
        texts = [f"{d.title}. {d.text}" for d in documents]
        truncated = [t if len(t) <= MAX_CHARS else t[:MAX_CHARS] for t in texts]
        n_truncated = sum(1 for o, t in zip(texts, truncated) if len(o) > MAX_CHARS)
        if n_truncated:
            print(f"  Truncated {n_truncated} over-long paragraphs to {MAX_CHARS} chars.")

        # Probe embedding dim (one Ollama call) if we don't have an mmap yet.
        # Once the mmap exists, we read dim from its shape — no extra call.
        if sidecars["emb"].exists():
            try:
                existing_shape = np.load(sidecars["emb"], mmap_mode="r").shape
                dim = int(existing_shape[1])
            except Exception:
                dim = None
        else:
            dim = None
        if dim is None:
            probe_text = truncated[0] if truncated else "x"
            probe = Embedder().embed([probe_text])
            dim = int(probe.shape[1])

        if extend_n > 0 and sidecars["emb"].exists():
            # Resize the existing mmap to fit the new corpus, preserving
            # the first ``extend_n`` rows that we know match.
            emb_arr, start_count = _extend_emb_mmap(
                sidecars["emb"], sidecars["emb_count"],
                n_old=extend_n, n_new=len(documents), dim=dim,
            )
        else:
            emb_arr, start_count = _open_emb_mmap(
                sidecars["emb"], sidecars["emb_count"],
                n=len(documents), dim=dim,
            )

        if start_count >= len(documents):
            print(f"  Resumed embeddings from {sidecars['emb']} (all {len(documents)} done).")
        else:
            if start_count > 0:
                print(
                    f"Resuming embedding from row {start_count}/{len(documents)} "
                    f"(workers={args.workers}, batch=64)..."
                )
            else:
                print(
                    f"Embedding {len(documents)} paragraphs "
                    f"(workers={args.workers}, batch=64)..."
                )
            _embed_to_mmap(
                truncated, emb_arr, sidecars["emb_count"], start_count,
                batch_size=64, max_workers=args.workers,
            )

    # ---- Assemble final index from sidecars ----
    bm25 = BM25Index.__new__(BM25Index)
    bm25.documents = documents
    bm25.tokenized_corpus = tokenized_corpus
    bm25.bm25 = bm25_obj

    if args.retrieval_mode == "bm25":
        index = bm25
    elif args.retrieval_mode == "dense":
        # np.asarray(emb_arr) materializes the mmap into a regular ndarray.
        index = DenseIndex(documents, np.asarray(emb_arr))
    elif args.retrieval_mode == "hybrid":
        dense = DenseIndex(documents, np.asarray(emb_arr))
        index = HybridIndex(bm25, dense)
    else:
        raise ValueError(f"Unknown retrieval_mode={args.retrieval_mode!r}")

    index.save(args.output)
    print(f"Built {type(index).__name__} with {len(index.documents)} documents.")
    print(f"Saved index to {args.output}.")
    print(
        f"(Sidecars kept at {sidecars['docs'].parent}/* for future resume. "
        f"Delete the sidecars to fully rebuild from scratch.)"
    )


if __name__ == "__main__":
    main()
