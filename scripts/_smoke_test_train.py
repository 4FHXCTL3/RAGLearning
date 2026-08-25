"""Smoke test: ensure the optimized training pipeline still runs end-to-end
without errors, with a mocked LLM/Embedder so we don't need Ollama.

Run: python scripts/_smoke_test_train.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class _MockEmbedder:
    def __init__(self, dim: int = 64, seed: int = 0):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self._cache: dict = {}

    def embed(self, texts):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        out = []
        for t in items:
            if t not in self._cache:
                v = self.rng.normal(0, 1, self.dim).astype(np.float32)
                v /= np.linalg.norm(v) + 1e-12
                self._cache[t] = v
            out.append(self._cache[t])
        return np.stack(out) if len(out) > 1 else np.array(out).reshape(1, -1)


class _MockDoc:
    def __init__(self, title, text):
        self.title = title
        self.text = text


class _MockIndex:
    def __init__(self, n_docs: int = 30):
        self.documents = [
            _MockDoc(f"Title {i}", f"Paragraph {i} about topic {i}.")
            for i in range(n_docs)
        ]

    def search(self, query, top_k: int = 5):
        return [(d, 1.0) for d in self.documents[:top_k]]


class _MockLLM:
    def __init__(self):
        self.model = "mock"
        self.n_calls = 0

    def generate(self, prompt, system_prompt=None, temperature=None):
        self.n_calls += 1
        return "Paris"

    def generate_batch(self, prompts, system_prompts=None, temperatures=None):
        self.n_calls += len(prompts)
        answers = ["Paris", "London", "Berlin"]
        return [answers[(self.n_calls + i) % 3] for i in range(len(prompts))]


def main():
    print("=" * 60)
    print("Smoke test: end-to-end training pipeline (mocked I/O)")
    print("=" * 60)

    from rag_learning.policy import PolicyNetwork
    from rag_learning.state import StateEncoder
    from rag_learning.environment import RAGEnv, RewardWeights
    from rag_learning.rag import build_prompt, clean_answer

    llm = _MockLLM()
    encoder = StateEncoder(embedder=_MockEmbedder(dim=64, seed=1))
    env = RAGEnv(
        index=_MockIndex(n_docs=30),
        llm=llm,
        encoder=encoder,
        weights=RewardWeights(),
        initial_k=5,
        k_increment=5,
        max_retrievals=2,
        group_size=3,
        sample_temperature=0.7,
        retrieval_mode="bm25",
    )
    policy = PolicyNetwork(seed=0)
    rng = np.random.default_rng(42)

    n_questions = 10
    print(f"Running {n_questions} questions through the optimized training loop...")
    from rag_learning.policy import RunningBaseline
    baseline = RunningBaseline(alpha=0.9)
    total_max_dw = 0.0
    total_obj = 0.0
    n_actions = {"stop": 0, "retrieve": 0, "reject": 0}

    for q_idx in range(n_questions):
        question = f"q{q_idx}: What is the capital of country {q_idx}?"
        golds = ["Paris", "London"]
        gold_titles_set = {"title 0", "title 1"}

        init_state = env.reset(question, golds, gold_titles_set)
        init_snap = env.snapshot()
        init_state_vec = init_state.to_vector()
        shared_cache = policy._forward_cache(init_state_vec)
        shared_probs = shared_cache["probs"]

        group_samples = []
        for _ in range(3):
            env.restore(init_snap)
            first_action, old_prob = policy.sample_action_from_probs(shared_probs, rng)
            action = first_action
            total_return = 0.0
            done = False
            while not done:
                step_result = env.step(action)
                total_return += step_result.reward
                done = step_result.done
                if not done:
                    action, _p, _c = policy.sample_action(step_result.state.to_vector(), rng)
            if first_action == 0: n_actions["stop"] += 1
            elif first_action == 1: n_actions["retrieve"] += 1
            else: n_actions["reject"] += 1
            group_samples.append({
                "state": init_state_vec, "action": first_action,
                "old_prob": old_prob, "return": total_return, "cache": shared_cache,
            })

        returns = [s["return"] for s in group_samples]
        baseline.update(returns)
        advantages = baseline.advantages(returns)
        obj, max_dw = policy.grpo_update(group_samples, advantages, lr=0.05, clip_eps=0.2)
        total_obj += obj
        total_max_dw += max_dw
        print(f"  q{q_idx}: actions={n_actions}  obj={obj:+.4f}  max_dW={max_dw:.2e}  "
              f"returns={[f'{r:+.3f}' for r in returns]}")

    print()
    print(f"Total LLM calls: {llm.n_calls}")
    print(f"Action distribution: {n_actions}")
    print(f"Mean objective: {total_obj / n_questions:+.4f}")
    print(f"Mean max_dW:    {total_max_dw / n_questions:.2e}")
    assert llm.n_calls > 0, "LLM was never called"
    assert sum(n_actions.values()) == n_questions * 3, "Action count mismatch"
    assert total_max_dw > 0, "Weights never moved (gradient broken)"

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "policy.npz"
        policy.save(path)
        loaded = PolicyNetwork.load(path)
        assert np.allclose(policy.w1, loaded.w1)
        assert np.allclose(policy.w2, loaded.w2)
        print(f"Policy save/load roundtrip OK -> {path}")

    ctx = [_MockDoc("T", "Paris is the capital of France.")]
    p = build_prompt("What is the capital of France?", ctx)
    assert "Paris" in p
    assert clean_answer("<think>...\n</think>\nThe answer is Paris.") == "Paris"
    assert clean_answer("London.") == "London"
    print("build_prompt / clean_answer OK")

    print()
    print("=" * 60)
    print("SMOKE TEST PASSED - all optimizations compatible end-to-end")
    print("=" * 60)


if __name__ == "__main__":
    main()
