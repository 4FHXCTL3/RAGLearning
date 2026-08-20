from __future__ import annotations

from pathlib import Path

import numpy as np

from rag_learning.environment import NUM_ACTIONS

STATE_DIM = 4
HIDDEN_DIM = 64


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.sum(e)


class PolicyNetwork:
    """Minimal MLP policy for GRPO (framework module 2).

    Architecture: 4 (state) -> 64 (tanh) -> 3 (action logits) -> softmax.
    Pure NumPy, no autograd: forward caches activations and we backprop the
    GRPO clipped objective by hand. No value network (GRPO is critic-free).
    """

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        # Small init keeps early logits near-uniform (exploratory policy).
        self.w1 = rng.normal(0, 0.1, size=(STATE_DIM, HIDDEN_DIM)).astype(np.float64)
        self.b1 = np.zeros(HIDDEN_DIM, dtype=np.float64)
        self.w2 = rng.normal(0, 0.1, size=(HIDDEN_DIM, NUM_ACTIONS)).astype(np.float64)
        self.b2 = np.zeros(NUM_ACTIONS, dtype=np.float64)

    # --- forward ---------------------------------------------------------

    def _forward_cache(self, state: np.ndarray) -> dict:
        x = state.astype(np.float64)
        z1 = x @ self.w1 + self.b1
        h = np.tanh(z1)
        logits = h @ self.w2 + self.b2
        probs = softmax(logits)
        return {"x": x, "z1": z1, "h": h, "logits": logits, "probs": probs}

    def action_probs(self, state: np.ndarray) -> np.ndarray:
        return self._forward_cache(state)["probs"]

    def sample_action(self, state: np.ndarray, rng: np.random.Generator) -> tuple[int, float]:
        probs = self.action_probs(state)
        action = int(rng.choice(NUM_ACTIONS, p=probs))
        return action, float(probs[action])

    # --- gradient of log pi(a|s) w.r.t. all params ----------------------

    def _grad_log_prob(self, cache: dict, action: int) -> dict:
        """Gradient of log pi(action | state) for one sample.

        d log p_a / d logits = (onehot(a) - probs). Backprop through the MLP.
        """
        probs = cache["probs"]
        dlogits = -probs.copy()
        dlogits[action] += 1.0  # onehot(a) - probs

        h = cache["h"]
        x = cache["x"]
        gw2 = np.outer(h, dlogits)                 # (HIDDEN, ACTIONS)
        gb2 = dlogits
        dh = self.w2 @ dlogits                     # (HIDDEN,)
        dz1 = dh * (1.0 - h * h)                    # tanh'
        gw1 = np.outer(x, dz1)                      # (STATE, HIDDEN)
        gb1 = dz1
        return {"w1": gw1, "b1": gb1, "w2": gw2, "b2": gb2}

    # --- GRPO update -----------------------------------------------------

    def grpo_update(
        self,
        samples: list[dict],
        advantages: np.ndarray,
        lr: float = 0.05,
        clip_eps: float = 0.2,
    ) -> float:
        """One GRPO gradient step over a group of samples.

        Each sample: {"state": vec, "action": int, "old_prob": float}.
        advantages: per-sample group-relative advantage A_i.

        Objective (maximize): mean_i min(ratio_i * A_i, clip(ratio_i,
        1-eps, 1+eps) * A_i), where ratio_i = pi_new(a_i|s_i)/pi_old(a_i|s_i).
        We ascend, so params += lr * grad(objective).
        """
        grads = {k: np.zeros_like(v) for k, v in self._params().items()}
        total_obj = 0.0
        n = len(samples)
        if n == 0:
            return 0.0

        for sample, adv in zip(samples, advantages):
            cache = self._forward_cache(sample["state"])
            new_prob = cache["probs"][sample["action"]]
            old_prob = max(sample["old_prob"], 1e-8)
            ratio = new_prob / old_prob

            # Clipped surrogate: objective = min(unclipped, clipped).
            unclipped = ratio * adv
            clipped = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv
            total_obj += min(unclipped, clipped)

            # The clipped branch is constant in theta (zero gradient); gradient
            # flows only when the unclipped term is the binding (smaller) one.
            # d(ratio*A)/dtheta = A * ratio * d log pi / dtheta.
            if unclipped <= clipped:
                glp = self._grad_log_prob(cache, sample["action"])
                scale = ratio * adv
                for k in grads:
                    grads[k] += scale * glp[k]

        # Ascent step on the mean objective.
        for k, param in self._params().items():
            param += lr * grads[k] / n

        return float(total_obj / n)

    # --- persistence -----------------------------------------------------

    def _params(self) -> dict:
        return {"w1": self.w1, "b1": self.b1, "w2": self.w2, "b2": self.b2}

    def params_snapshot(self) -> dict:
        """Deep copy of current parameters.

        Used by the trainer to measure how much a GRPO step actually moved the
        weights (max abs delta). The GRPO objective is ~0 by construction
        (advantages are normalized to mean 0), so weight delta is the reliable
        "is it learning" signal.
        """
        return {k: v.copy() for k, v in self._params().items()}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)

    @classmethod
    def load(cls, path: str | Path) -> "PolicyNetwork":
        data = np.load(Path(path))
        net = cls()
        net.w1, net.b1 = data["w1"], data["b1"]
        net.w2, net.b2 = data["w2"], data["b2"]
        return net


class RunningBaseline:
    """EMA baseline for advantage computation.

    Tracks a running average of recent group returns so that advantage
    signals are non-zero even when a single group's returns have tiny
    variance (the root cause of the previous near-zero gradient issue).
    """

    def __init__(self, alpha: float = 0.9):
        self.alpha = alpha
        self.value: float = 0.0
        self.initialized = False

    def update(self, returns: list[float]) -> None:
        mean_return = float(np.mean(returns))
        if not self.initialized:
            self.value = mean_return
            self.initialized = True
        else:
            self.value = self.alpha * self.value + (1.0 - self.alpha) * mean_return

    def advantages(self, returns: list[float]) -> np.ndarray:
        """A_i = (R_i - baseline) / max(std, 1e-8).

        Normalising by the group std keeps advantages dimensionless so the
        gradient scale is stable across groups with very different return
        magnitudes.
        """
        r = np.asarray(returns, dtype=np.float64)
        adv = r - self.value
        std = r.std()
        if std > 1e-8:
            adv = adv / std
        return adv


def group_advantages(rewards: list[float]) -> np.ndarray:
    """Legacy group-relative advantage (kept for backward compat).

    Prefer RunningBaseline for actual training.
    """
    r = np.asarray(rewards, dtype=np.float64)
    std = r.std()
    if std < 1e-8:
        return np.zeros_like(r)
    return (r - r.mean()) / std
