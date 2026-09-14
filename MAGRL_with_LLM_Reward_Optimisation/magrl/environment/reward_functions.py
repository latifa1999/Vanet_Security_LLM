"""
reward_functions.py

Classes:
  AdaptivePenaltyTracker  — tracks rolling FP/FN rates, scales penalties
  RewardNormalizer        — running normalization of reward values
  RewardFunction          — base class
  DynamicReward           — wraps LLM-generated compute_reward()
"""

from collections import deque
from typing import Dict, Optional

import numpy as np


# ── Adaptive Penalty Tracker ───────────────────────────────────────────────────


class AdaptivePenaltyTracker:
    """
    Tracks rolling FP/FN rates and scales penalties automatically.
    Applied on top of ALL LLM-generated reward functions.
    """

    def __init__(
        self,
        window_size: int   = 100,
        beta_base:   float = 1.0,
        gamma_base:  float = 1.0,
        max_scale:   float = 3.0,
        verbose:     bool  = True,
    ):
        self.beta_base     = beta_base
        self.gamma_base    = gamma_base
        self.max_scale     = max_scale
        self.verbose       = verbose
        self.fp_window     = deque(maxlen=window_size)
        self.fn_window     = deque(maxlen=window_size)
        self.current_beta  = beta_base
        self.current_gamma = gamma_base
        self._last_beta    = beta_base
        self._last_gamma   = gamma_base

    def update(self, fp_count: int, fn_count: int, total: int):
        if total == 0:
            return
        self.fp_window.append(fp_count / total)
        self.fn_window.append(fn_count / total)
        avg_fp = float(np.mean(self.fp_window))
        avg_fn = float(np.mean(self.fn_window))
        self.current_beta  = min(self.beta_base  * (1 + 2 * avg_fp), self.max_scale)
        self.current_gamma = min(self.gamma_base * (1 + 2 * avg_fn), self.max_scale)
        if self.verbose and len(self.fp_window) >= 10:
            if (abs(self.current_beta  - self._last_beta)  > 0.2 or
                    abs(self.current_gamma - self._last_gamma) > 0.2):
                print(f"[ADAPTIVE] β:{self._last_beta:.2f}→{self.current_beta:.2f}  "
                      f"γ:{self._last_gamma:.2f}→{self.current_gamma:.2f}  "
                      f"(FP={avg_fp:.3f}  FN={avg_fn:.3f})")
                self._last_beta  = self.current_beta
                self._last_gamma = self.current_gamma

    def scale_rewards_inplace(self, rewards: np.ndarray,
                               fp_mask: np.ndarray, fn_mask: np.ndarray):
        if self.current_beta != 1.0:
            rewards[fp_mask] *= self.current_beta
        if self.current_gamma != 1.0:
            rewards[fn_mask] *= self.current_gamma

    def get_scales(self) -> Dict:
        return {
            "beta":        self.current_beta,
            "gamma":       self.current_gamma,
            "avg_fp_rate": float(np.mean(self.fp_window)) if self.fp_window else 0.0,
            "avg_fn_rate": float(np.mean(self.fn_window)) if self.fn_window else 0.0,
        }


# ── Running Normalizer ─────────────────────────────────────────────────────────


class RewardNormalizer:
    def __init__(self, clip: float = 1.0):
        self.clip = clip
        self.mean = 0.0
        self.var  = 1.0
        self.n    = 0

    def normalize(self, rewards: np.ndarray) -> np.ndarray:
        self.n += len(rewards)
        alpha = 0.01
        bm = float(rewards.mean())
        bv = float(rewards.var())
        if self.n == len(rewards):
            self.mean, self.var = bm, max(bv, 1e-8)
        else:
            self.mean = (1 - alpha) * self.mean + alpha * bm
            self.var  = (1 - alpha) * self.var  + alpha * bv
        std = np.sqrt(self.var) + 1e-8
        return np.clip((rewards - self.mean) / std,
                       -self.clip, self.clip).astype(np.float32)


# ── Base class ─────────────────────────────────────────────────────────────────


class RewardFunction:
    def __init__(self, name: str = "R"):
        self.name       = name
        self.normalizer = RewardNormalizer()

    def compute_batch_np(
        self,
        actions:  np.ndarray,
        labels:   np.ndarray,
        preds:    np.ndarray,
        confs:    np.ndarray,
        adaptive: Optional[AdaptivePenaltyTracker] = None,
    ) -> Dict:
        raise NotImplementedError

    def compute_batch(
        self,
        actions,
        true_labels,
        ml_predictions,
        ml_confidences=None,
        normalize:        bool = True,
        adaptive_tracker: Optional[AdaptivePenaltyTracker] = None,
    ) -> Dict:
        a = np.asarray(actions,        dtype=np.int32)
        l = np.asarray(true_labels,    dtype=np.int32)
        p = np.asarray(ml_predictions, dtype=np.int32)
        c = np.asarray(
            ml_confidences if ml_confidences is not None else [0.5] * len(a),
            dtype=np.float32,
        )
        return self.compute_batch_np(a, l, p, c, adaptive_tracker)


# ── Dynamic (LLM-generated) reward ────────────────────────────────────────────


class DynamicReward(RewardFunction):
    """
    Wraps any LLM-generated compute_reward() function.
    Adaptive penalty scaling applied on top automatically.

    Expected LLM function signature:
      def compute_reward(action, true_label, ml_prediction, ml_confidence=0.5)
        -> (reward: float, category: str)
      category must be one of: 'TP', 'TN', 'FP', 'FN'
    """

    def __init__(self, name: str, code: str):
        super().__init__(name)
        self.code = code
        self._fn  = self._compile(code)

    @staticmethod
    def _compile(code: str):
        try:
            ns = {"np": np}
            exec(compile(code, "<llm>", "exec"), ns)
            fn = ns.get("compute_reward")
            if fn is None:
                raise ValueError("No compute_reward function found")
            # Smoke-test all 4 categories
            for a, l, p, c in [(0,1,1,0.8),(1,0,0,0.9),(0,0,0,0.7),(1,1,1,0.6)]:
                r, cat = fn(a, l, p, c)
                assert isinstance(r, (int, float)), \
                    f"reward must be numeric, got {type(r)}"
                assert cat in ("TP","TN","FP","FN"), \
                    f"category must be TP/TN/FP/FN, got '{cat}'"
            return fn
        except Exception as e:
            print(f"[REWARD] Compile/validate failed: {e}")
            return None

    def is_valid(self) -> bool:
        return self._fn is not None

    def compute_batch_np(self, actions, labels, preds, confs, adaptive=None):
        if self._fn is None:
            raise RuntimeError(
                f"[REWARD] {self.name} has invalid code — cannot compute rewards")

        n       = len(actions)
        rewards = np.zeros(n, dtype=np.float32)
        fp_mask = np.zeros(n, dtype=bool)
        fn_mask = np.zeros(n, dtype=bool)

        for i in range(n):
            r, cat = self._fn(int(actions[i]), int(labels[i]),
                              int(preds[i]),   float(confs[i]))
            rewards[i] = float(r)
            if cat == "FP":
                fp_mask[i] = True
            elif cat == "FN":
                fn_mask[i] = True

        if adaptive is not None:
            adaptive.scale_rewards_inplace(rewards, fp_mask, fn_mask)

        norm = self.normalizer.normalize(rewards)

        tp = int(((actions == 0) & (labels == 1)).sum())
        tn = int(((actions == 1) & (labels == 0)).sum())
        fp = int(fp_mask.sum())
        fn = int(fn_mask.sum())

        if adaptive is not None:
            adaptive.update(fp, fn, n)

        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-8)

        result = {
            "rewards":     norm.tolist(),
            "mean_reward": float(norm.mean()),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "f1": f1, "precision": prec, "recall": rec,
        }
        if adaptive is not None:
            result["penalty_scales"] = adaptive.get_scales()
        return result