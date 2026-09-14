"""
llm_reward_optimizer.py — LLM-based reward optimizer for MAPPO VANET security.

TWO-PHASE STRATEGY
------------------
Phase 1 — ZERO-SHOT EXPLORATION  (starts at episode 0)
  - R1 generated at episode 0 from environment description + AD metrics
  - Continues generating until eval_f1 consistently beats AD_F1

Phase 2 — SELECTIVE REFINEMENT  (triggered when eval_f1 >= AD_F1 * threshold)
  - LLM receives best reward code + current metrics
  - Targeted improvements only
"""

import hashlib
import json
import os
import re
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

MODE_ZERO_SHOT  = "zero_shot"
MODE_REFINEMENT = "refinement"


class LLMRewardOptimizer:

    def __init__(
        self,
        hf_token:             str = "TOKEN",  # ← replace with your HuggingFace token
        refinement_interval:  int   = 3000,
        output_dir:           Optional[str] = None,
        verbose:              bool  = True,
        max_versions:         int   = 6,
        eval_window:          int   = 500,
        refinement_threshold: float = 1.005,
    ):
        if not hf_token:
            raise ValueError("[LLM] hf_token is required — get one at huggingface.co/settings/tokens")
        self.hf_token             = hf_token
        self.refinement_interval  = refinement_interval
        self.output_dir           = output_dir
        self.verbose              = verbose
        self.max_versions         = max_versions
        self.eval_window          = eval_window
        self.refinement_threshold = refinement_threshold

        self.current_version      = 0
        self.current_version_name = "none"
        self.reward_history:    List[Dict] = []
        self.qualified_rewards: List[Dict] = []
        self.metrics_history:   List[Dict] = []

        self._f1_window: deque = deque(maxlen=eval_window)
        self.eval_f1: float    = 0.0

        self.mode:                str            = MODE_ZERO_SHOT
        self.anchor_reward:       Optional[Dict] = None
        self.zero_shot_attempts:  int            = 0
        self.refinement_attempts: int            = 0
        self._code_hashes:        set            = set()

        if verbose:
            print(f"\n[LLM] Reward Optimizer — Qwen2.5-Coder-32B via HuggingFace")
            print(f"      Model     : Qwen/Qwen2.5-Coder-32B-Instruct:nscale")
            print(f"      Interval  : every {refinement_interval} episodes")
            print(f"      Window    : last {eval_window} episodes")
            print(f"      Threshold : eval_F1 >= AD_F1 * {refinement_threshold}")
            print(f"      Max vers  : {max_versions}")

    # ── Rolling window ─────────────────────────────────────────────────────────

    def update(self, raw_f1: float) -> float:
        self._f1_window.append(raw_f1)
        self.eval_f1 = float(np.mean(self._f1_window))
        return self.eval_f1

    # ── Phase transition ───────────────────────────────────────────────────────

    def _check_phase_transition(self, ad_f1: float):
        threshold = ad_f1 * self.refinement_threshold
        if self.mode == MODE_ZERO_SHOT and self.eval_f1 >= threshold:
            self.mode = MODE_REFINEMENT
            self.anchor_reward = {
                "name":    self.current_version_name,
                "code":    self._get_current_code(),
                "eval_f1": self.eval_f1,
            }
            if self.verbose:
                print(f"\n[LLM] ZERO-SHOT → REFINEMENT")
                print(f"      Best reward : {self.anchor_reward['name']}")
                print(f"      eval_F1     : {self.eval_f1:.4f} >= {threshold:.4f}")

    def _get_current_code(self) -> str:
        for rv in reversed(self.reward_history):
            if rv["name"] == self.current_version_name:
                return rv.get("code", "")
        return ""

    # ── Trigger decision ───────────────────────────────────────────────────────

    def should_refine(self, episode: int, metrics: Dict) -> bool:
        self.metrics_history.append({**metrics, "episode": episode})
        ad_f1 = metrics.get("ml_baseline_f1", 0)
        self._check_phase_transition(ad_f1)

        if self.current_version >= self.max_versions:
            if self.verbose:
                print(f"[LLM] Ep {episode}: SKIP — max versions ({self.max_versions}) reached")
            return False

        if episode == 0:
            if self.verbose:
                print(f"\n[LLM] Ep 0: TRIGGER — zero-shot cold start")
            return True

        if episode % self.refinement_interval != 0:
            return False

        if len(self._f1_window) < self.eval_window:
            if self.verbose:
                print(f"[LLM] Ep {episode}: SKIP — window not full "
                      f"({len(self._f1_window)}/{self.eval_window})")
            return False

        threshold = ad_f1 * self.refinement_threshold
        if self.eval_f1 >= threshold and self.mode == MODE_REFINEMENT:
            if self.verbose:
                print(f"\n[LLM] Ep {episode}: SKIP — eval_F1={self.eval_f1:.4f} "
                      f">= {threshold:.4f} — agent good enough")
            return False

        if self.verbose:
            print(f"\n[LLM] Ep {episode}: TRIGGER [{self.mode.upper()}] — "
                  f"eval_F1={self.eval_f1:.4f} < {threshold:.4f}")
        return True

    # ── Generate new reward ────────────────────────────────────────────────────

    def generate_new_reward(self, metrics: Dict, ml_baseline: Dict) -> Optional[str]:
        temp   = 0.8 if self.mode == MODE_ZERO_SHOT else 0.4
        prompt = self._build_prompt(metrics, ml_baseline)

        if self.verbose:
            print(f"\n[LLM] Calling Qwen2.5-Coder-32B [{self.mode.upper()}] T={temp} ...")

        resp = self._call_hf(prompt, temp)
        if not resp:
            if self.verbose:
                print("[LLM] API call failed — staying on current reward until next interval")
            return None

        code = self._extract_code(resp)
        if not code:
            if self.verbose:
                print("[LLM] Could not extract Python function from response")
                print(f"[LLM] Raw response was:\n{resp[:500]}")
            return None

        if not self._validate(code):
            if self.verbose:
                print("[LLM] Generated code failed validation — staying on current reward")
                print("[LLM] Invalid code was:")
                for line in code.split("\n"):
                    print(f"[LLM]   {line}")
            return None

        h = self._hash_code(code)
        if h in self._code_hashes:
            if self.verbose:
                print("[LLM] SKIP — generated code is identical to a previous version")
            return None

        self._code_hashes.add(h)
        self.current_version += 1
        name = f"R{self.current_version}"

        if self.mode == MODE_ZERO_SHOT:
            self.zero_shot_attempts += 1
        else:
            self.refinement_attempts += 1

        ad_f1     = ml_baseline.get("f1", 0)
        qualifies = self.eval_f1 >= ad_f1

        info = {
            "name":               name,
            "code":               code,
            "code_hash":          h,
            "phase":              self.mode,
            "qualifies":          qualifies,
            "created_at":         datetime.now().isoformat(),
            "created_at_episode": metrics.get("episode", 0),
            "metrics_at_creation": {
                "raw_f1":  metrics.get("f1_score", 0),
                "eval_f1": self.eval_f1,
                "fp":      metrics.get("total_fp", 0),
                "fn":      metrics.get("total_fn", 0),
                "ad_f1":   ad_f1,
            },
        }
        self.reward_history.append(info)
        if qualifies:
            self.qualified_rewards.append(info)
            if self.verbose:
                print(f"[LLM] {name} beats AD baseline "
                      f"(eval_F1={self.eval_f1:.4f} >= {ad_f1:.4f})")

        self.current_version_name = name
        if self.output_dir:
            self._save_reward(info)
            self._save_history()
            self._log_reward_to_txt(info)

        if self.verbose:
            print(f"[LLM] {name} generated [{self.mode.upper()}]  qualifies={qualifies}")
            print("[LLM] --- CODE ---")
            for line in code.split("\n"):
                print(f"[LLM]   {line}")
            print("[LLM] ----------")

        return code

    # ── HuggingFace API ─────────────────────────────────────────────────────────

    def _call_hf(self, prompt: str, temperature: float) -> Optional[str]:
        try:
            import requests
            r = requests.post(
                "https://router.huggingface.co/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.hf_token}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":       "Qwen/Qwen2.5-Coder-32B-Instruct",
                    "messages":    [{"role": "user", "content": prompt}],
                    "max_tokens":  1500,
                    "temperature": temperature,
                },
                timeout=180,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            # Print full error so user knows exactly what went wrong
            print(f"[LLM] HTTP {r.status_code} error:")
            print(f"[LLM]   {r.text[:500]}")
            if r.status_code == 401:
                print("[LLM] → Token expired or invalid. "
                      "Get a new one at huggingface.co/settings/tokens")
            elif r.status_code == 429:
                print("[LLM] → Rate limit reached. Wait a few minutes.")
            elif r.status_code == 503:
                print("[LLM] → Model is loading. Retry in ~30s.")
        except Exception as e:
            print(f"[LLM] Request exception: {e}")
        return None

    # ── Prompt builder ─────────────────────────────────────────────────────────

    def _build_prompt(self, metrics: Dict, ml_baseline: Dict) -> str:
        if self.mode == MODE_ZERO_SHOT:
            return self._zero_shot_prompt(metrics, ml_baseline)
        return self._refinement_prompt(metrics, ml_baseline)

    def _zero_shot_prompt(self, metrics: Dict, ml_baseline: Dict) -> str:
        ad_f1   = ml_baseline.get("f1", 0)
        eval_f1 = self.eval_f1
        episode = metrics.get("episode", 0)
        fp      = metrics.get("total_fp", 0)
        fn      = metrics.get("total_fn", 0)
        sc      = metrics.get("penalty_scales", {})

        if episode == 0:
            perf = (f"  AD baseline F1 : {ad_f1:.4f}  "
                    f"(FP={ml_baseline.get('fp',0)}, FN={ml_baseline.get('fn',0)})\n"
                    f"  RL metrics     : not available yet (episode 0)\n"
                    f"  GOAL           : beat AD_F1={ad_f1:.4f}\n")
        else:
            perf = (f"  AD baseline F1 : {ad_f1:.4f}\n"
                    f"  RL eval F1     : {eval_f1:.4f}  "
                    f"(gap={ad_f1 - eval_f1:+.4f})\n"
                    f"  Recent errors  : FP={fp}  FN={fn}\n"
                    f"  GOAL           : eval_F1 > AD_F1 ({ad_f1:.4f})\n")

        penalty_note = ""
        if sc:
            penalty_note = (
                f"\nADAPTIVE PENALTIES (do NOT replicate — applied automatically):\n"
                f"  beta(FP)={sc.get('beta',1):.2f}  gamma(FN)={sc.get('gamma',1):.2f}\n"
            )

        prev = ""
        if self.zero_shot_attempts > 0:
            prev = (f"\nPREVIOUS {self.zero_shot_attempts} ATTEMPTS FAILED TO BEAT AD.\n"
                    f"Each attempt got stuck around eval_F1~{eval_f1:.3f}.\n"
                    f"You MUST try a STRUCTURALLY DIFFERENT approach.\n"
                    f"Consider: asymmetric rewards, confidence-gated bonuses, "
                    f"correction-focused design, or non-linear scaling.\n")

        patterns = [
            "asymmetric: weight FN correction 3x more than FP correction",
            "confidence-gated: reward magnitude scales with |conf - 0.5| * 2",
            "correction-bonus-heavy: give large bonus when agent corrects AD error",
            "threshold-based: full reward only when confidence > 0.7, else small",
            "f1-proxy: reward shaped to approximate the F1 score directly",
            "margin-based: reward proportional to distance from decision boundary",
            "recall-focused: penalise FN twice as hard as FP",
            "precision-focused: penalise FP twice as hard as FN",
            "agreement-bonus: extra reward when agent agrees with confident detector",
            "asymmetric with decay: strong early penalties, softer correction signal later",
        ]
        pattern = patterns[self.zero_shot_attempts % len(patterns)]

        issues = self._identify_issues(metrics, ml_baseline)

        return (
            "You are an RL expert designing reward functions for VANET security.\n\n"
            "ENVIRONMENT:\n"
            "  Vehicular Ad-hoc Network — malicious vehicles inject false messages.\n"
            "  MAPPO: each vehicle = one RL agent. Shared actor, centralized critic.\n\n"
            "AGENT TASK — POST-DETECTION CORRECTOR:\n"
            "  An ML Anomaly Detector (AD) already classifies each V2V message.\n"
            "  The RL agent CORRECTS the AD's mistakes (FP and FN).\n"
            "  action=0 → PRUNE (agent says malicious)\n"
            "  action=1 → KEEP  (agent says normal)\n\n"
            "INPUTS TO compute_reward():\n"
            "  action        : 0=PRUNE  1=KEEP\n"
            "  true_label    : 0=normal  1=malicious\n"
            "  ml_prediction : AD prediction (0 or 1)\n"
            "  ml_confidence : Prob_Malicious (0.0–1.0)\n\n"
            "CATEGORIES:\n"
            "  TP: action=0 & true=1  → correctly blocked malicious\n"
            "  TN: action=1 & true=0  → correctly kept normal\n"
            "  FP: action=0 & true=0  → wrongly blocked normal  (bad)\n"
            "  FN: action=1 & true=1  → missed malicious  (very bad)\n\n"
            "AD CORRECTION:\n"
            "  ad_wrong = (ml_prediction != true_label)\n"
            "  When agent is correct AND ad_wrong → agent corrected AD → give bonus\n\n"
            "CURRENT PERFORMANCE:\n"
            f"{perf}"
            f"{penalty_note}"
            f"\nPROBLEMS TO FIX:\n{issues}\n"
            f"{prev}\n"
            f"DESIGN PATTERN TO USE THIS TIME: {pattern}\n\n"
            "STRICT RULES:\n"
            "  1. Signature: def compute_reward(action, true_label, ml_prediction, ml_confidence=0.5)\n"
            "  2. Returns: (reward: float, category: str)\n"
            "  3. category MUST be exactly one of: 'TP', 'TN', 'FP', 'FN'\n"
            "  4. reward MUST be in range [-2.0, +2.0] — always clip at end:\n"
            "     reward = max(-2.0, min(2.0, reward))\n"
            "  5. Do NOT add FP/FN penalty scaling (handled externally)\n"
            "  6. CRITICAL: category depends ONLY on action+true_label, NEVER on confidence.\n"
            "     Always assign category first:\n"
            "       if action==0 and true_label==1: category='TP'\n"
            "       elif action==1 and true_label==0: category='TN'\n"
            "       elif action==0 and true_label==0: category='FP'\n"
            "       else: category='FN'\n"
            "     Then compute reward using confidence if desired.\n\n"
            "Return ONLY the Python function inside a ```python block. "
            "No explanation, no other text.\n\n"
            "```python\n"
            "def compute_reward(action, true_label, ml_prediction, ml_confidence=0.5):\n"
        )

    def _refinement_prompt(self, metrics: Dict, ml_baseline: Dict) -> str:
        ad_f1   = ml_baseline.get("f1", 0)
        eval_f1 = self.eval_f1
        fp      = metrics.get("total_fp", 0)
        fn      = metrics.get("total_fn", 0)

        anchor_block = ""
        if self.anchor_reward:
            anchor_block = (
                f"\nBEST REWARD SO FAR ({self.anchor_reward['name']}, "
                f"eval_F1={self.anchor_reward['eval_f1']:.4f}):\n"
                f"```python\n{self.anchor_reward['code']}\n```\n"
                "Make TARGETED improvements only — do not redesign from scratch.\n"
            )

        issues = self._identify_issues(metrics, ml_baseline)

        return (
            "You are an RL expert refining a reward function for VANET security.\n\n"
            "CONTEXT: MAPPO agents correct ML Anomaly Detector errors.\n"
            "  action=0=PRUNE, action=1=KEEP\n"
            "  TP=correct prune, TN=correct keep, FP=wrong prune, FN=missed attack\n\n"
            "CURRENT PERFORMANCE:\n"
            f"  AD baseline F1 : {ad_f1:.4f}\n"
            f"  RL eval F1     : {eval_f1:.4f}  (gap={ad_f1 - eval_f1:+.4f})\n"
            f"  Recent errors  : FP={fp}  FN={fn}\n"
            f"{anchor_block}\n"
            f"PROBLEMS:\n{issues}\n"
            "STRICT RULES:\n"
            "  1. Signature: def compute_reward(action, true_label, ml_prediction, ml_confidence=0.5)\n"
            "  2. Returns: (reward: float, category: str)\n"
            "  3. category must be exactly: 'TP', 'TN', 'FP', or 'FN'\n"
            "  4. reward in [-2.0, +2.0] — always clip: reward = max(-2.0, min(2.0, reward))\n"
            "  5. CRITICAL: category depends ONLY on action+true_label, never on confidence.\n\n"
            "Return ONLY the Python function inside a ```python block.\n\n"
            "```python\n"
            "def compute_reward(action, true_label, ml_prediction, ml_confidence=0.5):\n"
        )

    def _identify_issues(self, metrics: Dict, ml_baseline: Dict) -> str:
        issues  = []
        eval_f1 = self.eval_f1
        ad_f1   = ml_baseline.get("f1", 0)
        episode = metrics.get("episode", 0)
        fp      = metrics.get("total_fp", 0)
        fn      = metrics.get("total_fn", 0)
        sc      = metrics.get("penalty_scales", {})

        if episode == 0:
            issues.append(f"  - Episode 0: target beat AD_F1={ad_f1:.4f}")
            issues.append(f"  - AD errors: FP={ml_baseline.get('fp',0)} FN={ml_baseline.get('fn',0)}")
        else:
            target = ad_f1 * self.refinement_threshold
            if eval_f1 < target:
                issues.append(f"  - eval_F1 ({eval_f1:.4f}) < target ({target:.4f})")
            if fp > ml_baseline.get("fp", 0) * 1.1:
                issues.append(f"  - Excess FP: {fp} vs AD {ml_baseline.get('fp',0)}")
            if fn > ml_baseline.get("fn", 0) * 1.1:
                issues.append(f"  - Excess FN: {fn} vs AD {ml_baseline.get('fn',0)}")
            if sc.get("beta", 1) > 2.0:
                issues.append(f"  - Very high FP rate (beta={sc['beta']:.2f})")
            if sc.get("gamma", 1) > 2.0:
                issues.append(f"  - Very high FN rate (gamma={sc['gamma']:.2f})")
            if not issues:
                issues.append(f"  - Stagnating near {eval_f1:.4f} — need novel structure")
        return "\n".join(issues)

    # ── Code utilities ─────────────────────────────────────────────────────────

    def _extract_code(self, response: str) -> Optional[str]:
        m = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        if m:
            return m.group(1).strip()
        m = re.search(r"(def compute_reward.*?)(?=\n\ndef |\Z)", response, re.DOTALL)
        if m:
            return m.group(1).strip()
        return None

    def _validate(self, code: str) -> bool:
        try:
            compile(code, "<string>", "exec")
            ns = {}
            exec(code, ns)
            fn = ns.get("compute_reward")
            if fn is None:
                print("[LLM] Validation: compute_reward not found in generated code")
                return False
            # Test all 4 categories at 3 confidence levels
            for a, l, p, c, exp_cat in [
                (0,1,1,0.9,"TP"),(0,1,1,0.5,"TP"),(0,1,1,0.3,"TP"),
                (1,0,0,0.9,"TN"),(1,0,0,0.5,"TN"),(1,0,0,0.3,"TN"),
                (0,0,1,0.9,"FP"),(0,0,1,0.5,"FP"),(0,0,0,0.3,"FP"),
                (1,1,0,0.9,"FN"),(1,1,0,0.5,"FN"),(1,1,1,0.3,"FN"),
            ]:
                r, cat = fn(a, l, p, c)
                if not isinstance(r, (int, float)):
                    print(f"[LLM] Validation FAIL: reward not a number — "
                          f"action={a} label={l} conf={c} → {r}")
                    return False
                if cat not in ("TP","TN","FP","FN"):
                    print(f"[LLM] Validation FAIL: invalid category '{cat}' — "
                          f"action={a} label={l} conf={c}")
                    return False
                if cat != exp_cat:
                    print(f"[LLM] Validation FAIL: wrong category — "
                          f"action={a} label={l} conf={c} "
                          f"got='{cat}' expected='{exp_cat}'")
                    return False
                if not (-2.0 <= float(r) <= 2.0):
                    print(f"[LLM] Validation FAIL: reward {r:.3f} out of [-2.0, 2.0] — "
                          f"action={a} label={l} conf={c}")
                    return False
            return True
        except Exception as e:
            print(f"[LLM] Validation error: {e}")
            return False

    @staticmethod
    def _hash_code(code: str) -> str:
        lines = [l.strip() for l in code.splitlines()
                 if l.strip() and not l.strip().startswith("#")]
        return hashlib.md5("\n".join(lines).encode()).hexdigest()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _save_reward(self, info: Dict):
        rdir = os.path.join(self.output_dir, "rewards")
        os.makedirs(rdir, exist_ok=True)
        with open(os.path.join(rdir, f"{info['name']}.py"), "w") as f:
            m = info.get("metrics_at_creation", {})
            f.write(f"# {info['name']}  phase={info['phase']}  "
                    f"qualifies={info['qualifies']}\n")
            f.write(f"# ep={info.get('created_at_episode','?')}  "
                    f"{info['created_at']}\n")
            f.write(f"# eval_F1={m.get('eval_f1',0):.4f}  "
                    f"AD_F1={m.get('ad_f1',0):.4f}  "
                    f"FP={m.get('fp',0)}  FN={m.get('fn',0)}\n\n")
            f.write(info["code"] + "\n")

    def _log_reward_to_txt(self, info: Dict):
        path = os.path.join(self.output_dir, "rewards_log.txt")
        m    = info.get("metrics_at_creation", {})
        with open(path, "a") as f:
            f.write(f"\n{'='*60}\n{info['name']}  {info['created_at']}\n")
            f.write(f"eval_F1={m.get('eval_f1',0):.4f}  "
                    f"AD_F1={m.get('ad_f1',0):.4f}\n")
            f.write(f"{'='*60}\n{info['code']}\n")

    def _save_history(self):
        if not self.output_dir:
            return
        with open(os.path.join(self.output_dir, "llm_history.json"), "w") as f:
            json.dump({
                "current_version":      self.current_version,
                "current_version_name": self.current_version_name,
                "mode":                 self.mode,
                "eval_f1":              self.eval_f1,
                "zero_shot_attempts":   self.zero_shot_attempts,
                "refinement_attempts":  self.refinement_attempts,
                "anchor_reward":        self.anchor_reward,
                "reward_history":       self.reward_history,
                "qualified_rewards":    self.qualified_rewards,
            }, f, indent=2, default=str)

    def load_history(self, path: str):
        try:
            with open(path) as f:
                h = json.load(f)
            self.current_version      = h.get("current_version", 0)
            self.current_version_name = h.get("current_version_name", "none")
            self.mode                 = h.get("mode", MODE_ZERO_SHOT)
            self.anchor_reward        = h.get("anchor_reward", None)
            self.zero_shot_attempts   = h.get("zero_shot_attempts", 0)
            self.refinement_attempts  = h.get("refinement_attempts", 0)
            self.reward_history       = h.get("reward_history", [])
            self.qualified_rewards    = h.get("qualified_rewards", [])
            self._code_hashes         = set()
            for rv in self.reward_history:
                hv = rv.get("code_hash")
                if hv:
                    self._code_hashes.add(hv)
                elif "code" in rv:
                    self._code_hashes.add(self._hash_code(rv["code"]))
            print(f"[LLM] Loaded: {self.current_version_name}  mode={self.mode}")
            if self.anchor_reward:
                print(f"[LLM] Best reward: {self.anchor_reward['name']}  "
                      f"eval_F1={self.anchor_reward.get('eval_f1',0):.4f}")
        except Exception as e:
            print(f"[LLM] Could not load history: {e}")

    def get_current_version(self) -> str:  return self.current_version_name
    def get_eval_f1(self)         -> float: return self.eval_f1
    def get_mode(self)            -> str:   return self.mode
    def get_qualified_pool(self)  -> List[Dict]: return self.qualified_rewards


# ══════════════════════════════════════════════════════════════════
#  MAIN — test Qwen end to end
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    hf_token = os.environ.get("HF_TOKEN", "TOKEN")  # ← replace with your HuggingFace token
    if not hf_token:
        print("ERROR: HF_TOKEN not set")
        print("  export HF_TOKEN='hf_...'")
        sys.exit(1)

    print("=" * 60)
    print("Testing LLM Reward Optimizer — Qwen2.5-Coder-32B via HF")
    print("=" * 60)

    optimizer = LLMRewardOptimizer(
        hf_token=hf_token,
        refinement_interval=3000,
        verbose=True,
        max_versions=3,
        eval_window=10,
        refinement_threshold=1.005,
    )

    dummy_metrics  = {
        "episode": 0, "f1_score": 0.0,
        "total_fp": 200, "total_fn": 400,
        "ml_baseline_f1": 0.9162,
        "penalty_scales": {"beta": 1.0, "gamma": 1.0},
    }
    dummy_baseline = {"f1": 0.9162, "fp": 200, "fn": 400}

    print("\n--- Generating R1 (zero-shot cold start) ---")
    code = optimizer.generate_new_reward(dummy_metrics, dummy_baseline)

    if code:
        print("\n--- Testing generated function ---")
        ns = {}
        exec(code, ns)
        fn = ns["compute_reward"]
        all_ok = True
        for action, label, pred, conf, exp_cat in [
            (0, 1, 1, 0.9, "TP"),
            (1, 0, 0, 0.8, "TN"),
            (0, 0, 1, 0.7, "FP"),
            (1, 1, 0, 0.6, "FN"),
        ]:
            reward, cat = fn(action, label, pred, conf)
            ok = (cat == exp_cat) and (-2.0 <= reward <= 2.0)
            print(f"  [{'OK' if ok else 'FAIL'}] "
                  f"action={action} label={label} pred={pred} conf={conf} "
                  f"→ reward={reward:.3f}  cat={cat}  (expected {exp_cat})")
            if not ok:
                all_ok = False
        if all_ok:
            print("\n[OK] Qwen2.5-Coder-32B working correctly — ready for training")
        else:
            print("\n[FAIL] Check generated code above")
    else:
        print("\n[FAIL] No code returned — check error messages above")
        sys.exit(1)

