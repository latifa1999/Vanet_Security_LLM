"""
train.py  —  MAPPO for VANET Security with GNNActor + LLM reward optimization.

"""

import argparse
import glob
import json
import os
import re
import sys
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from magrl.utils.data_loader import UltraFastDataLoader
from magrl.environment.reward_functions import AdaptivePenaltyTracker, DynamicReward
from magrl.llm_optimizer.llm_reward_optimizer import LLMRewardOptimizer
from magrl.agents.mappo_agent import GNNActor, GNNEpisodeRunner, ppo_update_gnn


# ══════════════════════════════════════════════════════════════════
#  CRITIC  (unchanged — MLP on global state)
# ══════════════════════════════════════════════════════════════════

class Critic(nn.Module):
    def __init__(self, gs_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(gs_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════
#  GPU TENSOR STORE
# ══════════════════════════════════════════════════════════════════

class GPUDataStore:
    def __init__(self, loader: UltraFastDataLoader, max_edges: int,
                 device: torch.device):
        T, me = len(loader), max_edges
        print(f"\n[GPU] Building tensor store  T={T}  max_edges={me}")
        t0 = time.perf_counter()

        ed  = np.zeros((T, me, 10), dtype=np.float32)
        lbl = np.zeros((T, me),     dtype=np.int32)
        prd = np.zeros((T, me),     dtype=np.int32)
        snd = np.full ((T, me), -1, dtype=np.int32)
        rcv = np.full ((T, me), -1, dtype=np.int32)
        ne  = np.zeros(T,           dtype=np.int32)

        for t, g in enumerate(loader.graphs):
            n = min(g["n_edges"], me)
            ne[t] = n
            ed[t, :n]  = g["edge_data"][:n]
            lbl[t, :n] = g["labels"][:n]
            prd[t, :n] = g["preds"][:n]
            snd[t, :n] = g["send_ids"][:n]
            rcv[t, :n] = g["recv_ids"][:n]

        self.edge_data = torch.from_numpy(ed).to(device)
        self.labels    = torch.from_numpy(lbl).to(device)
        self.preds     = torch.from_numpy(prd).to(device)
        self.send_ids  = torch.from_numpy(snd).to(device)
        self.recv_ids  = torch.from_numpy(rcv).to(device)
        self.n_edges   = torch.from_numpy(ne).to(device)
        self.T = T; self.me = me; self.device = device
        mb = (ed.nbytes + lbl.nbytes + prd.nbytes + snd.nbytes + rcv.nbytes) / 1e6
        print(f"[GPU] Ready: {mb:.1f} MB on {device}  ({time.perf_counter()-t0:.1f}s)")


# ══════════════════════════════════════════════════════════════════
#  TENSORBOARD
# ══════════════════════════════════════════════════════════════════

class TBLogger:
    def __init__(self, log_dir: str):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(log_dir)
            self._enabled = True
            print(f"[TB] Logging → {log_dir}")
        except ImportError:
            self.tb = None; self._enabled = False
            print("[TB] tensorboard not installed")

    def log_episode(self, ep, f1, prec, rec, acc, tp, tn, fp, fn,
                    losses, beta, gamma, fp_rate, fn_rate,
                    curr_rw, best_f1, ad_f1, mean_reward=0.0, eval_f1=0.0):
        if not self._enabled: return
        tb = self.tb; tot = max(tp+tn+fp+fn, 1)
        tb.add_scalar("Performance/F1",            f1,         ep)
        tb.add_scalar("Performance/F1_eval",       eval_f1,    ep)
        tb.add_scalar("Performance/Precision",     prec,       ep)
        tb.add_scalar("Performance/Recall",        rec,        ep)
        tb.add_scalar("Performance/Accuracy",      acc,        ep)
        tb.add_scalar("Performance/BestF1",        best_f1,    ep)
        tb.add_scalar("Performance/DeltaVsAD",     f1-ad_f1,   ep)
        tb.add_scalar("Performance/EvalDeltaVsAD", eval_f1-ad_f1, ep)
        tb.add_scalar("Performance/AD_F1",         ad_f1,      ep)
        tb.add_scalar(f"PerReward/{curr_rw}/F1",      f1,      ep)
        tb.add_scalar(f"PerReward/{curr_rw}/F1_eval", eval_f1, ep)
        tb.add_scalar(f"PerReward/{curr_rw}/FP",      fp,      ep)
        tb.add_scalar(f"PerReward/{curr_rw}/FN",      fn,      ep)
        tb.add_scalar("Confusion/TP",      tp,       ep)
        tb.add_scalar("Confusion/TN",      tn,       ep)
        tb.add_scalar("Confusion/FP",      fp,       ep)
        tb.add_scalar("Confusion/FN",      fn,       ep)
        tb.add_scalar("Confusion/FP_rate", fp/tot,   ep)
        tb.add_scalar("Confusion/FN_rate", fn/tot,   ep)
        tb.add_scalar("AdaptivePenalty/Beta",  beta,  ep)
        tb.add_scalar("AdaptivePenalty/Gamma", gamma, ep)
        tb.add_scalar("PPO/ActorLoss",  losses.get("actor_loss",  0), ep)
        tb.add_scalar("PPO/CriticLoss", losses.get("critic_loss", 0), ep)
        tb.add_scalar("PPO/Entropy",    losses.get("entropy",     0), ep)

    def log_reward_switch(self, ep, old_rw, new_rw, f1, eval_f1, version_idx):
        if not self._enabled: return
        self.tb.add_scalar("RewardSwitch/VersionIndex",     version_idx, ep)
        self.tb.add_scalar("RewardSwitch/F1_at_switch",     f1,          ep)
        self.tb.add_scalar("RewardSwitch/EvalF1_at_switch", eval_f1,     ep)

    def log_llm_stats(self, ep, eval_f1, n_versions, n_dedup_skips, mode, n_qualified):
        if not self._enabled: return
        self.tb.add_scalar("LLM/EvalF1",           eval_f1,    ep)
        self.tb.add_scalar("LLM/VersionsGenerated", n_versions, ep)
        self.tb.add_scalar("LLM/Mode", 1 if mode == "refinement" else 0, ep)
        self.tb.add_scalar("LLM/QualifiedPool", n_qualified, ep)

    def add_reward_code(self, version_name: str, code: str):
        if not self._enabled: return
        self.tb.add_text(f"RewardCode/{version_name}", f"```python\n{code}\n```", 0)

    def close(self):
        if self._enabled: self.tb.close()


# ══════════════════════════════════════════════════════════════════
#  CHECKPOINT
# ══════════════════════════════════════════════════════════════════

def save_checkpoint(out_dir, ep, actor, critic, actor_opt, critic_opt,
                    best_f1, best_f1_reward_name, curr_rw_name, args):
    ck_dir = os.path.join(out_dir, "checkpoints")
    a = actor.module  if isinstance(actor,  nn.DataParallel) else actor
    c = critic.module if isinstance(critic, nn.DataParallel) else critic
    torch.save({
        "episode": ep, "actor": a.state_dict(), "critic": c.state_dict(),
        "actor_opt": actor_opt.state_dict(), "critic_opt": critic_opt.state_dict(),
        "best_f1": best_f1, "best_f1_reward_name": best_f1_reward_name,
        "curr_rw_name": curr_rw_name, "args": vars(args),
    }, os.path.join(ck_dir, f"ep{ep}.pth"))


def load_latest_checkpoint(resume_dir, actor, critic, actor_opt, critic_opt, device):
    ck_dir = os.path.join(resume_dir, "checkpoints")
    ckpts  = glob.glob(os.path.join(ck_dir, "ep*.pth"))
    if not ckpts:
        raise ValueError(f"No checkpoints in {ck_dir}")
    ep_nums = sorted(
        [(int(re.search(r"ep(\d+)\.pth", os.path.basename(p)).group(1)), p)
         for p in ckpts if re.search(r"ep(\d+)\.pth", os.path.basename(p))],
        reverse=True)
    latest_ep, latest_path = ep_nums[0]
    print(f"\n[RESUME] Loading: {latest_path}")
    ckpt = torch.load(latest_path, map_location=device, weights_only=False)
    a = actor.module  if isinstance(actor,  nn.DataParallel) else actor
    c = critic.module if isinstance(critic, nn.DataParallel) else critic
    a.load_state_dict(ckpt["actor"])
    c.load_state_dict(ckpt["critic"])
    try:
        actor_opt.load_state_dict(ckpt["actor_opt"])
        critic_opt.load_state_dict(ckpt["critic_opt"])
        print("[RESUME] Optimizers loaded")
    except Exception as e:
        print(f"[RESUME] Optimizer load failed ({e}) — fresh optimizers")
    best_f1 = ckpt.get("best_f1", 0.0) or 0.0
    if not best_f1:
        try:
            df = pd.read_csv(os.path.join(resume_dir, "metrics.csv"))
            best_f1 = float(df["f1"].max())
        except Exception:
            pass
    start_ep            = ckpt.get("episode", latest_ep) + 1
    best_f1_reward_name = ckpt.get("best_f1_reward_name", ckpt.get("curr_rw_name", "none"))
    curr_rw_name        = ckpt.get("curr_rw_name", "none")
    print(f"[RESUME] Ep={start_ep}  best_F1={best_f1:.4f}  reward={curr_rw_name}")
    return start_ep, best_f1, best_f1_reward_name, curr_rw_name, ckpt.get("args", {})


# ══════════════════════════════════════════════════════════════════
#  FINAL EVALUATION SUMMARY
# ══════════════════════════════════════════════════════════════════

def print_final_evaluation(best_f1, best_f1_reward_name, ad_f1, ml_bl,
                            reward_history, metrics_csv_path,
                            total_hours, llm_versions, llm_dedup_skips, out_dir):
    sep = "=" * 68
    improvement = best_f1 - ad_f1
    rel_gain    = improvement / max(ad_f1, 1e-8) * 100
    verdict     = ("IMPROVEMENT" if improvement > 0.01
                   else ("NEUTRAL" if improvement > -0.01 else "DEGRADATION"))

    print(f"\n{sep}\nTRAINING COMPLETE\n{sep}")
    print(f"  AD baseline F1 : {ad_f1:.4f}")
    print(f"  RL best raw F1 : {best_f1:.4f}  [{best_f1_reward_name}]")
    print(f"  Improvement    : {improvement:+.4f}  ({rel_gain:+.1f}%)  {verdict}")
    print(f"  Total time     : {total_hours:.2f} h")
    print(f"  LLM versions   : {llm_versions}  dedup_skips={llm_dedup_skips}")

    if reward_history:
        print(f"\n--- PER-REWARD SUMMARY ---")
        per_version = {}
        try:
            df = pd.read_csv(metrics_csv_path)
            for rw in df["reward_version"].unique():
                sub = df[df["reward_version"] == rw]
                per_version[rw] = {
                    "n_eps": len(sub), "mean_f1": float(sub["f1"].mean()),
                    "max_f1": float(sub["f1"].max()), "std_f1": float(sub["f1"].std()),
                }
        except Exception:
            pass
        for rv in reward_history:
            name  = rv["name"]; phase = rv.get("phase","?")
            m_c   = rv.get("metrics_at_creation", {})
            stats = per_version.get(name, {})
            print(f"\n  {name}  [{phase}]  ep={rv.get('created_at_episode','?')}")
            print(f"    At creation: eval_F1={m_c.get('eval_f1',0):.4f}  "
                  f"FP={m_c.get('fp',0)}  FN={m_c.get('fn',0)}")
            if stats:
                beat = "BEAT AD" if stats["mean_f1"] >= ad_f1 else "below AD"
                print(f"    {stats['n_eps']} eps: mean={stats['mean_f1']:.4f}  "
                      f"max={stats['max_f1']:.4f}  std={stats['std_f1']:.4f}  [{beat}]")

    print(f"\n  TensorBoard: tensorboard --logdir {out_dir}/tb\n{sep}")


# ══════════════════════════════════════════════════════════════════
#  MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════════════

def train(args):
    if args.resume:
        out = os.path.abspath(args.resume)
        resuming = True
        config_path = os.path.join(out, "results", "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                old_args = json.load(f)
            for k in ["data_path","chunk_size","max_edges_total",
                      "max_edges_per_agent","max_agents","hidden_dim"]:
                if k in old_args:
                    setattr(args, k, old_args[k])
    else:
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(args.output_dir, f"mappo_gnn_{ts}")
        resuming = False

    ck = os.path.join(out, "checkpoints")
    rs = os.path.join(out, "results")
    for d in (ck, rs): os.makedirs(d, exist_ok=True)

    dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    print("\n" + "=" * 68)
    print("GNN-MAPPO  —  VANET Security")
    print("=" * 68)
    print(f"Device   : {dev}  ({n_gpu} GPU)")
    print(f"Episodes : {args.num_episodes}")
    print(f"Mode     : {'RESUME' if resuming else 'FRESH'}")
    print("=" * 68)

    # ── Data ──────────────────────────────────────────────────────────────────
    cache  = os.path.join(os.path.dirname(os.path.abspath(args.data_path)),
                          "vanet_cache.pkl")
    loader = UltraFastDataLoader(args.data_path, args.scaler_path, cache)
    ml_bl  = loader.ml_baseline
    store  = GPUDataStore(loader, max_edges=args.max_edges_total, device=dev)

    print(f"\nAD Baseline: F1={ml_bl['f1']:.4f}  "
          f"TP={ml_bl['tp']} TN={ml_bl['tn']} FP={ml_bl['fp']} FN={ml_bl['fn']}")

    # ── Networks ──────────────────────────────────────────────────────────────
    gs_dim = args.max_edges_total * 10

    actor  = GNNActor(hidden_dim=args.hidden_dim, heads=4).to(dev)
    critic = Critic(gs_dim, args.hidden_dim * 2).to(dev)

    if n_gpu > 1:
        actor  = nn.DataParallel(actor)
        critic = nn.DataParallel(critic)
        print(f"[TRAIN] DataParallel: {n_gpu} GPUs")

    actor_opt  = optim.Adam(actor.parameters(),  lr=args.lr_actor)
    critic_opt = optim.Adam(critic.parameters(), lr=args.lr_critic)

    # ── Adaptive tracker ──────────────────────────────────────────────────────
    tracker = AdaptivePenaltyTracker(
        window_size=args.adaptive_window, beta_base=args.beta_base,
        gamma_base=args.gamma_base, max_scale=args.max_scale, verbose=True)

    # ── Resume / fresh state ──────────────────────────────────────────────────
    if resuming:
        start_ep, best_f1, best_f1_reward_name, curr_rw_name, _ = \
            load_latest_checkpoint(out, actor, critic, actor_opt, critic_opt, dev)
    else:
        start_ep = 0; best_f1 = 0.0
        best_f1_reward_name = curr_rw_name = "none"

    # ── Freeze GNN layers — only edge_dec trains ──────────────────────────────
    # Freezing the message-passing layers (node_enc, conv1, conv2) prevents
    # node embeddings from drifting on every PPO mini-batch step.
    # This was the main cause of policy instability (std=0.10-0.15).
    # The GNN still runs forward for inference — just not backward.
    _actor_base = actor.module if isinstance(actor, nn.DataParallel) else actor
    _actor_base.freeze_gnn(True)

    runner = GNNEpisodeRunner(
        store, actor, None, tracker,
        max_agents=args.max_agents,
        max_edges_per_agent=args.max_edges_per_agent,
        chunk_size=args.chunk_size,
        device=dev,
        use_action_masking=args.use_action_masking,
        mask_mode='honest',
    )
    _masking_active = False  

    # ── LLM optimizer ─────────────────────────────────────────────────────────
    llm = LLMRewardOptimizer(
        hf_token=args.hf_token or os.environ.get("HF_TOKEN"),
        refinement_interval=args.llm_interval,
        output_dir=rs,
        verbose=True,
        max_versions=args.llm_max_versions,
        eval_window=args.llm_eval_window,
        refinement_threshold=args.llm_threshold,
    )

    hist_path = os.path.join(rs, "llm_history.json")
    if resuming and os.path.exists(hist_path):
        llm.load_history(hist_path)

    llm_dedup_skips = 0

    # ── Generate R1 before training — LLM cold start ─────────────────────────
    if not resuming:
        print("\n" + "=" * 68)
        print("PRE-TRAINING: LLM generating R1 (zero-shot cold start)")
        print("=" * 68)
        init_metrics = {
            "episode": 0, "f1_score": 0.0,
            "total_fp": ml_bl.get("fp", 0), "total_fn": ml_bl.get("fn", 0),
            "ml_baseline_f1": ml_bl["f1"],
        }
        llm.should_refine(0, init_metrics)
        code = llm.generate_new_reward(init_metrics, ml_bl)
        if code:
            runner.rf    = DynamicReward(llm.get_current_version(), code)
            curr_rw_name = runner.rf.name
            print(f"\n[TRAIN] Training starts with {curr_rw_name}")
        else:
            raise RuntimeError("[TRAIN] LLM failed to generate R1.")
    else:
        if llm.reward_history:
            latest = llm.reward_history[-1]
            code   = latest.get("code", "")
            if code:
                runner.rf    = DynamicReward(latest["name"], code)
                curr_rw_name = latest["name"]
                print(f"[RESUME] Reward restored: {curr_rw_name}")
            else:
                raise RuntimeError("[RESUME] No reward code.")
        else:
            raise RuntimeError("[RESUME] No LLM history.")

    # ── TensorBoard + CSV ─────────────────────────────────────────────────────
    tb = TBLogger(os.path.join(out, "tb"))
    if llm.reward_history:
        tb.add_reward_code(curr_rw_name, llm.reward_history[-1].get("code", ""))

    mf_path = os.path.join(out, "metrics.csv")
    if resuming and os.path.exists(mf_path):
        mf = open(mf_path, "a")
    else:
        mf = open(mf_path, "w")
        mf.write("episode,reward_version,phase,f1,accuracy,precision,recall,"
                 "tp,tn,fp,fn,ad_f1,delta_f1,eval_f1,beta,gamma\n")

    if not resuming:
        with open(os.path.join(rs, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2, default=str)

    # ── Training loop ─────────────────────────────────────────────────────────
    t_start = time.perf_counter()
    speeds  = deque(maxlen=50)

    pbar = tqdm(range(start_ep, args.num_episodes), desc="MAPPO", unit="ep",
                initial=start_ep, total=args.num_episodes)

    for ep in pbar:
        t_ep = time.perf_counter()

        # Random chunk sampling — diverse data every episode
        t0  = np.random.randint(0, max(1, store.T - args.chunk_size))
        buf = runner.run(t0)
        if not buf:
            continue

        tp, tn, fp, fn = buf["ep_tp"], buf["ep_tn"], buf["ep_fp"], buf["ep_fn"]
        tot      = tp + tn + fp + fn
        prec     = tp / max(tp + fp, 1)
        rec      = tp / max(tp + fn, 1)
        f1       = 2 * prec * rec / max(prec + rec, 1e-8)
        acc      = (tp + tn) / max(tot, 1)
        sc       = tracker.get_scales()
        beta, gamma = sc["beta"], sc["gamma"]
        fp_rate  = sc.get("avg_fp_rate", fp / max(tot, 1))
        fn_rate  = sc.get("avg_fn_rate", fn / max(tot, 1))
        mean_rew = float(buf["rews"].mean().item())

        losses = ppo_update_gnn(actor, critic, actor_opt, critic_opt, buf, args)

        ep_s = time.perf_counter() - t_ep
        speeds.append(ep_s)
        avg_s = float(np.mean(speeds))
        eta_h = (args.num_episodes - ep) * avg_s / 3600

        if f1 > best_f1:
            best_f1             = f1
            best_f1_reward_name = curr_rw_name
            a = actor.module  if isinstance(actor, nn.DataParallel) else actor
            c = critic.module if isinstance(critic, nn.DataParallel) else critic
            torch.save({"actor": a.state_dict(), "critic": c.state_dict()},
                       os.path.join(ck, "best.pth"))
            tqdm.write(f"  NEW BEST F1: {best_f1:.4f}  [{best_f1_reward_name}]")

        eval_f1  = llm.update(f1)
        mode_tag = "ZS" if llm.get_mode() == "zero_shot" else "RF"

        tb.log_episode(ep=ep, f1=f1, prec=prec, rec=rec, acc=acc,
                       tp=tp, tn=tn, fp=fp, fn=fn,
                       losses=losses, beta=beta, gamma=gamma,
                       fp_rate=fp_rate, fn_rate=fn_rate,
                       curr_rw=curr_rw_name, best_f1=best_f1,
                       ad_f1=ml_bl["f1"], mean_reward=mean_rew, eval_f1=eval_f1)

        pbar.set_postfix({
            "F1": f"{f1:.4f}", "evalF1": f"{eval_f1:.4f}",
            "Rw": curr_rw_name, "mode": mode_tag,
            "s/ep": f"{avg_s:.2f}", "ETA": f"{eta_h:.1f}h",
        })

        mf.write(f"{ep},{curr_rw_name},{llm.get_mode()},{f1:.4f},{acc:.4f},"
                 f"{prec:.4f},{rec:.4f},{tp},{tn},{fp},{fn},{ml_bl['f1']:.4f},"
                 f"{f1-ml_bl['f1']:.4f},{eval_f1:.4f},{beta:.4f},{gamma:.4f}\n")
        mf.flush()

        if ep % args.print_interval == 0:
            delta     = f1 - ml_bl["f1"]
            elapsed_h = (time.perf_counter() - t_start) / 3600
            tqdm.write(
                f"\nEp {ep:>5d}  F1={f1:.4f}  evalF1={eval_f1:.4f}  "
                f"AD={ml_bl['f1']:.4f}  D={delta:+.4f}  "
                f"TP={tp} TN={tn} FP={fp} FN={fn}  "
                f"b={beta:.2f} g={gamma:.2f}  "
                f"{avg_s:.2f}s/ep  ETA={eta_h:.1f}h  "
                f"[{curr_rw_name}|{mode_tag}]  elapsed={elapsed_h:.1f}h  "
                f"best={best_f1:.4f}[{best_f1_reward_name}]"
            )

        # ── LLM check (skipped when --no_llm) ───────────────────────────────────
        if ep > 0 and not args.no_llm:
            info = {
                "f1_score": f1, "total_fp": fp, "total_fn": fn,
                "ml_baseline_f1": ml_bl["f1"], "penalty_scales": sc, "episode": ep,
            }
            if ep % args.llm_interval == 0:
                tb.log_llm_stats(ep, eval_f1, llm.current_version,
                                 llm_dedup_skips, llm.get_mode(),
                                 len(llm.get_qualified_pool()))

            if llm.should_refine(ep, info):
                old_rw = curr_rw_name
                code   = llm.generate_new_reward(info, ml_bl)
                if code is None:
                    llm_dedup_skips += 1
                    tqdm.write(f"  [LLM] Dedup skip #{llm_dedup_skips} — staying on {curr_rw_name}")
                else:
                    runner.rf    = DynamicReward(llm.get_current_version(), code)
                    curr_rw_name = runner.rf.name
                    tb.log_reward_switch(ep, old_rw, curr_rw_name,
                                         f1, eval_f1, llm.current_version)
                    tb.add_reward_code(curr_rw_name, code)
                    tqdm.write(
                        f"  [LLM] {old_rw} → {curr_rw_name}  "
                        f"mode={llm.get_mode()}  eval_F1={eval_f1:.4f}")

        if ep > 0 and ep % args.save_interval == 0:
            save_checkpoint(out, ep, actor, critic, actor_opt, critic_opt,
                            best_f1, best_f1_reward_name, curr_rw_name, args)
            llm._save_history()

    # ── Final saves ───────────────────────────────────────────────────────────
    save_checkpoint(out, args.num_episodes-1, actor, critic, actor_opt, critic_opt,
                    best_f1, best_f1_reward_name, curr_rw_name, args)
    a = actor.module  if isinstance(actor, nn.DataParallel) else actor
    c = critic.module if isinstance(critic, nn.DataParallel) else critic
    torch.save({"actor": a.state_dict(), "critic": c.state_dict()},
               os.path.join(ck, "final.pth"))
    mf.close(); llm._save_history(); tb.close()
    total_h = (time.perf_counter() - t_start) / 3600

    print_final_evaluation(
        best_f1=best_f1, best_f1_reward_name=best_f1_reward_name,
        ad_f1=ml_bl["f1"], ml_bl=ml_bl,
        reward_history=llm.reward_history,
        metrics_csv_path=os.path.join(out, "metrics.csv"),
        total_hours=total_h, llm_versions=llm.current_version,
        llm_dedup_skips=llm_dedup_skips, out_dir=out,
    )


# ══════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="GNN-MAPPO VANET Security")

    p.add_argument("--resume",      default=None)
    p.add_argument("--data_path",   type=str,
                   default="/home/latifa.elbouga/lustre/vr_outsec-vh2sz1t4fks/users/"
                           "latifa.elbouga/Data/Generated_data_2/out/test_data_with_predictions.csv")
    p.add_argument("--scaler_path", type=str, default=None)
    p.add_argument("--output_dir",  type=str, default="./outputs")

    p.add_argument("--num_episodes", type=int,   default=20000)
    p.add_argument("--chunk_size",   type=int,   default=1000)
    p.add_argument("--batch_size",   type=int,   default=4096)
    p.add_argument("--ppo_epochs",   type=int,   default=4)
    p.add_argument("--hidden_dim",   type=int,   default=128)
    p.add_argument("--lr_actor",     type=float, default=1e-5)
    p.add_argument("--lr_critic",    type=float, default=5e-5)
    p.add_argument("--gamma",        type=float, default=0.99)
    p.add_argument("--gae_lambda",   type=float, default=0.95)
    p.add_argument("--clip_eps",     type=float, default=0.1)
    p.add_argument("--entropy_coef", type=float, default=0.0005)
    p.add_argument("--value_coef",   type=float, default=0.5)
    p.add_argument("--max_grad_norm",type=float, default=0.5)

    p.add_argument("--max_agents",          type=int, default=100)
    p.add_argument("--max_edges_per_agent", type=int, default=30)
    p.add_argument("--max_edges_total",     type=int, default=130)

    p.add_argument("--adaptive_window", type=int,   default=100)
    p.add_argument("--beta_base",       type=float, default=1.0)
    p.add_argument("--gamma_base",      type=float, default=1.0)
    p.add_argument("--max_scale",       type=float, default=3.0)

    p.add_argument("--use_action_masking",  action="store_true", default=False)
    p.add_argument("--normal_keep_thr",     type=float, default=0.90)
    p.add_argument("--malicious_prune_thr", type=float, default=0.85)
    p.add_argument("--mask_mode", type=str, default='honest',
                   choices=['honest', 'none'],
                   help="honest=AD prediction (default), none=pure RL")

    p.add_argument("--hf_token",          default="Token") 
    p.add_argument("--llm_interval",      type=int,   default=3000)
    p.add_argument("--llm_max_versions",  type=int,   default=6)
    p.add_argument("--llm_eval_window",   type=int,   default=500)
    p.add_argument("--llm_threshold",     type=float, default=1.005)

    p.add_argument("--no_llm", action="store_true", default=False,
                   help="Disable LLM reward switching — train with R3 seed only")

    p.add_argument("--print_interval", type=int, default=100)
    p.add_argument("--save_interval",  type=int, default=500)

    args = p.parse_args()
    if not torch.cuda.is_available():
        print("WARNING: No GPU — reducing batch size")
        args.batch_size = 512
    args.use_llm = True
    train(args)

if __name__ == "__main__":
    main()