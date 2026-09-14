import argparse
import json
import os
import sys
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from magrl.utils.data_loader import UltraFastDataLoader

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GATv2Conv
    HAS_PYG = True
except Exception as e:
    HAS_PYG = False
    print(f"[WARN] torch_geometric import failed: {e}")


# ══════════════════════════════════════════════════════════════════
#  GNN ACTORS
# ══════════════════════════════════════════════════════════════════

class GNNActorV1(nn.Module):
    EDGE_DIM = 10; NODE_INIT = 10
    def __init__(self, hidden_dim=128, heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_enc = nn.Sequential(nn.Linear(10, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
        self.edge_enc = nn.Sequential(nn.Linear(10, hidden_dim), nn.ReLU())
        hd = hidden_dim // heads
        self.conv1 = GATv2Conv(hidden_dim, hd, heads=heads, edge_dim=hidden_dim, dropout=dropout, concat=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv2 = GATv2Conv(hidden_dim, hd, heads=heads, edge_dim=hidden_dim, dropout=dropout, concat=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.edge_dec = nn.Sequential(
            nn.Linear(hidden_dim*2+10, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid())
    def forward(self, batch):
        h = self.node_enc(batch.x); e = self.edge_enc(batch.edge_attr)
        h = torch.relu(self.norm1(self.conv1(h, batch.edge_index, e)))
        h = torch.relu(self.norm2(self.conv2(h, batch.edge_index, e)))
        s, d = batch.edge_index
        return self.edge_dec(torch.cat([h[s], h[d], batch.edge_attr], 1)).squeeze(-1)


class GNNActorV2(nn.Module):
    EDGE_DIM = 10; NODE_INIT = 20
    def __init__(self, hidden_dim=128, heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_enc = nn.Sequential(nn.Linear(20, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
        self.edge_enc = nn.Sequential(nn.Linear(10, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
        hd = hidden_dim // heads
        self.conv1 = GATv2Conv(hidden_dim, hd, heads=heads, edge_dim=hidden_dim, dropout=dropout, concat=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv2 = GATv2Conv(hidden_dim, hd, heads=heads, edge_dim=hidden_dim, dropout=dropout, concat=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        dec_in = hidden_dim * 2 + 10
        self.edge_dec = nn.Sequential(
            nn.Linear(dec_in, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(),
            nn.Linear(hidden_dim//2, 1), nn.Sigmoid())
    def forward(self, batch):
        h = self.node_enc(batch.x); e = self.edge_enc(batch.edge_attr)
        h1 = torch.relu(self.norm1(self.conv1(h,  batch.edge_index, e))); h = h1 + h
        h2 = torch.relu(self.norm2(self.conv2(h,  batch.edge_index, e))); h = h2 + h
        s, d = batch.edge_index
        return self.edge_dec(torch.cat([h[s], h[d], batch.edge_attr], 1)).squeeze(-1)


class MLPActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Sigmoid())
    def forward(self, x): return self.net(x)


# ══════════════════════════════════════════════════════════════════
#  GPU DATA STORE
# ══════════════════════════════════════════════════════════════════

class GPUDataStore:
    def __init__(self, loader, max_edges, device):
        T, me = len(loader), max_edges
        ed  = np.zeros((T, me, 10), dtype=np.float32)
        lbl = np.zeros((T, me),     dtype=np.int32)
        prd = np.zeros((T, me),     dtype=np.int32)
        snd = np.full ((T, me), -1, dtype=np.int32)
        rcv = np.full ((T, me), -1, dtype=np.int32)
        ne  = np.zeros(T,           dtype=np.int32)
        for t, g in enumerate(loader.graphs):
            n = min(g["n_edges"], me)
            ne[t] = n
            ed[t, :n] = g["edge_data"][:n]; lbl[t, :n] = g["labels"][:n]
            prd[t, :n] = g["preds"][:n];   snd[t, :n] = g["send_ids"][:n]
            rcv[t, :n] = g["recv_ids"][:n]
        self.edge_data = torch.from_numpy(ed).to(device)
        self.labels    = torch.from_numpy(lbl).to(device)
        self.preds     = torch.from_numpy(prd).to(device)
        self.send_ids  = torch.from_numpy(snd).to(device)
        self.recv_ids  = torch.from_numpy(rcv).to(device)
        self.n_edges   = torch.from_numpy(ne).to(device)
        self.T = T; self.me = me; self.device = device


# ══════════════════════════════════════════════════════════════════
#  ARCHITECTURE DETECTION
# ══════════════════════════════════════════════════════════════════

def detect_architecture(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    keys = list(ckpt["actor"].keys())
    is_gnn = any('node_enc' in k or 'conv1' in k or 'edge_dec' in k for k in keys)
    if is_gnn:
        hidden_dim        = ckpt["actor"]["node_enc.0.weight"].shape[0]
        node_enc_in       = ckpt["actor"]["node_enc.0.weight"].shape[1]
        has_edge_enc_norm = "edge_enc.1.weight" in ckpt["actor"]
        has_deep_dec      = "edge_dec.3.weight" in ckpt["actor"]
        arch = 'gnn_v2' if (node_enc_in == 20 and has_deep_dec and has_edge_enc_norm) else 'gnn_v1'
        print(f"[AUTO-DETECT] {arch.upper()}  hidden={hidden_dim}  node_enc_in={node_enc_in}")
        return arch, hidden_dim, None
    else:
        w = ckpt["actor"]["net.0.weight"].shape
        print(f"[AUTO-DETECT] MLP  hidden={w[0]}  max_ea={w[1]//10}")
        return 'mlp', w[0], w[1] // 10


# ══════════════════════════════════════════════════════════════════
#  MASKING FUNCTIONS
# ══════════════════════════════════════════════════════════════════
def mask_honest(actions, ad_predictions, confidences,
                normal_keep_thr=0.90, malicious_prune_thr=0.95,
                fp_win=None, fn_win=None):
    """Honest masking — uses ml_prediction + confidence only."""
    act = actions.copy()
    nk_thr = normal_keep_thr; mp_thr = malicious_prune_thr

    if fp_win is not None and len(fp_win) > 0:
        fp_r = float(np.mean(fp_win)); fn_r = float(np.mean(fn_win))
        if fp_r > 0.25: nk_thr = min(0.92, nk_thr + 0.05)
        elif fp_r < 0.15: nk_thr = max(0.85, nk_thr - 0.01)
        if fn_r > 0.05: mp_thr = min(0.99, mp_thr + 0.05)
        elif fn_r < 0.02: mp_thr = max(0.95, mp_thr - 0.01)

    ni = np.where(ad_predictions == 0)[0]
    if len(ni):
        min_keep = max(1, int(len(ni) * nk_thr))
        n_keeps  = int((act[ni] == 1).sum())
        if n_keeps < min_keep:
            prune = ni[act[ni] == 0]
            if len(prune):
                act[prune[np.argsort(confidences[prune])[:min_keep - n_keeps]]] = 1

    mi = np.where(ad_predictions == 1)[0]
    if len(mi):
        min_prune = max(1, int(len(mi) * mp_thr))
        n_prunes  = int((act[mi] == 0).sum())
        if n_prunes < min_prune:
            keep = mi[act[mi] == 1]
            if len(keep):
                act[keep[np.argsort(-confidences[keep])[:min_prune - n_prunes]]] = 0

    return act


# ══════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════

def calculate_metrics(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-8)
    f1w  = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    pw   = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rw   = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    acc  = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'f1_binary': f1, 'precision_binary': prec, 'recall_binary': rec,
            'f1_weighted': f1w, 'precision_weighted': pw, 'recall_weighted': rw,
            'accuracy': acc}


def calculate_ad_baseline(loader):
    all_preds, all_labels = [], []
    for g in loader.graphs:
        all_preds.extend(g["preds"].tolist())
        all_labels.extend(g["labels"].tolist())
    return calculate_metrics(np.array(all_labels), np.array(all_preds))


# ══════════════════════════════════════════════════════════════════
#  GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════

def build_single_graph(ed_v, snd_v, rcv_v, device, gnn_version):
    ne = len(ed_v)
    all_ids = torch.cat([snd_v, rcv_v]).unique()
    id_map  = torch.full((int(all_ids.max()) + 1,), -1, dtype=torch.long, device=device)
    id_map[all_ids] = torch.arange(len(all_ids), device=device)
    src = id_map[snd_v]; dst = id_map[rcv_v]
    n_nodes = len(all_ids); ones_e = torch.ones(ne, 1, device=device)

    if gnn_version == 'gnn_v2':
        sf = torch.zeros(n_nodes, 10, device=device); sc = torch.zeros(n_nodes, 1, device=device)
        sf.scatter_add_(0, src.unsqueeze(1).expand(-1, 10), ed_v)
        sc.scatter_add_(0, src.unsqueeze(1), ones_e)
        rf = torch.zeros(n_nodes, 10, device=device); rc = torch.zeros(n_nodes, 1, device=device)
        rf.scatter_add_(0, dst.unsqueeze(1).expand(-1, 10), ed_v)
        rc.scatter_add_(0, dst.unsqueeze(1), ones_e)
        node_feat = torch.cat([sf / sc.clamp(min=1), rf / rc.clamp(min=1)], dim=1)
    else:
        nf = torch.zeros(n_nodes, 10, device=device); nc = torch.zeros(n_nodes, 1, device=device)
        for ids in (src, dst):
            nf.scatter_add_(0, ids.unsqueeze(1).expand(-1, 10), ed_v)
            nc.scatter_add_(0, ids.unsqueeze(1), ones_e)
        node_feat = nf / nc.clamp(min=1)

    return Data(x=node_feat.float(),
                edge_index=torch.stack([src, dst], 0).long(),
                edge_attr=ed_v.float())


# ══════════════════════════════════════════════════════════════════
#  TIMESTEP RUNNER
# ══════════════════════════════════════════════════════════════════

def run_timestep(t, store, actor, gnn_version, mask_mode,
                 normal_keep_thr, malicious_prune_thr, fp_win, fn_win):
    """
    Run one timestep and return (labels, preds_masked, preds_raw).
    mask_mode: 'honest' | 'none'
    """
    n_e = int(store.n_edges[t])
    if n_e == 0:
        return None

    ed_gpu  = store.edge_data[t, :n_e]
    lbl_gpu = store.labels[t,   :n_e]
    prd_gpu = store.preds[t,    :n_e]
    snd_gpu = store.send_ids[t, :n_e]
    rcv_gpu = store.recv_ids[t, :n_e]

    valid = (snd_gpu >= 0) & (rcv_gpu >= 0)
    if valid.sum() == 0:
        return None

    ed_v    = ed_gpu[valid]; snd_v = snd_gpu[valid]; rcv_v = rcv_gpu[valid]
    lbl_np  = lbl_gpu[valid].cpu().numpy()
    prd_np  = prd_gpu[valid].cpu().numpy()
    conf_np = ed_v[:, 1].cpu().numpy()
    ne      = int(valid.sum())

    g     = build_single_graph(ed_v, snd_v, rcv_v, store.device, gnn_version)
    batch = Batch.from_data_list([g]).to(store.device)

    with torch.no_grad():
        probs    = actor(batch)
        acts_raw = (probs > 0.5).long().cpu().numpy().astype(np.int32)

    if mask_mode == 'honest':
        acts_final = mask_honest(
            acts_raw, prd_np, conf_np,
            normal_keep_thr=normal_keep_thr,
            malicious_prune_thr=malicious_prune_thr,
            fp_win=fp_win, fn_win=fn_win,
        )
    else:
        # ── NONE: pure RL output ──────────────────────────────────────────────
        acts_final = acts_raw

    # action=0 → PRUNE → predict malicious (1)
    # action=1 → KEEP  → predict normal    (0)
    preds_final = 1 - acts_final
    preds_raw   = 1 - acts_raw

    fp_t = int(((preds_final == 1) & (lbl_np == 0)).sum())
    fn_t = int(((preds_final == 0) & (lbl_np == 1)).sum())
    fp_win.append(fp_t / max(ne, 1))
    fn_win.append(fn_t / max(ne, 1))

    return lbl_np.tolist(), preds_final.tolist(), preds_raw.tolist()


# ══════════════════════════════════════════════════════════════════
#  MAIN EVALUATE
# ══════════════════════════════════════════════════════════════════

def evaluate(args):
    print("\n" + "=" * 68)
    print("MAPPO EVALUATION")
    print("=" * 68)
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Data        : {args.data_path}")
    print(f"Mask mode   : {args.mask_mode.upper()}")
    if args.mask_mode == 'honest':
        print(f"  Uses ml_prediction + confidence (production masking)")
    else:
        print(f"  Pure RL actor output, no masking")
    print("=" * 68)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")

    arch, hidden_dim, extra = detect_architecture(args.checkpoint, dev)
    if args.hidden_dim:
        hidden_dim = args.hidden_dim

    cache  = os.path.join(os.path.dirname(os.path.abspath(args.data_path)),
                          "vanet_cache.pkl")
    loader = UltraFastDataLoader(args.data_path, None, cache)
    store  = GPUDataStore(loader, max_edges=args.max_edges_total, device=dev)

    print(f"\n[AD BASELINE]")
    ad_m = calculate_ad_baseline(loader)
    print(f"  Binary F1   : {ad_m['f1_binary']:.4f}")
    print(f"  Weighted F1 : {ad_m['f1_weighted']:.4f}")
    print(f"  TP={ad_m['tp']:,}  TN={ad_m['tn']:,}  FP={ad_m['fp']:,}  FN={ad_m['fn']:,}")

    # Load actor
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    assert HAS_PYG, "torch_geometric required"
    ActorClass = GNNActorV2 if arch == 'gnn_v2' else (GNNActorV1 if arch == 'gnn_v1' else None)

    if ActorClass:
        actor = ActorClass(hidden_dim=hidden_dim, heads=4).to(dev)
        actor.load_state_dict(ckpt["actor"])
        actor.eval()
        print(f"\n[LOADED] {arch.upper()}  hidden={hidden_dim}")
    else:
        max_ea  = extra if args.max_edges_per_agent is None else args.max_edges_per_agent
        obs_dim = max_ea * 10
        actor   = MLPActor(obs_dim, max_ea, hidden_dim).to(dev)
        actor.load_state_dict(ckpt["actor"])
        actor.eval()
        arch = 'mlp'
        print(f"\n[LOADED] MLPActor  obs_dim={obs_dim}")

    fp_win = deque(maxlen=100)
    fn_win = deque(maxlen=100)
    all_labels, all_preds, all_preds_raw = [], [], []

    pbar = tqdm(range(store.T), desc=f"Eval ({args.mask_mode})", unit="t")
    for t in pbar:
        res = run_timestep(
            t, store, actor, arch, args.mask_mode,
            args.normal_keep_thr, args.malicious_prune_thr,
            fp_win, fn_win,
        )
        if res is None:
            continue
        labels, preds, preds_raw = res
        all_labels.extend(labels)
        all_preds.extend(preds)
        all_preds_raw.extend(preds_raw)
        if t % 100 == 0 and all_preds:
            m = calculate_metrics(np.array(all_labels), np.array(all_preds))
            pbar.set_postfix(F1=f"{m['f1_binary']:.4f}", F1w=f"{m['f1_weighted']:.4f}")

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_raw  = np.array(all_preds_raw)

    m  = calculate_metrics(y_true, y_pred)
    mr = calculate_metrics(y_true, y_raw)

    print("\n" + "=" * 68)
    print("RESULTS")
    print("=" * 68)
    print(f"Architecture : {arch.upper()}")
    print(f"Mask mode    : {args.mask_mode.upper()}")
    print(f"Decisions    : {len(y_true):,}")
    print()

    label = {
        'honest': 'HONEST (AD prediction)',
        'none':   'NONE (pure RL)',
    }[args.mask_mode]

    print(f"── {label} ──")
    print(f"  Binary F1        : {m['f1_binary']:.4f}")
    print(f"    Precision      : {m['precision_binary']:.4f}")
    print(f"    Recall         : {m['recall_binary']:.4f}")
    print(f"  Weighted F1      : {m['f1_weighted']:.4f}")
    print(f"    Precision      : {m['precision_weighted']:.4f}")
    print(f"    Recall         : {m['recall_weighted']:.4f}")
    print(f"  Accuracy         : {m['accuracy']:.4f}")
    print(f"  TP={m['tp']:,}  TN={m['tn']:,}  FP={m['fp']:,}  FN={m['fn']:,}")

    print()
    print("── Raw RL actor (no masking) ──")
    print(f"  Binary F1   : {mr['f1_binary']:.4f}")
    print(f"  FP={mr['fp']:,}  FN={mr['fn']:,}")

    print()
    print("── vs AD BASELINE ──")
    print(f"  Binary F1   : {m['f1_binary']:.4f} vs {ad_m['f1_binary']:.4f}"
          f"  (Δ {m['f1_binary']-ad_m['f1_binary']:+.4f})")
    print(f"  Weighted F1 : {m['f1_weighted']:.4f} vs {ad_m['f1_weighted']:.4f}"
          f"  (Δ {m['f1_weighted']-ad_m['f1_weighted']:+.4f})")
    print(f"  FP reduction: {ad_m['fp']:,} → {m['fp']:,}  ({ad_m['fp']-m['fp']:+,})")
    print(f"  FN reduction: {ad_m['fn']:,} → {m['fn']:,}  ({ad_m['fn']-m['fn']:+,})")

    print("=" * 68)

    # Save JSON
    out_dir = os.path.dirname(args.checkpoint)
    result  = {
        "checkpoint":  args.checkpoint,
        "architecture": arch,
        "mask_mode":   args.mask_mode,
        "uses_true_label": args.mask_mode == 'none',
        "total_decisions": len(y_true),
        "metrics": {
            "binary_f1":        m['f1_binary'],
            "precision_binary": m['precision_binary'],
            "recall_binary":    m['recall_binary'],
            "weighted_f1":      m['f1_weighted'],
            "accuracy":         m['accuracy'],
            "tp": m['tp'], "tn": m['tn'], "fp": m['fp'], "fn": m['fn'],
        },
        "metrics_raw_actor": {
            "binary_f1": mr['f1_binary'], "fp": mr['fp'], "fn": mr['fn'],
        },
        "ad_baseline": {
            "binary_f1": ad_m['f1_binary'], "weighted_f1": ad_m['f1_weighted'],
            "tp": ad_m['tp'], "tn": ad_m['tn'], "fp": ad_m['fp'], "fn": ad_m['fn'],
        },
        "improvement_vs_ad": {
            "binary_f1":    m['f1_binary']  - ad_m['f1_binary'],
            "weighted_f1":  m['f1_weighted'] - ad_m['f1_weighted'],
            "fp_reduction": ad_m['fp'] - m['fp'],
            "fn_reduction": ad_m['fn'] - m['fn'],
        },
    }
    path = os.path.join(out_dir, f"evaluation_{arch}_{args.mask_mode}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[SAVED] {path}")

    # Plots
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f"Eval — {arch.upper()}  |  Mask: {args.mask_mode.upper()}",
            fontsize=13, fontweight='bold')

        # Confusion matrix — masked
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=["Normal", "Malicious"]).plot(
            ax=axes[0], cmap='Blues', values_format=',d')
        axes[0].set_title(f"{args.mask_mode.upper()} F1={m['f1_binary']:.4f}")

        # Confusion matrix — raw
        cm_raw = confusion_matrix(y_true, y_raw, labels=[0, 1])
        ConfusionMatrixDisplay(cm_raw, display_labels=["Normal", "Malicious"]).plot(
            ax=axes[1], cmap='Oranges', values_format=',d')
        axes[1].set_title(f"Raw RL F1={mr['f1_binary']:.4f}")

        # Bar chart comparison
        names = ['Binary F1', 'Weighted F1']
        x = np.arange(2); w = 0.25
        axes[2].bar(x-w, [ad_m['f1_binary'], ad_m['f1_weighted']],  w, label='AD Baseline', color='steelblue', alpha=0.8)
        axes[2].bar(x,   [m['f1_binary'],     m['f1_weighted']],     w, label=f'Masked ({args.mask_mode})', color='green', alpha=0.8)
        axes[2].bar(x+w, [mr['f1_binary'],    mr['f1_weighted']],    w, label='Raw RL', color='orange', alpha=0.8)
        axes[2].set_xticks(x); axes[2].set_xticklabels(names)
        axes[2].set_ylabel('F1'); axes[2].set_title('F1 Comparison')
        axes[2].legend(); axes[2].grid(alpha=0.3, axis='y')

        plt.tight_layout()
        plot_path = os.path.join(out_dir, f"evaluation_{arch}_{args.mask_mode}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[SAVED] {plot_path}")
    except Exception as e:
        print(f"[WARNING] Plot failed: {e}")


# ══════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Honest/none masking evaluation for MAPPO checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="")

    p.add_argument("--checkpoint", type=str,
                   default="/home/latifa.elbouga/lustre/vr_outsec-vh2sz1t4fks/users/latifa.elbouga/Complete_MAGRL_LLM_FS_final_yarbi_copy_masking_merged/outputs/mappo_gnn_20260511_105253/checkpoints/best.pth")
    p.add_argument("--data_path", type=str,
                   default="/home/latifa.elbouga/lustre/vr_outsec-vh2sz1t4fks/users/latifa.elbouga/Data/Generated_data_2/Attacks/temp2.csv")

    p.add_argument("--mask_mode", type=str, default='none',
                   choices=['honest', 'none'],
                   help="honest=AD prediction, none=pure RL")

    p.add_argument("--max_edges_total",     type=int,   default=130)
    p.add_argument("--max_edges_per_agent", type=int,   default=None)
    p.add_argument("--max_agents",          type=int,   default=50)
    p.add_argument("--hidden_dim",          type=int,   default=None)
    p.add_argument("--normal_keep_thr",     type=float, default=0.50)
    p.add_argument("--malicious_prune_thr", type=float, default=0.50)

    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()