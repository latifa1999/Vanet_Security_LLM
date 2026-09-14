"""
GNN-based Actor for VANET Multi-Agent MAPPO

Architecture:
  GNNActor:
    - Node init     : linear(edge_feat_agg → hidden)
    - 2x GATConv    : message passing over vehicle graph
    - Edge decoder  : per-edge action probabilities
    - Batched graph : ALL agents from ALL timesteps → ONE PyG Batch → ONE GPU call

  Critic stays MLP (global state already encodes full graph).

"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from collections import deque

# ── Optional PyG import (graceful fallback message) ───────────────────────────
try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GATv2Conv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("[GNN] torch_geometric not found. Install it:")
    print("      pip install torch-geometric --break-system-packages")


# ══════════════════════════════════════════════════════════════════════════════
#  GNN ACTOR
# ══════════════════════════════════════════════════════════════════════════════

class GNNActor(nn.Module):
    """
    Input:  PyG Batch (heterogeneous vehicle graphs)
    Output: per-edge action probabilities  shape (total_edges_in_batch,)

    Node features:
      Each vehicle node = mean of its connected edge features (10-dim)
      → projected to hidden_dim

    Edge features:
      Raw 10-dim per edge  → used in GATv2 edge_attr

    Message passing:
      2 x GATv2Conv  (edge-aware attention, proven better than GATv1 for graphs
                       with varying edge attributes)

    Edge decoder:
      concat(src_node_feat, dst_node_feat, edge_feat) → linear → sigmoid
      Produces one probability per edge (PRUNE=0 / KEEP=1)
    """

    EDGE_DIM  = 10   # [pred, conf, posx, posy, spdx, spdy, aclx, acly, hedx, hedy]
    NODE_INIT = 20   # 10 as sender + 10 as receiver (directional node features)

    def __init__(self,
                 hidden_dim:  int   = 128,
                 heads:       int   = 4,
                 dropout:     float = 0.1):
        super().__init__()
        assert HAS_PYG, "torch_geometric required for GNNActor"

        self.hidden_dim = hidden_dim
        self._gnn_frozen = False  # when True, only edge_dec trains

        # ── Node encoder ──────────────────────────────────────────────────────
        # Input is now 20-dim (sender-role feats + receiver-role feats)
        self.node_enc = nn.Sequential(
            nn.Linear(self.NODE_INIT, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # ── Edge encoder (for GATv2 edge_attr) ────────────────────────────────
        self.edge_enc = nn.Sequential(
            nn.Linear(self.EDGE_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # ── GATv2 layers with residual connections ────────────────────────────
        # Residual: h_out = GATv2(h) + h  — stabilises training, helps gradient
        head_dim = hidden_dim // heads
        self.conv1 = GATv2Conv(
            in_channels  = hidden_dim,
            out_channels = head_dim,
            heads        = heads,
            edge_dim     = hidden_dim,
            dropout      = dropout,
            concat       = True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.conv2 = GATv2Conv(
            in_channels  = hidden_dim,
            out_channels = head_dim,
            heads        = heads,
            edge_dim     = hidden_dim,
            dropout      = dropout,
            concat       = True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # ── Edge action decoder — 3-layer with LayerNorm ──────────────────────
        # concat(src_h, dst_h, raw_edge_feat) → hidden → hidden//2 → 1
        # Extra layer + LayerNorm helps distinguish subtle edge differences
        dec_in = hidden_dim * 2 + self.EDGE_DIM
        self.edge_dec = nn.Sequential(
            nn.Linear(dec_in,         hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,     hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, batch: "Batch") -> torch.Tensor:
        """
        Args:
            batch: PyG Batch with fields:
                     x           (total_nodes, NODE_INIT)   — node features
                     edge_index  (2, total_edges)            — [src, dst]
                     edge_attr   (total_edges, EDGE_DIM)     — raw edge features
        Returns:
            probs  (total_edges,)  — action probability per edge
        """
        h     = self.forward_embeddings(batch)
        src, dst   = batch.edge_index
        edge_input = torch.cat([h[src], h[dst], batch.edge_attr], dim=1)
        return self.edge_dec(edge_input).squeeze(-1)

    def forward_embeddings(self, batch: "Batch") -> torch.Tensor:
        """
        Run only the GNN layers (node encoder + GATv2 with residuals).
        Returns node embeddings h  (total_nodes, hidden_dim).
        Called once per PPO epoch; edge decoder runs separately on mini-batches.

        Residual connections: h = relu(norm(conv(h))) + h
        Prevents over-smoothing — vehicles keep their own identity after
        message passing, rather than converging to a neighbourhood average.
        """
        h   = self.node_enc(batch.x)          # (N, hidden)
        e_h = self.edge_enc(batch.edge_attr)  # (E, hidden)

        # Layer 1 + residual
        h1  = torch.relu(self.norm1(self.conv1(h, batch.edge_index, e_h)))
        h   = h1 + h   # residual

        # Layer 2 + residual
        h2  = torch.relu(self.norm2(self.conv2(h, batch.edge_index, e_h)))
        h   = h2 + h   # residual

        return h

    def freeze_gnn(self, freeze: bool = True):
        """
        Freeze/unfreeze GNN layers (node_enc, edge_enc, conv1, conv2, norm1, norm2).
        When frozen, only edge_dec parameters are updated during PPO.
        Use this after the policy has found a good region — prevents the
        message-passing layers from drifting away from stable node representations.
        """
        self._gnn_frozen = freeze
        for name, param in self.named_parameters():
            if 'edge_dec' not in name:
                param.requires_grad = not freeze
        status = "FROZEN" if freeze else "UNFROZEN"
        print(f"[GNN] GNN layers {status} — "
              f"{'only edge_dec trains' if freeze else 'full network trains'}")


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH BUILDER  
# ══════════════════════════════════════════════════════════════════════════════

def build_pyg_graph(ed_gpu:  torch.Tensor,   # (n_e, 10)
                    snd_gpu: torch.Tensor,   # (n_e,)   vehicle IDs
                    rcv_gpu: torch.Tensor,   # (n_e,)
                    device:  torch.device,
                    ) -> "Data":
    """
    Build ONE PyG Data object for a single timestep.

    Nodes  = unique vehicles in this timestep (re-indexed 0..N-1)
    Edges  = V2V messages (directed: sender → receiver)

    Node feature = mean of all edge features the vehicle participates in
    Edge feature = raw 10-dim edge_data row
    """
    # Re-index vehicle IDs → contiguous 0..N-1
    all_ids = torch.cat([snd_gpu, rcv_gpu]).unique()
    id_map  = torch.full((int(all_ids.max()) + 1,), -1,
                         dtype=torch.long, device=device)
    id_map[all_ids] = torch.arange(len(all_ids), device=device)

    src = id_map[snd_gpu]   # (n_e,)
    dst = id_map[rcv_gpu]   # (n_e,)
    n   = len(all_ids)
    n_e = len(ed_gpu)

    # ── Node features: SEPARATE sender-role and receiver-role aggregations ──────
    # A vehicle sending messages has a different role than one receiving them.
    # Concatenating both gives the GNN directional context lost by plain mean.
    # node_feat = [mean_as_sender (10) | mean_as_receiver (10)] → 20-dim
    ones_e = torch.ones(n_e, 1, device=device)

    snd_feat = torch.zeros(n, 10, device=device)
    snd_cnt  = torch.zeros(n, 1,  device=device)
    snd_feat.scatter_add_(0, src.unsqueeze(1).expand(-1, 10), ed_gpu)
    snd_cnt.scatter_add_ (0, src.unsqueeze(1), ones_e)

    rcv_feat = torch.zeros(n, 10, device=device)
    rcv_cnt  = torch.zeros(n, 1,  device=device)
    rcv_feat.scatter_add_(0, dst.unsqueeze(1).expand(-1, 10), ed_gpu)
    rcv_cnt.scatter_add_ (0, dst.unsqueeze(1), ones_e)

    snd_feat = snd_feat / snd_cnt.clamp(min=1)
    rcv_feat = rcv_feat / rcv_cnt.clamp(min=1)
    node_feat = torch.cat([snd_feat, rcv_feat], dim=1)   # (N, 20)

    edge_index = torch.stack([src, dst], dim=0)   # (2, n_e)

    return Data(
        x          = node_feat.float(),    # (N, 20)
        edge_index = edge_index.long(),
        edge_attr  = ed_gpu.float(),       # (n_e, 10)
        n_nodes    = n,
        n_edges    = n_e,
    )


def build_pyg_batch(store,           # GPUDataStore
                    t_start: int,
                    t_end:   int,
                    device:  torch.device,
                    ) -> Tuple["Batch", List[Dict]]:
    """
    Build a PyG Batch covering timesteps [t_start, t_end).

    Returns:
        batch      : PyG Batch — all graphs merged, offsets handled by PyG
        meta       : list[dict] per timestep:
                       {
                         't'         : int,
                         'n_edges'   : int,
                         'edge_slice': (start, end),   # into batch.edge_attr
                         'agents'    : LongTensor,     # unique vehicle IDs
                         'snd'       : LongTensor,
                         'rcv'       : LongTensor,
                         'labels'    : LongTensor,
                         'preds'     : LongTensor,
                       }
    """
    graphs = []
    meta   = []
    edge_offset = 0

    for t in range(t_start, t_end):
        n_e = int(store.n_edges[t])
        if n_e == 0:
            continue

        ed  = store.edge_data[t, :n_e]
        snd = store.send_ids[t,  :n_e]
        rcv = store.recv_ids[t,  :n_e]
        lbl = store.labels[t,    :n_e]
        prd = store.preds[t,     :n_e]

        valid = (snd >= 0) & (rcv >= 0)
        if valid.sum() == 0:
            continue

        ed  = ed[valid]; snd = snd[valid]; rcv = rcv[valid]
        lbl = lbl[valid]; prd = prd[valid]
        ne  = len(ed)

        g = build_pyg_graph(ed, snd, rcv, device)
        graphs.append(g)

        agents = torch.cat([snd, rcv]).unique()
        meta.append({
            't':          t,
            'n_edges':    ne,
            'edge_slice': (edge_offset, edge_offset + ne),
            'agents':     agents,
            'snd':        snd,
            'rcv':        rcv,
            'labels':     lbl,
            'preds':      prd,
            'conf':       ed[:, 1],   # Prob_Malicious
        })
        edge_offset += ne

    if not graphs:
        return None, []

    batch = Batch.from_data_list(graphs)
    return batch, meta


# ══════════════════════════════════════════════════════════════════════════════
#  GNN EPISODE RUNNER  (drop-in replacement for EpisodeRunner)
# ══════════════════════════════════════════════════════════════════════════════

class GNNEpisodeRunner:
    """
    Drop-in replacement for EpisodeRunner.
    Builds ONE PyG Batch per chunk → ONE GNNActor forward pass.

    Compatible with same reward_fn, tracker, action masking as before.
    """

    def __init__(
        self,
        store,
        actor:               GNNActor,
        reward_fn,
        tracker,
        max_agents:          int,
        max_edges_per_agent: int,
        chunk_size:          int,
        device:              torch.device,
        use_action_masking:  bool  = True,
        normal_keep_thr:     float = 0.90, #0.95
        malicious_prune_thr: float = 0.85, #0.98
        mask_mode:           str   = 'honest',
    ):
        self.store    = store
        self.actor    = actor
        self.rf       = reward_fn
        self.tracker  = tracker
        self.max_a    = max_agents
        self.max_ea   = max_edges_per_agent
        self.chunk    = chunk_size
        self.device   = device
        self.use_mask = use_action_masking
        self.nk_thr   = normal_keep_thr
        self.mp_thr   = malicious_prune_thr
        self.mask_mode = mask_mode  # 'honest' 
        self.gs_dim   = store.me * 10

        self.fp_win = deque(maxlen=100)
        self.fn_win = deque(maxlen=100)

    def run(self, t_start: int) -> Dict:
        if self.rf is None:
            raise RuntimeError("[GNN RUNNER] No reward function set.")

        t_end = min(t_start + self.chunk, self.store.T)

        # ── Build full batch for this chunk ───────────────────────────────────
        batch, meta = build_pyg_batch(self.store, t_start, t_end, self.device)
        if not meta:
            return {}

        batch = batch.to(self.device)

        # ── ONE forward pass for ALL edges in ALL timesteps ───────────────────
        with torch.no_grad():
            all_probs = self.actor(batch)          # (total_edges_in_chunk,)
            dist      = torch.distributions.Bernoulli(all_probs)
            all_acts  = dist.sample()              # (total_edges,)
            all_lps   = dist.log_prob(all_acts)    # (total_edges,)

        # ── Global states for critic ──────────────────────────────────────────
        # One global state vector per timestep (same as before)
        gs_list = []
        for m in meta:
            t   = m['t']
            n_e = int(self.store.n_edges[t])
            k   = min(n_e, self.store.me)
            gs  = torch.zeros(self.gs_dim, device=self.device)
            gs[:k*10] = self.store.edge_data[t, :k].reshape(-1)
            # Expand to number of edges in this timestep (for GAE later)
            gs_list.append(gs.unsqueeze(0).expand(m['n_edges'], -1))

        # ── Per-timestep reward computation ───────────────────────────────────
        buf_acts = []; buf_lps  = []; buf_gs   = []
        buf_rews = []; buf_prbs = []

        ep_tp = ep_tn = ep_fp = ep_fn = 0

        for m in meta:
            s, e    = m['edge_slice']
            acts_t  = all_acts[s:e].cpu().numpy().astype(np.int32)
            lps_t   = all_lps[s:e]
            probs_t = all_probs[s:e]
            lbl_t   = m['labels'].cpu().numpy()
            prd_t   = m['preds'].cpu().numpy()
            conf_t  = m['conf'].cpu().numpy()

            # ── Action masking: honest | none ───────────────────
            if self.use_mask:
                acts_t = self._mask(acts_t, prd_t, conf_t)

            # ── Reward ────────────────────────────────────────────────────────
            result   = self.rf.compute_batch(
                acts_t, lbl_t, prd_t, conf_t,
                normalize=True, adaptive_tracker=self.tracker)
            rew_np   = np.array(result['rewards'], dtype=np.float32)
            rew_t    = torch.from_numpy(rew_np).to(self.device)

            ep_tp += result['tp']; ep_tn += result['tn']
            ep_fp += result['fp']; ep_fn += result['fn']

            n_dec = len(acts_t)
            self.fp_win.append(result['fp'] / max(n_dec, 1))
            self.fn_win.append(result['fn'] / max(n_dec, 1))

            buf_acts.append(torch.from_numpy(acts_t).float().to(self.device))
            buf_lps.append(lps_t)
            buf_rews.append(rew_t)
            buf_prbs.append(probs_t)

        gs_tensor = torch.cat(gs_list, dim=0)

        # ── Assemble buffer (same format as original EpisodeRunner) ───────────
        return {
            # We store edge-level tensors — PPO update iterates over all edges
            "obs":    batch.edge_attr,                 # (E_total, 10) raw feats
            "acts":   torch.cat(buf_acts,  dim=0).unsqueeze(1),  # (E, 1)
            "rews":   torch.cat(buf_rews,  dim=0),     # (E,)
            "lps":    torch.cat(buf_lps,   dim=0),     # (E,)
            "gs":     gs_tensor,                       # (E, gs_dim)
            "probs":  torch.cat(buf_prbs,  dim=0),     # (E,) for PPO ratio
            "batch":  batch,                           # full PyG batch (for actor re-forward)
            "ep_tp": ep_tp, "ep_tn": ep_tn,
            "ep_fp": ep_fp, "ep_fn": ep_fn,
        }

    def _mask(self, act, ml_pred, conf):
        """Identical honest masking as original EpisodeRunner._mask()"""
        act = act.copy()
        if self.fp_win and self.fn_win:
            fp_r = float(np.mean(self.fp_win))
            fn_r = float(np.mean(self.fn_win))
            if fp_r > 0.25: self.nk_thr = min(0.92, self.nk_thr + 0.05)
            elif fp_r < 0.15: self.nk_thr = max(0.85, self.nk_thr - 0.01)
            if fn_r > 0.05: self.mp_thr = min(0.99, self.mp_thr + 0.05)
            elif fn_r < 0.02: self.mp_thr = max(0.95, self.mp_thr - 0.01)

        ad_normal = np.where(ml_pred == 0)[0]
        if len(ad_normal):
            min_keep = max(1, int(len(ad_normal) * self.nk_thr))
            n_keeps  = int((act[ad_normal] == 1).sum())
            if n_keeps < min_keep:
                prune = ad_normal[act[ad_normal] == 0]
                if len(prune):
                    flip = prune[np.argsort(conf[prune])[:min_keep - n_keeps]]
                    act[flip] = 1

        ad_mal = np.where(ml_pred == 1)[0]
        if len(ad_mal):
            min_prune = max(1, int(len(ad_mal) * self.mp_thr))
            n_prunes  = int((act[ad_mal] == 0).sum())
            if n_prunes < min_prune:
                keep = ad_mal[act[ad_mal] == 1]
                if len(keep):
                    flip = keep[np.argsort(-conf[keep])[:min_prune - n_prunes]]
                    act[flip] = 0
        return act


# ══════════════════════════════════════════════════════════════════════════════

    def set_mask_mode(self, mode: str):
        """
        Switch masking mode at runtime.
          'honest' : use AD prediction (default — no privileged info)
          'none'   : no masking        (pure RL)
        Called automatically by train.py when eval_F1 crosses AD baseline.
        """
        assert mode in ('honest', 'none'), f"Unknown mask_mode: {mode}"
        if mode != self.mask_mode:
            print(f"[MASK] Switching mask mode: {self.mask_mode} → {mode}")
            self.mask_mode = mode

    def _mask(self, act, true_label, conf):
        act = act.copy()
        true_normal = np.where(true_label == 0)[0]
        if len(true_normal):
            min_keep = max(1, int(len(true_normal) * self.nk_thr))
            n_keeps  = int((act[true_normal] == 1).sum())
            if n_keeps < min_keep:
                prune = true_normal[act[true_normal] == 0]
                if len(prune):
                    flip = prune[np.argsort(conf[prune])[:min_keep - n_keeps]]
                    act[flip] = 1
        true_mal = np.where(true_label == 1)[0]
        if len(true_mal):
            min_prune = max(1, int(len(true_mal) * self.mp_thr))
            n_prunes  = int((act[true_mal] == 0).sum())
            if n_prunes < min_prune:
                keep = true_mal[act[true_mal] == 1]
                if len(keep):
                    flip = keep[np.argsort(-conf[keep])[:min_prune - n_prunes]]
                    act[flip] = 0
        return act

#  PPO UPDATE  (GNN-aware, FAST — GNN runs once per epoch, not per mini-batch)
# ══════════════════════════════════════════════════════════════════════════════

def ppo_update_gnn(actor, critic, actor_opt, critic_opt, buf, args) -> Dict:
    """
    KEY FIX for speed:
      OLD (slow): actor(full_batch) called inside every mini-batch loop
                  → ppo_epochs × ceil(N/batch_size) GNN forward passes
      NEW (fast): actor(full_batch) called ONCE per epoch → node embeddings cached
                  → mini-batches work on cached embeddings (pure MLP, tiny cost)
                  → ppo_epochs GNN forward passes total (e.g. 4 instead of 300+)

    How it works:
      GNNActor.forward_embeddings() runs the GATv2 layers and returns node
      embeddings h (N_nodes, hidden). The edge decoder is then a small MLP
      that takes concat(h[src], h[dst], edge_attr) — we can call that separately
      on any subset of edges. So we cache h once and reuse it across mini-batches.
    """
    import torch.nn as nn

    acts = buf["acts"].squeeze(1)   # (E,)
    rews = buf["rews"]              # (E,)
    lps  = buf["lps"]               # (E,)
    gs   = buf["gs"]                # (E, gs_dim)
    b    = buf["batch"]             # PyG Batch
    N    = len(acts)

    # Pre-compute src/dst index arrays for the full batch (used in mini-batches)
    src_all = b.edge_index[0]   # (E,)
    dst_all = b.edge_index[1]   # (E,)
    ea_all  = b.edge_attr       # (E, 10)

    # ── GAE with per-timestep advantage normalization ────────────────────────────
    # Normalizing advantages per-timestep (not over entire chunk) prevents
    # edges from different graph snapshots interfering with each other's credit.
    # Each timestep's edges are normalized independently, then concatenated.
    with torch.no_grad():
        vals = critic(gs).squeeze(-1)
        adv  = torch.zeros(N, device=acts.device)
        gae  = 0.0
        for t in reversed(range(N)):
            nv    = vals[t+1] if t < N-1 else vals[t]
            delta = rews[t] + args.gamma * nv - vals[t]
            gae   = delta + args.gamma * args.gae_lambda * gae
            adv[t] = gae
        ret = adv + vals
        # Per-timestep normalization using the batch structure
        # Each graph in the PyG batch corresponds to one timestep
        adv_norm = torch.zeros_like(adv)
        ptr = b.ptr  # PyG batch pointer: ptr[i]..ptr[i+1] = nodes of graph i
        # Map edges back to their graph using edge_index and ptr
        # edge belongs to graph g if ptr[g] <= src_node < ptr[g+1]
        src_nodes = b.edge_index[0]
        graph_ids = torch.zeros(N, dtype=torch.long, device=acts.device)
        for g in range(len(ptr) - 1):
            mask = (src_nodes >= ptr[g]) & (src_nodes < ptr[g+1])
            if mask.sum() > 1:
                a_g = adv[mask]
                adv_norm[mask] = (a_g - a_g.mean()) / (a_g.std() + 1e-8)
            elif mask.sum() == 1:
                adv_norm[mask] = 0.0
        adv = adv_norm

    al_s = cl_s = en_s = n_up = 0

    for _ in range(args.ppo_epochs):

        # ONE GNN pass per epoch — detached so mini-batch steps don't
        # corrupt the compute graph. edge_dec runs fresh each mini-batch.
        with torch.no_grad():
            h = actor.forward_embeddings(b)   # (total_nodes, hidden_dim)

        idx = torch.randperm(N, device=acts.device)
        actor_opt.zero_grad()   # accumulate across mini-batches, step once

        epoch_al = epoch_ent = 0.0
        n_mb = 0

        for s in range(0, N, args.batch_size):
            bidx = idx[s:s + args.batch_size]

            # Re-run edge_dec on this mini-batch (tiny MLP, very fast)
            edge_input = torch.cat([
                h[src_all[bidx]],
                h[dst_all[bidx]],
                ea_all[bidx],
            ], dim=1)                                         # (B, 2*hidden+10)
            p = actor.edge_dec(edge_input).squeeze(-1)        # (B,)

            dist_b = torch.distributions.Bernoulli(p)
            nlp    = dist_b.log_prob(acts[bidx])
            ent    = dist_b.entropy().mean()

            ratio = torch.exp(nlp - lps[bidx])
            s1    = ratio * adv[bidx]
            s2    = torch.clamp(ratio, 1 - args.clip_eps,
                                       1 + args.clip_eps) * adv[bidx]
            al    = -(torch.min(s1, s2).mean()) - args.entropy_coef * ent
            al.backward()   # accumulate grads — no retain_graph needed

            epoch_al  += al.item()
            epoch_ent += ent.item()
            n_mb      += 1

            # Critic: update with value clipping (PPO standard)
            # Prevents large value jumps that destabilize advantage estimates
            vp      = critic(gs[bidx]).squeeze(-1)
            vp_old  = vals[bidx]
            vp_clip = vp_old + torch.clamp(vp - vp_old,
                                           -args.clip_eps, args.clip_eps)
            cl_unclipped = nn.MSELoss()(vp,      ret[bidx])
            cl_clipped   = nn.MSELoss()(vp_clip, ret[bidx])
            cl = torch.max(cl_unclipped, cl_clipped) * args.value_coef
            critic_opt.zero_grad(); cl.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
            critic_opt.step()
            cl_s += cl.item()
            n_up += 1

        # One actor step per epoch
        nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
        actor_opt.step()

        al_s += epoch_al  / max(n_mb, 1)
        en_s += epoch_ent / max(n_mb, 1)

    return {
        "actor_loss":  al_s  / max(args.ppo_epochs, 1),
        "critic_loss": cl_s  / max(n_up, 1),
        "entropy":     en_s  / max(args.ppo_epochs, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  INTEGRATION GUIDE (what to change in train.py)
# ══════════════════════════════════════════════════════════════════════════════

INTEGRATION_NOTES = """
CHANGES NEEDED IN train.py
═══════════════════════════

1. IMPORTS — add at top:
   from gnn_mappo import GNNActor, GNNEpisodeRunner, ppo_update_gnn

2. REPLACE Actor instantiation:
   # OLD
   actor = Actor(obs_dim, act_dim, args.hidden_dim).to(dev)

   # NEW
   actor = GNNActor(hidden_dim=args.hidden_dim, heads=4).to(dev)

   # obs_dim / act_dim args no longer needed for actor
   # (they're still used by critic — keep gs_dim the same)

3. REPLACE EpisodeRunner:
   # OLD
   runner = EpisodeRunner(store, actor, None, tracker, ...)

   # NEW
   runner = GNNEpisodeRunner(store, actor, None, tracker,
               max_agents=args.max_agents,
               max_edges_per_agent=args.max_edges_per_agent,
               chunk_size=args.chunk_size,
               device=dev,
               use_action_masking=args.use_action_masking)

4. REPLACE ppo_update call:
   # OLD
   losses = ppo_update(actor, critic, actor_opt, critic_opt, buf, args)

   # NEW
   losses = ppo_update_gnn(actor, critic, actor_opt, critic_opt, buf, args)

5. CHECKPOINT — add to save/load:
   # GNNActor has same save/load interface as MLP Actor
   # No changes needed to save_checkpoint() / load_latest_checkpoint()

6. RECOMMENDED ARGS for speed:
   --hidden_dim 128        (was 256 — GNN expressive enough at 128)
   --ppo_epochs 3          (was 4  — saves ~25% PPO time)
   --chunk_size 300        (was 500 — smaller graphs per batch = more GPU parallelism)
   --batch_size 4096       (larger PPO minibatch — edges are cheap)

MEMORY NOTE:
  GNNActor forward re-runs during PPO update (ppo_update_gnn).
  The PyG Batch is kept in buf["batch"] — it stays on GPU.
  For chunk_size=300 with ~130 edges/step → ~39k edges/batch → ~20MB.
  Safe for any GPU with >= 8GB.
"""


if __name__ == "__main__":
    # ── Quick sanity check ────────────────────────────────────────────────────
    if not HAS_PYG:
        print("Install torch_geometric first")
        exit(1)

    print("=== GNNActor sanity check ===")
    dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = GNNActor(hidden_dim=128, heads=4).to(dev)

    # Fake graph: 5 vehicles, 8 edges
    n_nodes = 5; n_edges = 8
    x          = torch.randn(n_nodes, 10).to(dev)
    edge_index = torch.randint(0, n_nodes, (2, n_edges)).to(dev)
    edge_attr  = torch.randn(n_edges, 10).to(dev)

    from torch_geometric.data import Data, Batch
    g     = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    batch = Batch.from_data_list([g, g])   # 2 graphs in batch
    batch = batch.to(dev)

    probs = actor(batch)
    print(f"Input  : {n_nodes*2} nodes, {n_edges*2} edges (2 graphs)")
    print(f"Output : probs shape = {probs.shape}  (expected {n_edges*2})")
    print(f"Range  : [{probs.min():.3f}, {probs.max():.3f}]")
    print("✓ GNNActor forward pass OK")

    n_params = sum(p.numel() for p in actor.parameters())
    print(f"Parameters: {n_params:,}")