#!/usr/bin/env python3
"""
Real-time AD + mitigation server for the SUMO/OMNeT++/Veins co-simulation.

"""

import os
import pickle
import signal
import socket
import struct
import sys
import time
import traceback
from argparse import Namespace
from collections import deque

import numpy as np
import pandas as pd
import torch

from exp.exp_classification import Exp_Classification
from magrl.agents.mappo_agent import GNNActor, build_pyg_graph

try:
    from torch_geometric.data import Batch
except ImportError:
    print("torch_geometric is required. Install it with an index URL matching "
          "your torch/CUDA build, e.g.\n"
          "  pip install torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu118.html")
    raise


MSG_SIZE = 326
FEATURE_COLUMNS = ['posx', 'posy', 'spdx', 'spdy', 'aclx', 'acly', 'hedx', 'hedy']


# ══════════════════════════════════════════════════════════════════════════════
#  MESSAGE PARSING
# ══════════════════════════════════════════════════════════════════════════════

class MessageData:
    def __init__(self, data):
        try:
            pos = 0

            self.sender_id = data[pos:pos + 100].split(b'\0')[0].decode('utf-8', errors='ignore')
            pos += 100
            self.receiver_id = data[pos:pos + 100].split(b'\0')[0].decode('utf-8', errors='ignore')
            pos += 100

            nums = struct.unpack('10d', data[pos:pos + 80])
            pos += 80
            self.pos_x, self.pos_y = nums[0], nums[1]
            self.spd_x, self.spd_y = nums[2], nums[3]
            self.acl_x, self.acl_y = nums[4], nums[5]
            self.hed_x, self.hed_y = nums[6], nums[7]
            self.simulation_time = nums[8]

            self.attack_type = data[pos:pos + 42].split(b'\0')[0].decode('utf-8', errors='ignore')
            pos += 42

            self.is_malicious = 0
            if len(data) - pos >= 4:
                self.is_malicious = struct.unpack('i', data[pos:pos + 4])[0]

            self.sender_int = self._to_int(self.sender_id)
            self.receiver_int = self._to_int(self.receiver_id)

        except Exception as e:
            print(f"Error unpacking data: {e}")
            traceback.print_exc()
            raise

    @staticmethod
    def _to_int(vid):
        digits = ''.join(filter(str.isdigit, vid))
        return int(digits) if digits else (hash(vid) % 100000)

    def as_feature_vector(self):
        return np.array([
            self.pos_x, self.pos_y,
            self.spd_x, self.spd_y,
            self.acl_x, self.acl_y,
            self.hed_x, self.hed_y,
        ], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  MIST — sequence buffer + inference
# ══════════════════════════════════════════════════════════════════════════════

class SequenceBuffer:
    """
    Rolling window of the most recent messages, normalized with the scaler fitted
    during MIST training.

    Windows follow arrival order rather than being grouped per vehicle, matching
    how the training windows were built in data_loader.py.
    """

    def __init__(self, scaler_path, seq_len=96):
        self.seq_len = seq_len
        self.scaler = self._load_scaler(scaler_path)
        self.buf = deque(maxlen=seq_len)

    @staticmethod
    def _load_scaler(path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Scaler not found at {path}. MIST must reuse the StandardScaler "
                "fitted during training — different statistics shift the input "
                "distribution away from what the model saw.")
        try:
            import joblib
            scaler = joblib.load(path)
        except Exception:
            with open(path, "rb") as f:
                scaler = pickle.load(f)
        print(f"[MIST] scaler loaded from {path}")
        return scaler

    def push(self, msg_data):
        raw = msg_data.as_feature_vector().reshape(1, -1)
        scaled = self.scaler.transform(
            pd.DataFrame(raw, columns=FEATURE_COLUMNS)
        ).astype(np.float32)[0]

        self.buf.append(scaled)
        window = np.asarray(self.buf, dtype=np.float32)
        if len(window) < self.seq_len:                       # warm-up: left-pad
            pad = np.repeat(window[:1], self.seq_len - len(window), axis=0)
            window = np.vstack([pad, window])
        return window, scaled

    def is_warm(self):
        return len(self.buf) >= self.seq_len


def initialize_mist(checkpoint_path, args):
    exp = Exp_Classification(args)
    exp.model = exp._build_model()

    ckpt = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)

    model_dict = exp.model.state_dict()
    matched = {k: v for k, v in state_dict.items()
               if k in model_dict and model_dict[k].shape == v.shape}
    missing = set(model_dict) - set(matched)
    if missing:
        print(f"[MIST] WARNING: {len(missing)}/{len(model_dict)} parameters not "
              f"loaded — e.g. {sorted(missing)[:4]}")
        print("[MIST] A large count usually means the checkpoint is from a "
              "different architecture (an Informer checkpoint, for example).")

    model_dict.update(matched)
    exp.model.load_state_dict(model_dict)
    exp.model.to(args.device)
    exp.model.eval()
    return exp


@torch.no_grad()
def mist_prediction(exp, msg_data, buffer):
    """Returns (prediction, P(malicious), scaled_features)."""
    try:
        window, scaled = buffer.push(msg_data)
        batch_x = torch.from_numpy(window).unsqueeze(0).to(exp.args.device)

        outputs = exp.model(batch_x)                 # MIST takes one tensor
        probs = torch.softmax(outputs, dim=1)
        pred = int(torch.argmax(probs, dim=1).item())
        conf = float(probs[0, 1].item())
        return pred, conf, scaled

    except Exception as e:
        print(f"Error in MIST inference: {e}")
        traceback.print_exc()
        return 0, 0.0, msg_data.as_feature_vector()


# ══════════════════════════════════════════════════════════════════════════════
#  MAGRL — rolling graph window + GATv2 actor
# ══════════════════════════════════════════════════════════════════════════════

class MAGRLEvaluator:
    """
    Keeps a rolling window of recent communication links, rebuilds the PyG graph
    on each message, and reads the keep/prune probability for the newest edge.
    """

    EDGE_DIM = 10   # [pred, conf, posx, posy, spdx, spdy, aclx, acly, hedx, hedy]

    def __init__(self, checkpoint_path, device,
                 max_edges=130, use_masking=True,
                 normal_keep_thr=0.90, malicious_prune_thr=0.95,
                 log_dir="magrl_logs"):
        self.device = device
        self.max_edges = max_edges
        self.use_masking = use_masking
        self.nk_thr = normal_keep_thr
        self.mp_thr = malicious_prune_thr

        self.edges = deque(maxlen=max_edges)   # (snd_int, rcv_int, feat_10)

        self.fp_win = deque(maxlen=100)
        self.fn_win = deque(maxlen=100)
        self.counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "online_decisions.csv")
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write("timestamp,sim_time,sender_id,receiver_id,"
                        "ml_pred,ml_prob,edge_prob,raw_action,final_action,"
                        "is_malicious\n")

        self.actor = self._load_actor(checkpoint_path)

    def _load_actor(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        sd = ckpt["actor"] if "actor" in ckpt else ckpt

        if "node_enc.0.weight" not in sd:
            raise KeyError(
                "Checkpoint does not contain a GNNActor (no 'node_enc.0.weight'). "
                "A stable-baselines3 .zip from the single-agent GRL work will not "
                "load here — the architectures are unrelated.")

        hidden_dim = sd["node_enc.0.weight"].shape[0]
        heads = 4
        for k in ("conv1.att", "conv1.att_src", "conv1.att_l"):
            if k in sd and sd[k].dim() >= 2:
                heads = sd[k].shape[1]
                break

        actor = GNNActor(hidden_dim=hidden_dim, heads=heads).to(self.device)
        actor.load_state_dict(sd)
        actor.eval()
        print(f"[MAGRL] GNNActor loaded  (hidden_dim={hidden_dim}, heads={heads}, "
              f"{sum(p.numel() for p in actor.parameters()):,} params)")
        return actor

    # ── graph construction ───────────────────────────────────────────────────

    def _push_edge(self, msg_data, ml_pred, ml_prob, scaled_feats):
        feat = np.empty(self.EDGE_DIM, dtype=np.float32)
        feat[0] = float(ml_pred)
        feat[1] = float(ml_prob)
        feat[2:] = scaled_feats          # same scaler as MIST, as in training
        self.edges.append((msg_data.sender_int, msg_data.receiver_int, feat))

    def _build_graph(self):
        ed = torch.tensor(np.stack([e[2] for e in self.edges]),
                          dtype=torch.float32, device=self.device)
        snd = torch.tensor([e[0] for e in self.edges],
                           dtype=torch.long, device=self.device)
        rcv = torch.tensor([e[1] for e in self.edges],
                           dtype=torch.long, device=self.device)
        # Reuse the training-time builder so node features match exactly
        # (sender-role mean | receiver-role mean -> 20-dim).
        return build_pyg_graph(ed, snd, rcv, self.device)

    # ── honest action masking ────────────────────────────────────────────────

    def _mask(self, acts, ml_preds, confs):
        acts = acts.copy()

        if self.fp_win and self.fn_win:
            fp_r, fn_r = float(np.mean(self.fp_win)), float(np.mean(self.fn_win))
            if fp_r > 0.25:
                self.nk_thr = min(0.92, self.nk_thr + 0.05)
            elif fp_r < 0.15:
                self.nk_thr = max(0.85, self.nk_thr - 0.01)
            if fn_r > 0.05:
                self.mp_thr = min(0.99, self.mp_thr + 0.05)
            elif fn_r < 0.02:
                self.mp_thr = max(0.95, self.mp_thr - 0.01)

        ad_normal = np.where(ml_preds == 0)[0]
        if len(ad_normal):
            min_keep = max(1, int(len(ad_normal) * self.nk_thr))
            n_keeps = int((acts[ad_normal] == 1).sum())
            if n_keeps < min_keep:
                prune = ad_normal[acts[ad_normal] == 0]
                if len(prune):
                    flip = prune[np.argsort(confs[prune])[:min_keep - n_keeps]]
                    acts[flip] = 1

        ad_mal = np.where(ml_preds == 1)[0]
        if len(ad_mal):
            min_prune = max(1, int(len(ad_mal) * self.mp_thr))
            n_prunes = int((acts[ad_mal] == 0).sum())
            if n_prunes < min_prune:
                keep = ad_mal[acts[ad_mal] == 1]
                if len(keep):
                    flip = keep[np.argsort(-confs[keep])[:min_prune - n_prunes]]
                    acts[flip] = 0
        return acts

    def _update_rates(self, action, is_malicious):
        if action == 0 and is_malicious == 1:
            self.counts["tp"] += 1
        elif action == 1 and is_malicious == 0:
            self.counts["tn"] += 1
        elif action == 0 and is_malicious == 0:
            self.counts["fp"] += 1
        elif action == 1 and is_malicious == 1:
            self.counts["fn"] += 1

        total = sum(self.counts.values())
        if total >= 10:
            self.fp_win.append(self.counts["fp"] / max(total, 1))
            self.fn_win.append(self.counts["fn"] / max(total, 1))
            self.counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    # ── inference ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, msg_data, ml_pred, ml_prob, scaled_feats):
        """Returns the action for the current message: 0 = PRUNE, 1 = KEEP."""
        try:
            self._push_edge(msg_data, ml_pred, ml_prob, scaled_feats)

            graph = self._build_graph()
            batch = Batch.from_data_list([graph]).to(self.device)

            probs = self.actor(batch).cpu().numpy()      # (E,)
            acts = (probs > 0.5).astype(np.int32)
            raw_action = int(acts[-1])                   # newest edge is last

            if self.use_masking:
                ml_preds = np.array([e[2][0] for e in self.edges], dtype=np.int32)
                confs = np.array([e[2][1] for e in self.edges], dtype=np.float32)
                acts = self._mask(acts, ml_preds, confs)

            final_action = int(acts[-1])
            self._update_rates(final_action, msg_data.is_malicious)

            with open(self.log_file, 'a') as f:
                f.write(f"{time.time()},{msg_data.simulation_time},"
                        f"{msg_data.sender_id},{msg_data.receiver_id},"
                        f"{ml_pred},{ml_prob:.6f},{probs[-1]:.6f},"
                        f"{raw_action},{final_action},{msg_data.is_malicious}\n")

            return final_action, float(probs[-1]), raw_action

        except Exception as e:
            print(f"Error in MAGRL evaluation: {e}")
            traceback.print_exc()
            return 1, 0.0, 1        # fail open: keep the link


# ══════════════════════════════════════════════════════════════════════════════
#  SERVER
# ══════════════════════════════════════════════════════════════════════════════

def handle_client(conn, mist_exp, seq_buffer, magrl, verbose=True):
    try:
        data = b''
        remaining = MSG_SIZE
        while remaining > 0:
            chunk = conn.recv(remaining)
            if not chunk:
                return
            data += chunk
            remaining -= len(chunk)

        msg = MessageData(data)

        ml_pred, ml_prob, scaled = mist_prediction(mist_exp, msg, seq_buffer)
        action, edge_prob, raw_action = magrl.evaluate(msg, ml_pred, ml_prob, scaled)

        should_prune = 1 if action == 0 else 0
        conn.sendall(struct.pack('ii', ml_pred, should_prune))

        if verbose:
            warm = "" if seq_buffer.is_warm() else " [warming up]"
            flag = " *masked*" if raw_action != action else ""
            print(f"{msg.sender_id}->{msg.receiver_id}  "
                  f"MIST={ml_pred}(p={ml_prob:.3f})  "
                  f"MAGRL={'KEEP' if action == 1 else 'PRUNE'}(p={edge_prob:.3f})"
                  f"{flag}  true={msg.is_malicious}{warm}")

    except Exception as e:
        print(f"Error handling client: {e}")
        traceback.print_exc()
    finally:
        conn.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── MIST configuration ──────────────────────────────
    args = Namespace(**{
        "device": device,
        "use_gpu": device.type == "cuda",
        "use_multi_gpu": False,

        "model": "Mist",
        "task_name": "classification",
        "data": "Custom",
        "model_id": "mist_realtime",

        "seq_len": 96,          
        "period_len": 4,        
        "enc_in": 8,
        "num_classes": 2,

        "pred_len": 0, "label_len": 0, "dec_in": 8, "c_out": 2, "num_class": 2,
        "features": "M", "freq": "h", "checkpoints": "./checkpoints/",
        "d_model": 128, "n_heads": 8, "e_layers": 2, "d_layers": 1,
        "d_ff": 2048, "moving_avg": 25, "factor": 1, "distil": False,
        "dropout": 0.1, "embed": "timeF", "activation": "gelu",
        "output_attention": False, "num_workers": 0, "batch_size": 256,
    })

    # ── Paths ─────────────────────────────────────────────────────────────────
    MIST_CHECKPOINT  = "/home/latifa.elbouga/lustre/vr_outsec-vh2sz1t4fks/users/latifa.elbouga/AD_models/checkpoints/Generated_MIST_Generated/checkpoint.pth"
    SCALER_PATH      = "/home/latifa.elbouga/lustre/vr_outsec-vh2sz1t4fks/users/latifa.elbouga/AD_models/scalers/veremi_scaler.pkl"
    MAGRL_CHECKPOINT = "/home/latifa.elbouga/lustre/vr_outsec-vh2sz1t4fks/users/latifa.elbouga/Vanet_Security_LLM/MAGRL_with_LLM_Reward_Optimisation/outputs/mappo_gnn_20260812_155301/checkpoints/best.pth"

    print("Loading MIST ...")
    seq_buffer = SequenceBuffer(SCALER_PATH, seq_len=args.seq_len)
    mist_exp = initialize_mist(MIST_CHECKPOINT, args)
    mist_exp.args = args

    print("Loading MAGRL ...")
    magrl = MAGRLEvaluator(
        MAGRL_CHECKPOINT,
        device=device,
        max_edges=130,            
        use_masking=False,
    )

    def shutdown(sig, frame):
        print("\nShutting down.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    HOST, PORT = '127.0.0.1', 5000
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(20)
        print(f"\nServer listening on {HOST}:{PORT}  (device: {device})")
        print(f"Warm-up: first {args.seq_len} messages fill the MIST window; "
              f"the graph window fills over the first {magrl.max_edges} links.\n")

        while True:
            try:
                conn, _ = s.accept()
                handle_client(conn, mist_exp, seq_buffer, magrl)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error accepting connection: {e}")
                continue


if __name__ == "__main__":
    main()