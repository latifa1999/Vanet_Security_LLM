#!/usr/bin/env python3
"""
Real-time anomaly detection server for the SUMO/OMNeT++/Veins co-simulation.

"""

import os
import pickle
import socket
import struct
from argparse import Namespace
from collections import deque

import numpy as np
import pandas as pd
import torch

from exp.exp_classification import Exp_Classification


# ── Wire format ───────────────────────────────────────────────────────────────
# 100 bytes sender_id | 100 bytes receiver_id | 80 bytes (10 doubles)
# | 42 bytes attack_type | 4 bytes is_malicious
MSG_SIZE = 326

FEATURE_COLUMNS = ['posx', 'posy', 'spdx', 'spdy', 'aclx', 'acly', 'hedx', 'hedy']


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

            malicious_data = data[pos:pos + 4]
            self.is_malicious = struct.unpack('i', malicious_data)[0] if len(malicious_data) == 4 else 0

        except Exception as e:
            print(f"Error unpacking data: {e}")
            raise

    def as_feature_vector(self):
        """Ordered exactly as FEATURE_COLUMNS."""
        return np.array([
            self.pos_x, self.pos_y,
            self.spd_x, self.spd_y,
            self.acl_x, self.acl_y,
            self.hed_x, self.hed_y,
        ], dtype=np.float32)


class SequenceBuffer:
    """
    Rolling window of the most recent messages, normalized with the scaler that
    was fitted during training.

    """

    def __init__(self, scaler_path, seq_len=96, per_sender=False):
        self.seq_len = seq_len
        self.per_sender = per_sender
        self.scaler = self._load_scaler(scaler_path)

        self._global = deque(maxlen=seq_len)
        self._by_sender = {}
        self.n_seen = 0

    @staticmethod
    def _load_scaler(path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Scaler not found at {path}.\n"
                "MIST must use the StandardScaler fitted during training — "
                "re-normalizing with different statistics will degrade predictions.")
        try:
            import joblib
            scaler = joblib.load(path)
        except Exception:
            with open(path, "rb") as f:
                scaler = pickle.load(f)
        print(f"Loaded training scaler from {path}")
        print(f"  mean (first 3): {scaler.mean_[:3]}")
        print(f"  scale (first 3): {scaler.scale_[:3]}")
        return scaler

    def _buffer_for(self, sender_id):
        if not self.per_sender:
            return self._global
        if sender_id not in self._by_sender:
            self._by_sender[sender_id] = deque(maxlen=self.seq_len)
        return self._by_sender[sender_id]

    def push(self, msg_data):
        """
        Add one message and return the window to classify, shape (seq_len, 8).

        Until `seq_len` messages have arrived the window is left-padded by
        repeating the earliest available message. Predictions during this
        warm-up are less reliable; `is_warm()` reports when the buffer is full.
        """
        raw = msg_data.as_feature_vector().reshape(1, -1)
        scaled = self.scaler.transform(
            pd.DataFrame(raw, columns=FEATURE_COLUMNS)
        ).astype(np.float32)[0]

        buf = self._buffer_for(msg_data.sender_id)
        buf.append(scaled)
        self.n_seen += 1

        window = np.asarray(buf, dtype=np.float32)
        if len(window) < self.seq_len:
            pad = np.repeat(window[:1], self.seq_len - len(window), axis=0)
            window = np.vstack([pad, window])
        return window

    def is_warm(self, sender_id=None):
        buf = self._buffer_for(sender_id) if self.per_sender else self._global
        return len(buf) >= self.seq_len


def initialize_model(checkpoint_path, args):
    exp = Exp_Classification(args)
    exp.model = exp._build_model()

    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if 'state_dict' in state_dict and isinstance(state_dict['state_dict'], dict):
        state_dict = state_dict['state_dict']

    model_dict = exp.model.state_dict()
    matched = {k: v for k, v in state_dict.items()
               if k in model_dict and model_dict[k].shape == v.shape}

    missing = set(model_dict) - set(matched)
    if missing:
        print(f"WARNING: {len(missing)} parameters not loaded from the checkpoint.")
        print(f"  e.g. {sorted(missing)[:5]}")
        print("  A large count usually means the checkpoint belongs to a different "
              "architecture (for example an Informer checkpoint).")

    model_dict.update(matched)
    exp.model.load_state_dict(model_dict)
    exp.model.to(args.device)
    exp.model.eval()
    return exp


@torch.no_grad()
def process_and_predict(exp, msg_data, buffer):
    """Returns (prediction, confidence) where confidence is P(malicious)."""
    try:
        window = buffer.push(msg_data)                       # (seq_len, 8)
        batch_x = torch.from_numpy(window).unsqueeze(0)      # (1, seq_len, 8)
        batch_x = batch_x.to(exp.args.device)

        # MIST takes a single tensor 
        outputs = exp.model(batch_x)                         # (1, num_classes)
        probs = torch.softmax(outputs, dim=1)
        prediction = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, 1].item())               # P(malicious)

        return prediction, confidence

    except Exception as e:
        print(f"Error in processing: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0.0


def handle_client(conn, exp, buffer, verbose=False):
    try:
        data = b''
        remaining = MSG_SIZE
        while remaining > 0:
            chunk = conn.recv(remaining)
            if not chunk:
                print("Connection closed by client")
                return
            data += chunk
            remaining -= len(chunk)

        msg_data = MessageData(data)
        prediction, confidence = process_and_predict(exp, msg_data, buffer)
        conn.sendall(struct.pack('i', prediction))

        if verbose:
            warm = "" if buffer.is_warm(msg_data.sender_id) else "  [warming up]"
            print(f"{msg_data.sender_id} -> "
                  f"{'Malicious' if prediction == 1 else 'Normal'} "
                  f"(p={confidence:.3f})  true={msg_data.is_malicious}{warm}")

    except Exception as e:
        print(f"Error handling client: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()


def main():
    # ── MIST configuration — must match the values used at training time ──────
    args = Namespace(**{
        "device":         torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "use_gpu":        torch.cuda.is_available(),
        "use_multi_gpu":  False,

        "model":          "Mist",
        "task_name":      "classification",
        "data":           "Custom",
        "model_id":       "mist_realtime",

        # MIST-specific 
        "seq_len":        96,
        "period_len":     4,
        "enc_in":         8,
        "num_classes":    2,

        "pred_len":       0,
        "label_len":      0,
        "dec_in":         8,
        "c_out":          2,
        "num_class":      2,
        "features":       "M",
        "freq":           "h",
        "checkpoints":    "./checkpoints/",
        "d_model":        128,
        "n_heads":        8,
        "e_layers":       2,
        "d_layers":       1,
        "d_ff":           2048,
        "moving_avg":     25,
        "factor":         1,
        "distil":         False,
        "dropout":        0.1,
        "embed":          "timeF",
        "activation":     "gelu",
        "output_attention": False,
        "num_workers":    0,
        "batch_size":     256,
    })

    # ── Paths ─────────────────────────────────────────────────────────────────
    CHECKPOINT_PATH = "/home/latifa.elbouga/lustre/vr_outsec-vh2sz1t4fks/users/latifa.elbouga/AD_models/checkpoints/Generated_MIST_Generated/checkpoint.pth"
    SCALER_PATH     = "/home/latifa.elbouga/lustre/vr_outsec-vh2sz1t4fks/users/latifa.elbouga/AD_models/scalers/veremi_scaler.pkl"

    print("Initializing MIST...")
    buffer = SequenceBuffer(SCALER_PATH, seq_len=args.seq_len, per_sender=False)
    exp = initialize_model(CHECKPOINT_PATH, args)
    exp.args = args
    print(f"Model ready on {args.device}  "
          f"(seq_len={args.seq_len}, period_len={args.period_len})")

    HOST, PORT = '127.0.0.1', 5000
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(20)
        print(f"Server listening on {HOST}:{PORT}")
        print(f"First {args.seq_len} messages are warm-up (window not yet full).\n")

        while True:
            try:
                conn, _ = s.accept()
                handle_client(conn, exp, buffer, verbose=True)
            except KeyboardInterrupt:
                print("\nShutting down.")
                break
            except Exception as e:
                print(f"Error accepting connection: {e}")
                continue


if __name__ == "__main__":
    main()