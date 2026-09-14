"""
fast_data_loader.py

Loads your CSV once, converts every timestep to numpy arrays, saves a
pickle cache. Subsequent runs load the cache instantly (~75ms for 7825 steps).

CSV columns expected:
  sendTime, SenderID, ReceiverID,
  posx, posy, spdx, spdy, aclx, acly, hedx, hedy,
  AttackType, label, Prediction, Prob_Normal, Prob_Malicious
"""

import os
import pickle
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = ['posx', 'posy', 'spdx', 'spdy', 'aclx', 'acly', 'hedx', 'hedy']

ATTACK_TYPES = {
    'normal_behavior': 0,
    'DoS_Disruptive': 1,
    'Random_speed': 2,
    'Constant_Position_Offset': 3,
    'Constant_Speed_Offset': 4,
    'Random_Position': 5,
    'Replay_Attack': 6,
}


class UltraFastDataLoader:
    """
    Per-graph format (new cache):
      edge_data  : (n_edges, 10)  float32   [pred, conf, 8 kinematics]
      labels     : (n_edges,)     int32
      preds      : (n_edges,)     int32
      send_ids   : (n_edges,)     int32     vehicle IDs encoded as ints
      recv_ids   : (n_edges,)     int32
      n_edges    : int
    """

    def __init__(
        self,
        data_path: str,
        scaler_path: Optional[str] = None,
        cache_path: Optional[str] = None,
    ):
        print("\n" + "=" * 68)
        print("ULTRA-FAST DATA LOADER")
        print("=" * 68)

        # ── Try cache ────────────────────────────────────────────────────────
        if cache_path and os.path.exists(cache_path):
            t0 = time.perf_counter()
            print(f"[CACHE] Loading {cache_path} ...")
            with open(cache_path, "rb") as f:
                c = pickle.load(f)

            # Reject old-format caches (no 'n_edges' key)
            if c["graphs"] and "n_edges" not in c["graphs"][0]:
                print("[CACHE] Old format detected – rebuilding...")
                os.remove(cache_path)
            else:
                self.graphs      = c["graphs"]
                self.ml_baseline = c["ml_baseline"]
                self.n_timesteps = len(self.graphs)
                self.max_edges   = c["max_edges"]
                ms = (time.perf_counter() - t0) * 1000
                print(f"[CACHE] {self.n_timesteps} timesteps loaded in {ms:.0f}ms ✓")
                print("=" * 68 + "\n")
                return

        # ── Load CSV ─────────────────────────────────────────────────────────
        print(f"[1] Loading {data_path}")
        t0 = time.perf_counter()
        df = pd.read_csv(data_path)
        # take a sample of 5000 rows for testing
        #df = df.sample(n=5000, random_state=42)
        print(f"    {len(df):,} rows   cols: {list(df.columns)}")
        self._validate(df)
        self.ml_baseline = self._baseline(df)
        df               = self._normalize(df, scaler_path, cache_path)

        # ── Build graphs ─────────────────────────────────────────────────────
        print(f"\n[3] Building numpy graphs …")
        feat_cols  = [c for c in FEATURE_COLS if c in df.columns]
        timestamps = sorted(df["sendTime"].unique())

        # Encode string vehicle IDs → ints once
        all_vids = sorted(set(df["SenderID"]) | set(df["ReceiverID"]))
        vid2int  = {v: i for i, v in enumerate(all_vids)}

        self.graphs: List[Dict] = []
        edge_counts: List[int]  = []

        for t in timestamps:
            tdf = df[df["sendTime"] == t]
            n   = len(tdf)
            edge_counts.append(n)

            ed          = np.zeros((n, 10), dtype=np.float32)
            ed[:, 0]    = tdf["Prediction"].values.astype(np.float32)
            ed[:, 1]    = tdf["Prob_Malicious"].values.astype(np.float32)
            for fi, col in enumerate(feat_cols):
                ed[:, 2 + fi] = tdf[col].values.astype(np.float32)

            self.graphs.append({
                "edge_data": ed,
                "labels":    tdf["label"].values.astype(np.int32),
                "preds":     tdf["Prediction"].values.astype(np.int32),
                "send_ids":  np.array([vid2int[v] for v in tdf["SenderID"]], dtype=np.int32),
                "recv_ids":  np.array([vid2int[v] for v in tdf["ReceiverID"]], dtype=np.int32),
                "n_edges":   n,
            })

        self.n_timesteps = len(self.graphs)
        self.max_edges   = int(max(edge_counts))
        elapsed = time.perf_counter() - t0

        print(f"    {self.n_timesteps} graphs  avg={np.mean(edge_counts):.1f}  "
              f"max={self.max_edges} edges  ({elapsed:.1f}s)")
        print(f"    AD baseline: F1={self.ml_baseline['f1']:.4f}  "
              f"FP={self.ml_baseline['fp']}  FN={self.ml_baseline['fn']}")

        # ── Save cache ───────────────────────────────────────────────────────
        if cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            print(f"\n[CACHE] Saving {cache_path} …")
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "graphs":      self.graphs,
                    "ml_baseline": self.ml_baseline,
                    "max_edges":   self.max_edges,
                }, f)
            print("[CACHE] Saved ✓  (next run = instant!)")

        print("=" * 68 + "\n")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _validate(self, df: pd.DataFrame):
        required = ["sendTime", "SenderID", "ReceiverID", "Prediction", "label"]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        if "Prob_Malicious" not in df.columns:
            df["Prob_Malicious"] = 0.5
            print("    Prob_Malicious not found – using 0.5")
        n0 = (df["label"] == 0).sum()
        n1 = (df["label"] == 1).sum()
        print(f"    Normal={n0:,}  Malicious={n1:,}")
        if "AttackType" in df.columns:
            print(f"    Attacks: {df['AttackType'].value_counts().to_dict()}")

    def _baseline(self, df: pd.DataFrame) -> Dict:
        print(f"\n[2] AD Model Baseline:")
        y, p = df["label"].values, df["Prediction"].values
        f1   = f1_score(y, p, zero_division=0)
        prec = precision_score(y, p, zero_division=0)
        rec  = recall_score(y, p, zero_division=0)
        cm   = confusion_matrix(y, p)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        print(f"    F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")
        print(f"    TP={tp}  TN={tn}  FP={fp}  FN={fn}")
        return {"f1": f1, "precision": prec, "recall": rec,
                "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}

    def _normalize(self, df: pd.DataFrame, scaler_path: Optional[str],
                   cache_path: Optional[str]) -> pd.DataFrame:
        print(f"\n[2b] Normalizing: {FEATURE_COLS}")
        cols = [c for c in FEATURE_COLS if c in df.columns]
        if not cols:
            return df
        if scaler_path and os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                sc = pickle.load(f)
            df[cols] = sc.transform(df[cols])
            print(f"    Loaded scaler from {scaler_path} ✓")
        else:
            sc = StandardScaler()
            df[cols] = sc.fit_transform(df[cols])
            if cache_path:
                sp = cache_path.replace(".pkl", "_scaler.pkl")
                with open(sp, "wb") as f:
                    pickle.dump(sc, f)
                print(f"    New scaler saved → {sp} ✓")
        return df

    # ── Public API ────────────────────────────────────────────────────────────

    def get_graph(self, t: int) -> Dict:
        return self.graphs[t]

    def __len__(self) -> int:
        return self.n_timesteps
