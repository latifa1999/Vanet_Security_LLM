"""
Vectorized TRUE MULTI-AGENT Environment

Each vehicle (SenderID / ReceiverID) = one independent agent.
Runs num_envs episodes in parallel for speed.

"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque

from magrl.utils.data_loader import FastVANETDataLoader
from magrl.environment.reward_functions import RewardFunction, AdaptivePenaltyTracker


class VectorizedMultiAgentEnv:
    """
    Runs num_envs episodes in parallel.

    Architecture: TRUE MULTI-AGENT MAPPO
      - Each vehicle = independent agent
      - Local observations (edges the vehicle is involved in)
      - Local actions (PRUNE / KEEP per edge)
      - Local rewards
      - Global state for centralized critic
    """

    FEATURES_PER_EDGE = 10

    def __init__(self,
                 data_path:             str,
                 cache_path:            Optional[str] = None,
                 scaler_path:           Optional[str] = None,
                 reward_function:       Optional[RewardFunction] = None,
                 adaptive_tracker:      Optional[AdaptivePenaltyTracker] = None,
                 num_envs:              int   = 8,
                 max_agents:            int   = 50,
                 max_edges_per_agent:   int   = 10,
                 max_edges_total:       int   = 200,
                 use_action_masking:    bool  = False,
                 normal_keep_thr:       float = 0.75, 
                 malicious_prune_thr:   float = 0.80,   
                 verbose:               bool  = True):

        self.num_envs            = num_envs
        self.max_agents          = max_agents
        self.max_edges_per_agent = max_edges_per_agent
        self.max_edges_total     = max_edges_total
        self.use_action_masking  = use_action_masking
        self.normal_keep_thr     = normal_keep_thr
        self.malicious_prune_thr = malicious_prune_thr
        self.verbose             = verbose

        self.loader       = FastVANETDataLoader(data_path, scaler_path, cache_path)
        self.ml_baseline  = self.loader.ml_baseline

        # reward_function may be None initially (set via set_reward_function)
        self.reward_fn        = reward_function
        self.adaptive_tracker = adaptive_tracker

        self.obs_dim          = max_edges_per_agent * self.FEATURES_PER_EDGE
        self.action_dim       = max_edges_per_agent
        self.global_state_dim = max_edges_total * self.FEATURES_PER_EDGE

        self.steps    = np.zeros(num_envs, dtype=np.int32)
        self.ep_stats = [defaultdict(int) for _ in range(num_envs)]

        self.fp_windows = [deque(maxlen=100) for _ in range(num_envs)]
        self.fn_windows = [deque(maxlen=100) for _ in range(num_envs)]

        if verbose:
            print(f"\n[ENV] Vectorized Multi-Agent Environment")
            print(f"      Parallel envs      : {num_envs}")
            print(f"      Max agents/step    : {max_agents}")
            print(f"      Max edges/agent    : {max_edges_per_agent}")
            print(f"      Obs dim (per agent): {self.obs_dim}")
            print(f"      Global state dim   : {self.global_state_dim}")
            print(f"      Timesteps in data  : {len(self.loader)}")
            print(f"      Adaptive penalties : {'ON' if adaptive_tracker else 'OFF'}")
            print(f"      Action masking     : {'HONEST (ml_prediction only)' if use_action_masking else 'OFF'}")

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> List[Tuple[Dict, np.ndarray]]:
        self.steps.fill(0)
        for s in self.ep_stats:
            s.clear()
        return [self._build_obs(eid) for eid in range(self.num_envs)]

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action_dicts: List[Dict]) -> List[Tuple]:
        return [self._step_one(eid, action_dicts[eid])
                for eid in range(self.num_envs)]

    def _step_one(self, eid: int, action_dict: Dict) -> Tuple:
        step = self.steps[eid]

        if step >= len(self.loader):
            obs, gs = self._build_obs(eid)
            return obs, {}, {aid: True for aid in action_dict}, {'global_state': gs}

        graph  = self.loader.get_graph(step)
        edges  = graph['edges']

        agent_edges = defaultdict(list)
        for idx, e in enumerate(edges):
            agent_edges[e['sender_id']].append(idx)
            agent_edges[e['receiver_id']].append(idx)

        if self.use_action_masking:
            action_dict = self._mask_actions(action_dict, agent_edges, edges, eid)

        all_actions, all_labels, all_preds, all_confs = [], [], [], []
        agent_slices: Dict[int, Tuple[int, int]] = {}

        for aid, action in action_dict.items():
            edge_idxs = agent_edges[aid][:self.max_edges_per_agent]
            start = len(all_actions)
            for i, eidx in enumerate(edge_idxs):
                if i >= len(action):
                    break
                e = edges[eidx]
                all_actions.append(int(action[i]))
                all_labels.append(e['true_label'])    # used only for reward + metrics
                all_preds.append(e['prediction'])
                all_confs.append(e['confidence'])
            agent_slices[aid] = (start, len(all_actions))

        if all_actions and self.reward_fn is not None:
            result = self.reward_fn.compute_batch(
                all_actions, all_labels, all_preds, all_confs,
                normalize=True, adaptive_tracker=self.adaptive_tracker
            )
            rew_list = result['rewards']
            self.ep_stats[eid]['tp'] += result['tp']
            self.ep_stats[eid]['tn'] += result['tn']
            self.ep_stats[eid]['fp'] += result['fp']
            self.ep_stats[eid]['fn'] += result['fn']

            n = len(all_actions)
            self.fp_windows[eid].append(result['fp'] / max(n, 1))
            self.fn_windows[eid].append(result['fn'] / max(n, 1))
        else:
            result   = {}
            rew_list = []

        reward_dict = {aid: float(np.mean(rew_list[s:e])) if s < e else 0.0
                       for aid, (s, e) in agent_slices.items()}

        self.steps[eid] += 1
        done = (self.steps[eid] >= len(self.loader))

        obs_dict, global_state = self._build_obs(eid)
        done_dict = {aid: done for aid in action_dict}

        info = {'global_state': global_state,
                'step': int(self.steps[eid]),
                'n_agents': len(action_dict)}
        if 'penalty_scales' in result:
            info['penalty_scales'] = result['penalty_scales']
        if done:
            info.update(self._episode_metrics(eid))

        return obs_dict, reward_dict, done_dict, info

    # ── Observation builder ───────────────────────────────────────────────────

    def _build_obs(self, eid: int) -> Tuple[Dict, np.ndarray]:
        step = self.steps[eid]
        if step >= len(self.loader):
            return {}, np.zeros(self.global_state_dim, dtype=np.float32)

        graph = self.loader.get_graph(step)
        edges = graph['edges']

        agent_edges: Dict = defaultdict(list)
        for idx, e in enumerate(edges):
            agent_edges[e['sender_id']].append(idx)
            agent_edges[e['receiver_id']].append(idx)

        agent_ids = sorted(agent_edges.keys())[:self.max_agents]

        obs_dict: Dict = {}
        for aid in agent_ids:
            obs = np.zeros(self.obs_dim, dtype=np.float32)
            for i, eidx in enumerate(agent_edges[aid][:self.max_edges_per_agent]):
                e   = edges[eidx]
                off = i * self.FEATURES_PER_EDGE
                obs[off]               = e['prediction']
                obs[off + 1]           = e['confidence']
                obs[off + 2: off + 10] = e['features']
            obs_dict[aid] = obs

        global_state = np.zeros(self.global_state_dim, dtype=np.float32)
        for i, e in enumerate(edges[:self.max_edges_total]):
            off = i * self.FEATURES_PER_EDGE
            global_state[off]               = e['prediction']
            global_state[off + 1]           = e['confidence']
            global_state[off + 2: off + 10] = e['features']

        return obs_dict, global_state

    # ── HONEST Action masking ─────────────────────────────────────────────────

    def _mask_actions(self, action_dict: Dict, agent_edges: Dict,
                      edges: List[Dict], eid: int) -> Dict:
        """
        HONEST masking — uses ml_prediction + confidence ONLY.
        """
        # Adapt thresholds based on recent error rates for this env
        if self.fp_windows[eid] and self.fn_windows[eid]:
            fp_rate = float(np.mean(self.fp_windows[eid]))
            fn_rate = float(np.mean(self.fn_windows[eid]))

            if fp_rate > 0.25:
                self.normal_keep_thr = min(0.92, self.normal_keep_thr + 0.05)
            elif fp_rate < 0.15:
                self.normal_keep_thr = max(0.85, self.normal_keep_thr - 0.01)

            if fn_rate > 0.05:
                self.malicious_prune_thr = min(0.99, self.malicious_prune_thr + 0.05)
            elif fn_rate < 0.02:
                self.malicious_prune_thr = max(0.95, self.malicious_prune_thr - 0.01)

        masked = {}
        for aid, action in action_dict.items():
            act       = action.copy() if hasattr(action, 'copy') else np.array(action)
            edge_idxs = agent_edges[aid][:self.max_edges_per_agent]

            ml_preds    = [edges[i]['prediction'] for i in edge_idxs]  
            confidences = [edges[i]['confidence']  for i in edge_idxs]

            ad_normal_idx    = [i for i, p in enumerate(ml_preds) if p == 0]
            ad_malicious_idx = [i for i, p in enumerate(ml_preds) if p == 1]

            if ad_normal_idx:
                min_keep = max(1, int(len(ad_normal_idx) * self.normal_keep_thr))
                n_keeps  = sum(1 for i in ad_normal_idx if act[i] == 1)
                if n_keeps < min_keep:
                    prune_idx = sorted(
                        [i for i in ad_normal_idx if act[i] == 0],
                        key=lambda i: confidences[i])
                    for i in prune_idx[:min_keep - n_keeps]:
                        act[i] = 1
                        
            if ad_malicious_idx:
                min_prune = max(1, int(len(ad_malicious_idx) * self.malicious_prune_thr))
                n_prunes  = sum(1 for i in ad_malicious_idx if act[i] == 0)
                if n_prunes < min_prune:
                    keep_idx = sorted(
                        [i for i in ad_malicious_idx if act[i] == 1],
                        key=lambda i: confidences[i], reverse=True)
                    for i in keep_idx[:min_prune - n_prunes]:
                        act[i] = 0

            masked[aid] = act

        return masked

    # ── Episode metrics ───────────────────────────────────────────────────────

    def _episode_metrics(self, eid: int) -> Dict:
        s  = self.ep_stats[eid]
        tp, tn, fp, fn = s['tp'], s['tn'], s['fp'], s['fn']
        total = tp + tn + fp + fn

        acc  = (tp + tn) / max(total, 1)
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-8)

        return {
            'f1_score':       f1,
            'accuracy':       acc,
            'precision':      prec,
            'recall':         rec,
            'total_tp':       tp,
            'total_tn':       tn,
            'total_fp':       fp,
            'total_fn':       fn,
            'ml_baseline_f1': self.ml_baseline['f1'],
            'f1_improvement': f1 - self.ml_baseline['f1'],
            'n_agents':       self.max_agents,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_obs_dim(self)          -> int:  return self.obs_dim
    def get_action_dim(self)       -> int:  return self.action_dim
    def get_global_state_dim(self) -> int:  return self.global_state_dim
    def get_ml_baseline(self)      -> Dict: return self.ml_baseline

    def set_reward_function(self, rf: RewardFunction):
        self.reward_fn = rf
        if self.verbose:
            print(f"[ENV] Reward function updated → {rf.name}")