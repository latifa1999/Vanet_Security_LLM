# MAGRL with LLM-Guided Reward Optimisation

Multi-Agent Graph Reinforcement Learning for **post-detection mitigation** in
Vehicular Ad Hoc Networks, with reward functions generated and refined by a Large
Language Model during training.

This repository covers the **mitigation stage** of the framework described in
*"LLM-Guided Multi-Agent Graph Reinforcement Learning for Anomaly Detection and
Adaptive Mitigation in Vehicular Networks."* It consumes the per-message predictions
produced by the MIST anomaly detector, which lives in a separate repository.

---

## Overview

Anomaly detection identifies suspicious V2V messages but does not decide what the
network should do about them, and it makes mistakes. This repository learns that
decision.

At each simulation step the V2V traffic is represented as a **dynamic communication
graph**: vehicles are nodes, messages are directed edges. Every edge carries the
detector's prediction and confidence alongside the sender's kinematic state. Each
vehicle acts as an agent that decides, per link in its own neighbourhood, whether to
**keep** or **prune** the communication.

Training uses **MAPPO** under **Centralized Training with Decentralized Execution**:
agents share a graph-based policy and act on local observations, while a centralized
critic sees the full graph during training only.

Rather than hand-tuning reward coefficients, an LLM proposes candidate reward
functions as executable Python, which are compiled, validated, and swapped in during
training. A two-phase strategy explores diverse reward structures at high sampling
temperature, then refines the best one found at low temperature.

---

## Repository structure

```
MAGRL_with_LLM_Reward_Optimisation/
├── magrl/
│   ├── agents/
│   │   └── mappo_agent.py       # GNNActor (GATv2), episode runner, PPO update
│   ├── environment/
│   │   ├── magrl_env.py         # multi-agent environment
│   │   └── reward_functions.py  # R0 baseline, LLM-generated rewards, adaptive penalties
│   ├── llm_optimizer/
│   │   └── llm_reward_optimizer.py  # prompting, validation, two-phase search
│   └── utils/
│       └── data_loader.py       # CSV -> per-timestep graph tensors, with caching
├── outputs/                     # runs: checkpoints, metrics.csv, TensorBoard, rewards
├── train_magrl.py               # training entry point
├── evaluate.py                  # evaluate a trained checkpoint
└── requirements.txt
```

---

## Installation

Tested with Python 3.10 and PyTorch 2.x on Linux (NVIDIA A100).

```bash
git clone <repository-url>
cd MAGRL_with_LLM_Reward_Optimisation

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

**PyTorch Geometric needs a separate install** matched to your torch and CUDA
versions — a plain `pip install torch-geometric` often fails to resolve its
compiled dependencies:

```bash
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

Substitute your own versions; check with
`python -c "import torch; print(torch.__version__, torch.version.cuda)"`.

### LLM access

Reward generation calls a hosted model through the HuggingFace inference router.
Provide a token through the environment — **do not hardcode it in the source**:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

The default model is `Qwen/Qwen2.5-Coder-32B-Instruct`. Training can be run without
any LLM access using `--no_llm`, which keeps the fixed R0 reward throughout.

---

## Input data

The loader expects a CSV where each row is one V2V message **already scored by the
anomaly detector**:

| Column | Source | Description |
|---|---|---|
| `sendTime` | simulation | timestamp; defines the graph snapshots |
| `SenderID`, `ReceiverID` | simulation | edge endpoints |
| `posx`, `posy`, `spdx`, `spdy`, `aclx`, `acly`, `hedx`, `hedy` | simulation | sender kinematics |
| `label` | simulation | ground truth (0 = normal, 1 = malicious) |
| `Prediction` | detector | binary anomaly classification |
| `Prob_Malicious` | detector | classification confidence |
| `AttackType` | simulation | optional, used for per-attack analysis |

Rows sharing a `sendTime` form one graph. Each edge becomes a 10-dimensional feature
vector:

```
[Prediction, Prob_Malicious, posx, posy, spdx, spdy, aclx, acly, hedx, hedy]
```

Ground-truth labels are used **only** to compute rewards and metrics.

On first load the dataset is converted to numpy graphs and cached as
`vanet_cache.pkl` next to the CSV. Delete the cache after changing the data, or the
stale version is silently reused.

---

## Usage

### Training

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

python train_magrl.py \
  --data_path /path/to/data_with_predictions.csv \
  --output_dir ./outputs \
  --num_episodes 20000 \
  --chunk_size 1000 \
  --hidden_dim 128 \
```

Each run creates a timestamped directory under `outputs/` containing:

```
checkpoints/      best.pth, final.pth, periodic epN.pth
results/
  ├── rewards/    each generated reward as a .py file
  ├── rewards_log.txt
  ├── llm_history.json
  └── config.json
metrics.csv       per-episode F1, precision, recall, TP/TN/FP/FN, reward version
tb/               TensorBoard events
```

### Evaluation

```bash
python evaluate.py \
  --checkpoint outputs/<run>/checkpoints/best.pth \
  --data_path /path/to/scenario.csv
```

Reports the AD baseline and AD+MAGRL side by side on the same edge set, in both
weighted and malicious-class (binary) form.

### Ablations

```bash
# fixed hand-designed reward, no LLM
python train_magrl.py --no_llm ...
```

---

## Key arguments

| Argument | Default | Meaning |
|---|---|---|
| `--num_episodes` | 20000 | training episodes |
| `--chunk_size` | 1000 | timesteps per episode |
| `--hidden_dim` | 128 | GNN and critic width |
| `--lr_actor` / `--lr_critic` | 1e-5 / 5e-5 | learning rates |
| `--max_edges_total` | 130 | per-timestep edge cap; timesteps above this are truncated |
| `--max_edges_per_agent` | 30 | neighbourhood cap per agent |
| `--llm_interval` | 3000 | episodes between reward evaluations |
| `--llm_max_versions` | 6 | maximum reward versions generated |

### Action semantics

In the code, **`action = 0` means PRUNE and `action = 1` means KEEP**. A predicted
label is therefore `1 - action`. (The paper's Eq. 19 states the inverse convention;
the two are equivalent up to relabelling, but follow the code when reading this
repository.)

### Masking modes

- `honest` — biases actions using the detector's prediction and confidence only.
  This is what deployment would have available.
- `none` — raw policy output, no heuristic assistance.

### Note on the graph encoder

`GNNActor.freeze_gnn(True)` is called before training, which keeps the node/edge
encoders and GATv2 layers at their initialised values and trains only the edge
decoder. This was introduced to stabilise policy optimisation. Remove the call to
train the full actor end to end.

---

## Reward optimisation

Rewards are Python functions with a fixed signature:

```python
def compute_reward(action, true_label, ml_prediction, ml_confidence=0.5):
    ...
    return reward, category   # category in {'TP', 'TN', 'FP', 'FN'}
```

Every generated function is compiled, smoke-tested on the four action/label
combinations, and hashed against previous versions. Invalid or duplicate candidates
are rejected and the current reward stays active, so a bad generation never
interrupts training.

An `AdaptivePenaltyTracker` scales false-positive and false-negative penalties based
on rolling error rates. It applies on top of every reward, including R0, and is
independent of the LLM.

`R0_Baseline` in `reward_functions.py` is the fixed hand-designed reward used as the
no-LLM ablation.

---