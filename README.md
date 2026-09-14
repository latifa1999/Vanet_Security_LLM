# Vanet_Security_LLM

**LLM-Guided Multi-Agent Graph Reinforcement Learning for Anomaly Detection and
Adaptive Mitigation in Vehicular Networks**

An end-to-end VANET security framework that does not stop at detection. Malicious
V2V messages are identified by a lightweight time-series classifier, and the
network then decides — cooperatively, link by link — what to do about them.

---

## Overview

Anomaly detection in vehicular networks is passive: it flags suspicious messages
but says nothing about how the network should respond. This project closes that
loop with three components:

**MIST** (Multi-scale Inter-period Sparse-Temporal Network) classifies messages
from the temporal evolution of vehicle behaviour across consecutive V2V
transmissions, rather than judging each message in isolation.

**MAGRL** represents the network as a dynamic communication graph — vehicles as
nodes, messages as edges — and learns cooperative keep/prune decisions per link.
Each vehicle acts as an agent on its own neighbourhood, trained with MAPPO under
centralized training with decentralized execution.

**LLM-guided reward optimization** replaces hand-tuned reward coefficients. During
training, an LLM proposes reward functions as executable code, which are validated
and swapped in; a two-phase strategy explores diverse formulations before refining
the best one found.

<img width="1593" height="633" alt="pipline_LLM_v3" src="https://github.com/user-attachments/assets/8a714b44-eeba-487d-8236-dbb2ee511b9b" />

*Offline training (left): simulated V2V traffic is classified by MIST, the
predictions become edge features in the communication graph, and MAGRL learns
mitigation policies under rewards the LLM generates and refines from observed
performance. Online inference (right): the trained detector and policy run inside
the live simulation, pruning or preserving links in real time.*

---

## Repository layout

| Directory | Contents |
|---|---|
| [`MIST_Anomaly_Detection/`](MIST_Anomaly_Detection) | MIST training and benchmarking against time-series baselines |
| [`MAGRL_with_LLM_Reward_Optimisation/`](MAGRL_with_LLM_Reward_Optimisation) | MAPPO + GATv2 mitigation policy, LLM reward search |
| [`Omnet++_Sumo_Veins/`](Omnet++_Sumo_Veins) | Co-simulation: road networks, attack injection, V2V traffic |
| [`Python_C++/`](Python_C++) | Socket bridge between the simulation and the Python models |

Each directory has its own README with setup and usage details.

---

## How the pieces connect

```
  OMNeT++ / SUMO / Veins
      simulation
           │
           │  V2V messages + labels  (CSV, offline)
           ▼
  ┌──────────────────┐
  │  MIST classifier │  →  Prediction, Prob_Malicious
  └──────────────────┘
           │
           │  predictions become edge features
           ▼
  ┌──────────────────┐        ┌────────────────────┐
  │  MAGRL (MAPPO)   │ ◄────► │  LLM reward search │
  │  GATv2 policy    │        │  (training only)   │
  └──────────────────┘        └────────────────────┘
           │
           │  keep / prune per link
           ▼
     back into the simulation      (TCP socket, online)
```

The simulation produces labelled V2V traffic. MIST is trained on it and used to
score every message. Those scores, together with vehicle kinematics, form the edge
features of the communication graph that MAGRL learns over. At inference time both
trained models run behind a socket server the simulation queries on every
transmission.

The LLM participates only during offline training. It never touches the online
decision path.

---

## Getting started

Run the stages in order — each depends on the previous one's output.

### 1. Generate data

Set up the co-simulation and run a data-generation scenario. See
[`Omnet++_Sumo_Veins/README.md`](Omnet++_Sumo_Veins). Output is a CSV with one row
per message:

```
sendTime, SenderID, ReceiverID,
posx, posy, spdx, spdy, aclx, acly, hedx, hedy,
AttackType, label
```

### 2. Train the detector

See [`MIST_Anomaly_Detection/README.md`](MIST_Anomaly_Detection).

```bash
python run_classification.py --model Mist --seq_len 96 --period_len 4 ...
```

Then score the dataset, appending `Prediction`, `Prob_Normal`, and
`Prob_Malicious`.

### 3. Train the mitigation policy

See
[`MAGRL_with_LLM_Reward_Optimisation/README.md`](MAGRL_with_LLM_Reward_Optimisation).

```bash
export HF_TOKEN=hf_xxxxxxxxxxxx
python train_magrl.py --data_path /path/to/data_with_predictions.csv
```

### 4. Run online

Start the socket server, then launch the simulation. See
[`Python_C++/README.md`](Python_C++).

---

## Evaluation

The framework is evaluated on three real-world road topologies extracted from
OpenStreetMap, across three traffic densities, three attacker ratios, and six
message-level attack types.

| Road | Location | Area |
|---|---|---|
| Road 1 | Casablanca, Morocco | 12.5 km × 6.9 km |
| Road 2 | Orlando, USA | 6.9 km × 9.0 km |
| Road 3 | Madrid, Spain | 4.9 km × 10.8 km |

Densities of 50, 100, and 200 vehicles are combined with attacker ratios of 10%,
30%, and 50%, giving 27 evaluation configurations per attack scenario.

**Attacks:** DoS Disruptive, Constant Position Offset, Random Position, Constant
Speed Offset, Random Speed, and Replay Attack.

### Headline results

MIST reaches an F1-score of 92.79% on the held-out test set, ahead of the
evaluated time-series baselines, at roughly half the inference cost of Informer.

Adding MAGRL improves F1 over detection alone in every configuration, by 1.62,
4.54, and 7.59 percentage points on average across the three topologies, and by up
to 18.49 points in the sparsest, most heavily attacked scenario. The gain is
largest exactly where detection alone is weakest.

---

## Requirements

| Component | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.x |
| PyTorch Geometric | 2.4+ (MAGRL only) |
| OMNeT++ | 5.6.2 |
| SUMO | 1.8.0 |
| Veins | 5.2 |
| INET | 4.x |

Per-component dependencies are in each directory's `requirements.txt`. PyTorch
Geometric needs an index URL matched to your torch and CUDA build; see the MAGRL
README.

Reward generation calls a hosted LLM through the HuggingFace inference router and
expects an `HF_TOKEN` environment variable. Training can run without it using
`--no_llm`, which keeps a fixed reward throughout.

---

## Citation

```bibtex
@article{elbouga2026vanetllm,
  title   = {LLM-Guided Multi-Agent Graph Reinforcement Learning for Anomaly
             Detection and Adaptive Mitigation in Vehicular Networks},
  author  = {El Bouga, Latifa and others},
  journal = {Under review},
  year    = {2026}
}
```

This work builds on our earlier framework combining digital-twin anomaly detection
with single-agent graph reinforcement learning:

```bibtex
@article{elbouga2026dt,
  title   = {Securing Vehicular Ad Hoc Networks via Digital Twin-Driven Anomaly
             Detection and Graph Reinforcement Learning},
  author  = {El Bouga, Latifa and Andam, A. and Bentahar, J. and
             Amhoud, E. M. and Hedabou, M.},
  journal = {IEEE Transactions on Vehicular Technology},
  year    = {2026}
}
```

---

## Acknowledgements

MIST builds on [SparseTSF](https://github.com/lss-1138/SparseTSF) (Lin et al.,
ICML 2024). The detection experiment scaffolding follows
[Time-Series-Library](https://github.com/thuml/Time-Series-Library). MAGRL uses
MAPPO (Yu et al., NeurIPS 2022) and GATv2 (Brody et al., ICLR 2022) via
[PyTorch Geometric](https://pyg.org/). The simulation is built on
[Veins](https://veins.car2x.org/), [OMNeT++](https://omnetpp.org/), and
[SUMO](https://www.eclipse.org/sumo/).

---

## License

<!-- Add a license before making the repository public. -->
