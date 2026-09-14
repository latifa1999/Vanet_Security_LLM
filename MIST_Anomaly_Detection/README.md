# MIST — Message-Level Anomaly Detection for VANETs

Training and benchmarking code for **MIST** (Multi-scale Inter-Segment Sparse-Temporal
Network), a lightweight time-series architecture for message-level anomaly
classification in Vehicular Ad Hoc Networks.

This repository covers the **anomaly detection stage** of the framework described in
*"LLM-Guided Multi-Agent Graph Reinforcement Learning for Anomaly Detection and
Adaptive Mitigation in Vehicular Networks."* The downstream MAGRL mitigation module
lives in a separate repository.

---

## Overview

Malicious behaviour in VANETs usually shows up as inconsistency **across consecutive
messages** rather than in any single message. MIST therefore treats detection as a
binary time-series classification problem: a sliding window of `T` consecutive V2V
messages, each described by eight kinematic features, is classified as normal or
malicious.

MIST builds on the cross-period sparse reshaping strategy of
[SparseTSF](https://github.com/lss-1138/SparseTSF) and adds components suited to
classification rather than forecasting:

- a learnable temporal embedding, applied after sparse aggregation
- multi-scale 1D convolution branches (kernel sizes 3, 5, 7)
- two residual convolutional blocks
- four-head self-attention over temporal positions
- pointwise channel aggregation followed by dual (average + max) pooling

The repository also contains the baseline architectures MIST was benchmarked
against, all trained through the same experiment loop for a fair comparison:
Informer, Autoformer, FEDformer, PatchTST, DLinear, Linear, Transformer,
Non-stationary Transformer, FiLM, and SparseTSF.

---

## Repository structure

```
AD_models/
├── data_provider/
│   ├── data_factory.py        # dataset/loader selection by --data flag
│   └── data_loader.py         # windowing, splitting, scaling
├── exp/
│   ├── exp_basic.py           # base experiment class (device setup, model registry)
│   └── exp_classification.py  # training / validation / test loop, metrics, logging
├── layers/                    # shared building blocks used by the baseline models
├── models/
│   ├── Mist.py                # proposed architecture
│   ├── SparseTSF.py           # original SparseTSF (forecasting backbone)
│   ├── Informer.py  Autoformer.py  FEDformer.py  PatchTST.py
│   ├── DLinear.py   Linear.py      Transformer.py
│   ├── Nonstationary_Transformer.py  Film.py  Stat_models.py
├── utils/                     # early stopping, metrics, resource tracking, time features
├── scripts/                   # example run configurations
├── scalers/
│   └── veremi_scaler.pkl      # StandardScaler fitted on the training split
├── checkpoints/               # saved model weights (created at runtime)
├── results/                   # predictions, metrics, confusion matrices
├── logs/ , log_files/         # TensorBoard events and run logs
├── run_classification.py      # main entry point (training + testing)
├── test_script.py             # standalone inference on a trained checkpoint
├── hyperparameter.py          # hyperparameter sweep helper
└── requirements.txt
```

---

## Installation

Tested with Python 3.10 and PyTorch 2.x on Linux (NVIDIA A100).

```bash
git clone <repository-url>
cd AD_models_github

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

If you need a specific CUDA build of PyTorch, install it first from
[pytorch.org](https://pytorch.org/get-started/locally/), then install the rest:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

The code runs on CPU as well; pass `--use_gpu False` (training will be slower).

---

## Data format

The loader expects a single CSV with one row per V2V message:

| Column | Description |
|---|---|
| `sendTime` | message timestamp |
| `SenderID`, `ReceiverID` | vehicle identifiers |
| `posx`, `posy` | position components |
| `spdx`, `spdy` | speed components |
| `aclx`, `acly` | acceleration components |
| `hedx`, `hedy` | heading components |
| `AttackType` | attack label (`normal_behavior`, `DoS_Disruptive`, `Replay_Attack`, …) |
| `label` | binary ground truth (0 = normal, 1 = malicious) |

The eight kinematic columns are the model input. Rows are split 70/10/20 into
train/validation/test, windows of length `seq_len` are built within
each split, and each window takes the label of its last message.


Features are standardised with a `StandardScaler` fitted **on the training split
only**. The fitted scaler is written to `--scaler_path` and reloaded for validation,
test, and inference. Delete it if you change the dataset, or the stale scaler will be
reused silently.

---

## Usage

### Training

```bash
python run_classification.py \
  --is_training 1 \
  --model Mist \
  --model_id my_run \
  --data Generated \
  --root_path /path/to/data/ \
  --data_path dataset.csv \
  --seq_len 96 \
  --period_len 4 \
  --enc_in 8 \
  --num_classes 2 \
  --batch_size 256 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10
```

Training runs, then automatically evaluates on the test split. Checkpoints go to
`checkpoints/`, metrics and predictions to `results/`, TensorBoard events to `logs/`.

### Testing an existing checkpoint

```bash
python run_classification.py \
  --is_training 0 \
  --model Mist \
  --model_id my_run \
  --data Generated \
  --root_path /path/to/data/ \
  --data_path dataset.csv \
  --seq_len 96 --period_len 4
```

### Standalone inference

`test_script.py` loads a trained checkpoint and produces predictions for a CSV
without going through the experiment loop. Use it to generate the per-message
predictions and confidence scores consumed by the MAGRL stage.

```bash
python test_script.py
```

Edit the checkpoint, scaler, and data paths at the top of the script before running.

### Running a baseline instead

Swap `--model` for any name in `models/`:

```bash
python run_classification.py --model Informer --model_id informer_run ...
```

Baselines have their own architecture arguments (`--d_model`, `--n_heads`,
`--e_layers`, `--d_ff`, `--factor`); defaults are in `run_classification.py`.


## Outputs

Each run writes, under a directory named after `--model_id`:

- `predictions.txt` — per-sample predicted class
- `test_metrics.txt` — accuracy, precision, recall, F1 (weighted and per-class)
- `confusion_matrix.txt`
- resource logs and plots — RAM/GPU usage and inference timing
