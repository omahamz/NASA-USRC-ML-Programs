# Usage Guide — Multi-Fidelity Surrogate Models

## Quick Reference

```bash
# Install dependencies
pip install -r requirements.txt

# Train MLP (both phases, ~5-10 minutes)
python -m src.ml_models.train_mlp

# Train GP  (both phases, ~2-5 minutes)
python -m src.ml_models.train_gp

# Predict (in Python)
from src.ml_models import predict_mlp, predict_gp
```

---

## 1. Installation

From the project root:

```bash
pip install -r requirements.txt
```

New dependencies added for the ML models: `torch` (PyTorch) and `joblib`.

Verify:

```python
import torch; print(torch.__version__)
import sklearn; print(sklearn.__version__)
```

---

## 2. Directory Structure After Training

```
models/
├── mlp_pretrained_lf.pt       # MLP Phase-1 weights (LF pre-trained)
├── mlp_pretrained_lf.json     # Phase-1 architecture and training metadata
├── mlp_finetuned_hf.pt        # MLP Phase-2 weights (HF fine-tuned)
├── mlp_finetuned_hf.json      # Phase-2 architecture and training metadata
├── gp/
│   ├── gp_sea_lf.pkl          # LF GP for SEA
│   ├── gp_cfe_lf.pkl          # LF GP for CFE
│   ├── gp_sea_delta.pkl       # Correction GP for SEA
│   ├── gp_cfe_delta.pkl       # Correction GP for CFE
│   └── gp_metadata.json       # Kernel parameters, fit status, timestamps
├── scalers.pkl                 # StandardScaler objects (shared by MLP and GP)
├── mlp_comparison_metrics.json # MLP LF-only vs Transfer R², RMSE, MAE
├── gp_comparison_metrics.json  # GP LF-only vs Transfer R², RMSE, MAE
└── plots/
    ├── LC_Phase1_MLP.png       # MLP Phase-1 learning curves
    ├── LC_Phase2_MLP.png       # MLP Phase-2 learning curves
    ├── Parity_MLP_LF_.png      # MLP LF-only parity plots
    ├── Parity_MLP_TL_.png      # MLP Transfer parity plots
    ├── Parity_GP_LF_.png       # GP LF-only parity plots
    ├── Parity_GP_TL_.png       # GP Transfer parity plots
    └── GP_uncertainty_*.png    # GP posterior uncertainty plots
```

---

## 3. Training the MLP

### Full pipeline (recommended)

```bash
python -m src.ml_models.train_mlp
```

This runs Phase 1 (LF pre-training) followed by Phase 2 (HF fine-tuning) and saves
comparison metrics automatically.

### Phase 1 only

```bash
python -m src.ml_models.train_mlp --phase 1
```

### Phase 2 only (Phase 1 must be completed first)

```bash
python -m src.ml_models.train_mlp --phase 2
```

### Options

```
--seed INT          Random seed (default: 42)
--epochs-p1 INT     Phase-1 epoch cap (default: 600)
--epochs-p2 INT     Phase-2 epoch cap (default: 300)
--models-dir PATH   Directory to save models (default: models/)
```

### What to watch during training

Phase 1 output (every 25 epochs):

```
  [Phase1] ep  25  train=0.84632 (SEA=0.42316 CFE=0.42316)  val=0.89021
  [Phase1] ep  50  train=0.71048 (SEA=0.35524 CFE=0.35524)  val=0.74512
  ...
  [Phase1] Early stopping at epoch 312. Best val loss 0.38410 at epoch 262.
```

**Good signs**: Both SEA and CFE losses decrease together. Val loss tracks train loss
closely (no large gap = no overfitting).

**Warning signs**: If CFE loss stagnates while SEA decreases, the model may be
under-representing CFE. This is rare after standardization but worth monitoring.

Phase 2 output:

```
  [Phase2] ep  25  train=0.25318 (SEA=0.12659 CFE=0.12659)  val=0.28741
  ...
  [Phase2] HF test  →  SEA R²=0.8932  CFE R²=0.7841
```

---

## 4. Training the GP

### Full pipeline (recommended)

```bash
python -m src.ml_models.train_gp
```

### LF fitting only

```bash
python -m src.ml_models.train_gp --phase lf
```

### Correction fitting only (LF must be completed first)

```bash
python -m src.ml_models.train_gp --phase correction
```

### Options

```
--seed INT          Random seed (default: 42)
--restarts INT      Kernel optimizer restarts (default: 5; more = slower but better)
--models-dir PATH   Directory to save models (default: models/)
```

### Expected training time

| Phase                                    | Approximate time |
| ---------------------------------------- | ---------------- |
| LF GP fitting (n=938, restarts=5)        | 1–5 minutes      |
| Correction GP fitting (n=98, restarts=5) | 5–30 seconds     |

### What to watch

The kernel parameters after fitting reveal which inputs matter most:

```
SEA kernel: 0.841**2 * RBF(length_scale=[1.23, 2.54, 0.87, 1.41]) + WhiteKernel(noise=0.012)
```

Shorter length scales = that parameter has stronger influence on SEA.
Very large length scales (> 50) indicate the parameter is nearly irrelevant.

The delta kernel should have smaller amplitude than the LF kernel, confirming the
correction is smaller than the raw response:

```
delta-SEA kernel: 0.423**2 * RBF(length_scale=[...]) + WhiteKernel(noise=0.003)
```

---

## 5. Making Predictions

### Using the MLP

```python
import numpy as np
from src.ml_models import predict_mlp

# Single design point: [R, A, CC, VC]
X = np.array([[3.5, 60.0, 12.0, 6.0]])

# High-fidelity model (recommended)
sea, cfe = predict_mlp(X, phase='hf')
print(f"SEA = {sea[0]:.4f}")     # in original units (N·mm / mm³)
print(f"CFE = {cfe[0]:.4f}")     # in [0, 1]

# Low-fidelity model (baseline comparison)
sea_lf, cfe_lf = predict_mlp(X, phase='lf')
```

### Using the GP (with uncertainty)

```python
from src.ml_models import predict_gp

X = np.array([[3.5, 60.0, 12.0, 6.0],
              [5.0, 45.0,  8.0, 5.0]])

sea_mean, sea_std, cfe_mean, cfe_std = predict_gp(X)

for i in range(len(X)):
    print(f"Point {i}: SEA = {sea_mean[i]:.3f} ± {2*sea_std[i]:.3f} (95% CI)")
    print(f"          CFE = {cfe_mean[i]:.3f} ± {2*cfe_std[i]:.3f} (95% CI)")
```

### Scalarized objective

```python
from src.ml_models.predict import predict_objective

# Uses GP by default, k defaults to mean(SEA_pred)
obj = predict_objective(X, model='gp')
print(f"Obj = {obj}")

# Use a specific k value
obj = predict_objective(X, model='gp', k=2.5)

# Using MLP
obj = predict_objective(X, model='mlp', phase='hf')
```

### Batch predictions

Both functions accept any number of rows:

```python
# 1000 random candidate designs
rng = np.random.default_rng(42)
X_candidates = np.column_stack([
    rng.uniform(2.0, 8.8, 1000),          # R
    rng.uniform(30.0, 90.0, 1000),         # A
    rng.integers(4, 23, 1000).astype(float), # CC (will not be rounded by predict)
    rng.integers(4, 11, 1000).astype(float), # VC
])

sea, cfe = predict_mlp(X_candidates, phase='hf')
sea_mu, sea_std, cfe_mu, cfe_std = predict_gp(X_candidates)
```

### CC and VC are not auto-rounded

By design, predict_mlp and predict_gp do **not** round CC and VC to the nearest integer.
This lets you observe the model's interpolation between integer design points:

```python
# How does the model interpolate between CC=12 and CC=13?
X_sweep = np.array([[3.5, 60.0, cc, 6.0] for cc in np.linspace(12.0, 13.0, 11)])
sea, cfe = predict_mlp(X_sweep)
```

Only physically valid integer values of CC and VC correspond to real simulation runs.
A warning is raised if any input is outside the training domain.

---

## 6. Evaluation and Comparison

### Run the full comparison

After training both models, generate the comparison bar chart:

```python
import json
from src.ml_models.evaluate import comparison_bar_chart

# Load saved metrics
with open("models/mlp_comparison_metrics.json") as f:
    mlp_m = json.load(f)
with open("models/gp_comparison_metrics.json") as f:
    gp_m = json.load(f)

# Combine all variants
all_metrics = {**mlp_m, **gp_m}
comparison_bar_chart(all_metrics, title="All Models — R² on HF Test Set")
```

### Evaluate on custom data

```python
import numpy as np
from src.ml_models.evaluate import compute_metrics, parity_plot

# Your own true values and predictions
Y_true = np.array([[sea1, cfe1], [sea2, cfe2], ...])
Y_pred = np.array([[sea1_pred, cfe1_pred], ...])

metrics = compute_metrics(Y_true, Y_pred)
print(metrics)
# {'SEA': {'R2': 0.92, 'RMSE': 0.31, 'MAE': 0.24}, 'CFE': {...}}

parity_plot(Y_true, Y_pred, title="My Custom Evaluation")
```

---

## 7. Fine-Tuning with New High-Fidelity Data

When new solid simulation results become available, append them to the HF CSV
(`FDData_SobS_OD40L50G3_Solid_PD.csv`) and re-run Phase 2 only for each model.

### MLP

```bash
python -m src.ml_models.train_mlp --phase 2
```

This automatically loads the saved Phase-1 checkpoint (`mlp_pretrained_lf.pt`) and
fine-tunes on the updated HF dataset. The LF pre-training is unchanged.

### GP

```bash
python -m src.ml_models.train_gp --phase correction
```

This loads the LF GPs from `models/gp/` and refits only the correction GPs on the
expanded HF dataset.

### 7.2 Active Learning Workflow: Propose → Simulate → Fine-Tune

Active learning uses the GP's posterior uncertainty to select the most informative
next simulation points rather than adding them randomly. This is the recommended
approach when HF simulation budget is limited.

#### The Cycle

```
1. Run active_sampler.py  → proposed_<acq>_k<k>_<ts>.csv   (R, A, CC, VC)
2. Run solid FEA/FEM simulations at the proposed parameter values
3. Append simulation results (SEA, CFE columns) to the HF CSV
4. Retrain GP correction:  python -m src.ml_models.train_gp --phase correction
5. Retrain MLP Phase 2:    python -m src.ml_models.train_mlp --phase 2
6. Evaluate updated R² → decide whether to continue or stop
7. Repeat from step 1
```

#### CLI Examples

```bash
# Default: CFE-only uncertainty, 64 proposals, dry-run (no files written)
python src/active_sampler.py --k 64 --acq uncertainty_cfe --dry-run

# CFE-only uncertainty with save
python src/active_sampler.py --k 64 --acq uncertainty_cfe --save

# Weighted uncertainty (both outputs, CFE-biased)
python src/active_sampler.py --k 64 --acq uncertainty --w-sea 0.3 --w-cfe 0.7 --save

# EI on SEA only
python src/active_sampler.py --k 32 --acq ei_sea --save

# EI on CFE only
python src/active_sampler.py --k 32 --acq ei_cfe --save

# Weighted composite EI (recommended once CFE R² > 0.5)
python src/active_sampler.py --k 32 --acq ei_weighted --w-sea 0.2 --w-cfe 0.8 --save

# UCB with custom exploration parameter
python src/active_sampler.py --k 64 --acq ucb --beta 3.0 --save

# List all acquisition functions with descriptions
python src/active_sampler.py --list-acq
```

#### Output CSV Format

The saved CSV has exactly four columns: `R, A, CC, VC`. After running your
solid simulations, add the resulting `SEA` and `CFE` columns and append the
rows directly to `FDData_SobS_OD40L50G3_Solid_PD.csv`.

```
R,A,CC,VC
3.45,62.1,10,6
5.12,48.3,8,5
...
```

No index column is included; the file is ready to concatenate without modification.

#### k Selection Guidance

| k     | Notes                                                                                             |
| ----- | ------------------------------------------------------------------------------------------------- |
| 16–32 | Small batch; good for tight simulation budgets. Diversity filter is effective.                    |
| 64    | **Recommended default.** Balances diversity and budget (exceeds 10% HF threshold).                |
| 128   | Useful if running a large simulation batch in parallel. Diminishing diversity returns above this. |
| >128  | Not recommended without a larger candidate pool (`--pool-size 8192` or higher).                   |

k is automatically rounded up to the next power of 2 if a non-power-of-2 value is
given (e.g., `--k 50` → 64 proposals, with a printed warning).

#### Fine-Tuning Commands After Appending New Data

```bash
# Refit correction GPs only (fast; LF GPs are unchanged)
python -m src.ml_models.train_gp --phase correction

# Retrain MLP Phase 2 (only 562/2962 params update; Phase 1 weights preserved)
python -m src.ml_models.train_mlp --phase 2
```

Both commands load the existing scalers (fit on LF training data) — no scaler
refitting is needed after adding new HF points.

#### Reference

Forrester, A.I.J., Sóbester, A., & Keane, A.J. (2007). Multi-fidelity optimization
via surrogate modelling. _Proc. Royal Society A_, 463, 3251–3269. §4–5 discuss
typical campaign sizes and convergence criteria for engineering surrogate design.

---

## 8. Programmatic Access Without Full Retraining

```python
# Load MLP directly
from src.ml_models.mlp_model import SurrogateNet
model = SurrogateNet.load("models/mlp_finetuned_hf.pt")

# Load GP directly
from src.ml_models.gp_model import MultiFidelityGP
gp = MultiFidelityGP.load("models/gp")

# Load scalers
from src.ml_models.data_loader import load_scalers
x_sc, y_sc = load_scalers()

# Manual inference (MLP)
import torch, numpy as np
X_std = x_sc.transform(np.array([[3.5, 60.0, 12.0, 6.0]]))
with torch.no_grad():
    Y_std = model(torch.FloatTensor(X_std)).numpy()
Y = y_sc.inverse_transform(Y_std)
sea, cfe = Y[0, 0], Y[0, 1]

# Manual inference (GP) with uncertainty
X_std = x_sc.transform(X)
sea_mu_s, sea_std_s, cfe_mu_s, cfe_std_s = gp.predict(X_std, return_std=True)
# Convert std to original units
sea_std = sea_std_s * y_sc.scale_[0]
cfe_std = cfe_std_s * y_sc.scale_[1]
```

---

## 9. Inspecting Saved Metadata

Each model artifact includes a JSON metadata file:

```python
import json

# MLP Phase-2 metadata
with open("models/mlp_finetuned_hf.json") as f:
    meta = json.load(f)
print(meta)
# {
#   "layer_sizes": [4, 64, 32, 16, 2],
#   "total_params": 2962,
#   "phase": 2,
#   "frozen_layers": [0, 1],
#   "trainable_params": 562,
#   "epochs_run": 187,
#   "best_val_loss": 0.23841,
#   "hf_test_R2_SEA": 0.8932,
#   "hf_test_R2_CFE": 0.7841,
#   "hyperparams": {"lr": 0.0001, "wd": 0.001, "n_freeze": 2},
#   "saved_at": "2026-06-01T14:23:11"
# }

# GP kernel parameters
with open("models/gp/gp_metadata.json") as f:
    gp_meta = json.load(f)
print(gp_meta["lf_kernel_params"])
# {"SEA": "0.841**2 * RBF(...) + WhiteKernel(...)", "CFE": "..."}
```

---

## 10. Troubleshooting

| Problem                                             | Likely cause                          | Fix                                                                                      |
| --------------------------------------------------- | ------------------------------------- | ---------------------------------------------------------------------------------------- |
| `FileNotFoundError: Scalers not found`              | Training hasn't been run yet          | Run `train_mlp` or `train_gp` first                                                      |
| `FileNotFoundError: Metadata JSON not found`        | Model was saved with an older version | Re-run the corresponding training phase                                                  |
| `UserWarning: Column 'R' has values outside domain` | Input out of training range           | Check input values; extrapolation may be unreliable                                      |
| GP fitting takes very long                          | Large n_restarts or n points          | Reduce `--restarts` flag (minimum 3)                                                     |
| Phase-2 val loss higher than train loss immediately | HF dataset too small for current LR   | Reduce `--lr-p2` (not exposed yet: edit P2_LR in train_mlp.py)                           |
| MLP CFE R² < 0.5 after Phase 1                      | CFE loss not learning                 | Monitor per-output losses in training output; may need longer training or different seed |

---

## 11. Surrogate Optimization

### 11.1 Overview and Workflow

The optimizer asks the inverse question: _which design inputs produce the best outputs?_
It uses the trained GP or MLP surrogate as a cheap stand-in for FEA, searching over
(R, A, CC, VC) to maximize SEA while bringing CFE toward 1.

The recommended 3-step cycle:

1. **Optimize** — run `optimizer.py` to get a ranked list of promising designs.
2. **Simulate** — run FEA on the top proposals (the designs that look best according to the surrogate).
3. **Append & retrain** — add the new HF results to the HF CSV, then re-run `train_gp` / `train_mlp` (or fine-tune with `--fine-tune`).

Repeat. Each cycle improves the surrogate in the regions that matter most.

### 11.2 GP Optimization — All Four Modes

```bash
# Pure exploitation: rank by GP posterior mean
python src/optimizer.py --model gp --mode exploitation --n 10 --dry-run

# Expected Improvement: balances exploitation with high-uncertainty regions
python src/optimizer.py --model gp --mode ei --n 10 --dry-run

# Upper Confidence Bound: more aggressive exploration (beta=3 increases it)
python src/optimizer.py --model gp --mode ucb --beta 3.0 --n 10 --dry-run

# Pareto front: full SEA-CFE trade-off curve (n is ignored; returns all non-dominated)
python src/optimizer.py --model gp --mode pareto --n-weights 50 --dry-run

# Save proposals and plots (remove --dry-run)
python src/optimizer.py --model gp --mode exploitation --n 10 --alpha 0.5 --save
```

### 11.3 MLP Optimization — All Three Methods

```bash
# Differential Evolution: global search, best for multimodal landscapes (slow)
python src/optimizer.py --model mlp --method de --n 10 --pop-size 20 --max-iter 1000 --dry-run

# L-BFGS-B: gradient-based multi-start, fast but may miss global optimum
python src/optimizer.py --model mlp --method lbfgsb --n-restarts 50 --n 10 --dry-run

# Sobol pool: fastest — no iterative optimization, analogous to GP exploitation
python src/optimizer.py --model mlp --method sobol --n 10 --dry-run

# Save (remove --dry-run, add --save)
python src/optimizer.py --model mlp --method de --n 10 --save
```

### 11.4 `--compare`: Cross-Evaluating GP and MLP

```bash
# GP proposals, cross-evaluated by the MLP
python src/optimizer.py --model gp --mode exploitation --n 10 --compare --save

# MLP proposals, cross-evaluated by the GP
python src/optimizer.py --model mlp --method sobol --n 10 --compare --save
```

The `--compare` flag evaluates each proposal with the _other_ model and adds columns
`[mlp_SEA, mlp_CFE, delta_SEA, delta_CFE, agree]` (GP primary) or
`[gp_SEA, gp_SEA_std, gp_CFE, delta_SEA, delta_CFE, agree]` (MLP primary).

`agree = True` when `|delta_SEA| < 0.5 N·mm/mm³` and `|delta_CFE| < 0.05`.

**Why disagreement is informative:** points where GP and MLP diverge are regions where
the two models have learned different things about the response surface — high-uncertainty
regions in a model-ensemble sense. Prioritizing these for simulation combines the
optimization signal (these designs look good to at least one model) with an active-learning
signal (these designs would most improve the surrogate). When `--save` is also passed,
a separate `comparison_<model>_<mode>_<ts>.csv` file is written alongside the proposals.

### 11.5 Interpreting the Pareto Front Output

```bash
python src/optimizer.py --model gp --mode pareto --n-weights 50 --save
```

The Pareto front lists all designs where neither SEA nor CFE can be improved without
sacrificing the other. The output CSV and Plot 4 (`pareto_front_gp_<ts>.png`) show the
SEA-CFE trade-off curve sorted by SEA, with each point labeled `(CC, VC)`.

Reading the curve:

- **High-SEA end**: designs with low CC, VC; high energy absorption but lower progressiveness.
- **High-CFE end**: designs with high CC, VC; more progressive collapse but lower SEA.
- **Knee point**: the point with the largest curvature — often the best engineering compromise.

Choose a point on the front based on application requirements:

- Crash energy absorption dominant → choose from the high-SEA end.
- Progressive collapse stability dominant → choose from the high-CFE end.
- Balanced → choose the knee point (approximately `--alpha 0.5` in scalarized modes).

### 11.6 Choosing Mode, Method, and Alpha

| Situation                                              | Recommended choice                           |
| ------------------------------------------------------ | -------------------------------------------- |
| First optimization run                                 | `--model gp --mode exploitation --alpha 0.5` |
| GP uncertainty is high (early training, few HF points) | `--mode ei` or `--mode ucb`                  |
| Need the full design trade-off                         | `--mode pareto`                              |
| MLP only (no GP trained)                               | `--model mlp --method sobol` (fastest)       |
| MLP, want global optimum guarantee                     | `--model mlp --method de`                    |
| MLP, want gradient-guided speed                        | `--model mlp --method lbfgsb`                |
| Pure SEA maximization                                  | `--alpha 1.0`                                |
| Pure CFE maximization (toward 1)                       | `--alpha 0.0`                                |
| Match existing project convention                      | `--formula penalty`                          |

Alpha controls the trade-off weight in the normalized weighted sum: `alpha=0.5` balances
SEA and CFE equally; `alpha=0.8` strongly favors SEA; `alpha=0.2` strongly favors CFE.
The normalization by `SEA_ref = max(SEA_pred)` ensures the weights are dimensionally
consistent regardless of absolute SEA magnitude (Marler & Arora, 2004).

### 11.7 Reference

Koziel, S., & Leifsson, L. (2016). _Surrogate-Based Modeling and Optimization:
Applications in Engineering_. Springer. — Standard reference for surrogate-based
optimization in engineering design, including the optimize → simulate → retrain workflow.
