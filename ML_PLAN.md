# ML Plan: Multi-Fidelity Surrogate Models for Crashworthiness Optimization

## Overview

Two surrogate models — a Multi-Layer Perceptron (MLP) and a Multi-Output Gaussian Process (GP)
— are trained to predict **SEA** (Specific Energy Absorption) and **CFE** (Crushing Force
Efficiency) from four structural parameters (R, A, CC, VC). Both models are first pre-trained on
~938 low-fidelity shell simulation data points, then adapted to ~123 high-fidelity solid
simulation data points via transfer learning. The goal is to study and compare the two
approaches in the context of material science simulation surrogates.

---

## Data Description

| File | Type | Points | Columns |
|------|------|--------|---------|
| `FDData_SobS_OD40L50G3_Shell_Fixed_PD.csv` | Low-fidelity (LF) | 938 | R, A, CC, VC, T, AUC, CFE, Volume, SEA, Obj |
| `FDData_SobS_OD40L50G3_Solid_PD.csv` | High-fidelity (HF) | 123 | same |

**Geometry constants**: OD=40 mm, ID=32 mm, L=50 mm.

**T column**: Verified to be 0 for all 1,061 rows — not informative, excluded from model inputs.

**Overlap**: 118/123 HF points (95.9%) have an exact matching LF point. The 5 non-overlapping
points are handled naturally by evaluating the LF model at those locations.

**Fidelity delta statistics** (computed at the 118 overlapping HF locations):

| Output | Mean delta | Std dev | Min | Max |
|--------|-----------|---------|-----|-----|
| SEA    | +0.6694   | 0.5437  | -0.23 | +1.68 |
| CFE    | +0.0113   | 0.0936  | -0.25 | +0.19 |

Key observation: SEA shows a **systematic positive bias** of ~+0.67 between shell and solid
(solid absorbs ~30% more energy on average). The delta has significant variability (std=0.54),
confirming it is not a simple constant shift — it depends on the input parameters and thus is
worth modeling with a learned correction. CFE differences are smaller but non-negligible.

---

## Inputs and Outputs

```
Inputs  X = [R, A, CC, VC]
Outputs Y = [SEA, CFE]
```

| Variable | Type  | Domain       | Notes                           |
|----------|-------|--------------|---------------------------------|
| R        | float | [2.0, 8.8]   | Corner radius (mm)              |
| A        | float | [30, 90]     | Angle (degrees)                 |
| CC       | int   | {4, …, 22}   | Cell count                      |
| VC       | int   | {4, …, 10}   | Volume coefficient              |
| SEA      | float | [0, ∞)       | Target; higher is better        |
| CFE      | float | [0, 1]       | Target; closer to 1 is better   |

Geometric constraints reduce ~65% of the input space:
- C1: `CC < (π·OD) / (√3·R)`
- C2: `VC < [L - 2(G+R)] / |2R - √3·π·OD/(6·CC)| + 1`
- C3: `VC < [L - 2(R+G)] / R + 1`

All 1,061 data points already satisfy these constraints (they were sampled with constraint
enforcement in `sample.py`).

---

## Data Preprocessing

### Normalization

All input features and both outputs are independently standardized using `StandardScaler`:

```
X_std = (X - mean_X_LF_train) / std_X_LF_train
SEA_std = (SEA - mean_SEA_LF_train) / std_SEA_LF_train
CFE_std = (CFE - mean_CFE_LF_train) / std_CFE_LF_train
```

**Scalers are fit exclusively on the LF training split** to prevent data leakage. The same
scalers are applied to all subsequent data including HF.

**Why standardize CC and VC alongside the floats?** Both GPs and MLPs use Euclidean distance
or gradient signals across all features. Without scaling, CC (range 4–22) would dominate over
VC (range 4–10) and R (range 2–8.8). Standardization puts all inputs on equal footing, then
the models (via kernel length-scales or layer weights) learn the true relative importance.

### Train / Validation / Test Splits

```
LF data (938 points):
  ├── LF train    (750 points, 80%)  — pre-training
  └── LF val      (188 points, 20%)  — pre-training early stopping

HF data (123 points):
  ├── HF fine-tune (98 points, 80%)  — transfer learning
  └── HF test      (25 points, 20%)  — final held-out evaluation ONLY
```

The HF test set is **never used during any training phase**. It is the gold-standard for
comparing all model variants.

---

## Model 1: Multi-Layer Perceptron (MLP)

### Framework

**PyTorch** — chosen because:
- Layer-level freezing via `requires_grad=False` (scikit-learn MLPRegressor cannot freeze
  individual layers)
- `state_dict` serialization enables reloading the exact pre-trained weights for future
  fine-tuning sessions
- Adam optimizer with per-parameter LR groups for differential LR between frozen/unfrozen
  sections
- Industry standard for deep learning in material science surrogate modeling

Requires adding `torch` to `requirements.txt`.

### Architecture: [4 → 64 → 32 → 16 → 2]

```
Input Layer    :  4 neurons   (R, A, CC, VC — standardized)
Hidden Layer 1 : 64 neurons   + tanh
Hidden Layer 2 : 32 neurons   + tanh
Hidden Layer 3 : 16 neurons   + tanh
Output Layer   :  2 neurons   (SEA_std, CFE_std — linear)
```

**Total parameters**: (4×64+64) + (64×32+32) + (32×16+16) + (16×2+2) = 320+2080+528+34 = **2,962**

**Ratio of LF training samples to parameters**: 750 / 2,962 ≈ 0.25

This ratio is below the classical "10× rule," but with L2 regularization and early stopping
it is adequate. The architecture is kept deliberately moderate — 3 hidden layers rather than
4 or 5 — because:

1. **Data scale**: 938 LF samples is small for a deep network. Depth beyond 3 hidden layers
   risks overfitting unless dropout or batch normalization is added. For smooth physical
   simulation responses, 3 layers is standard in the material science surrogate literature.

2. **Non-linearity depth**: The output (SEA, CFE) is a function of geometric parameters through
   volume computation and a crushing simulation. This involves interactions at two scales:
   local (pairwise: R vs CC, VC vs CC) and global (all four together). Three hidden layers
   handle both scales without overparameterization.

3. **Transfer learning alignment**: Freezing the first 2 hidden layers leaves 1 hidden layer
   + output (528+34 = 562 trainable parameters) for HF fine-tuning. With 98 HF training samples,
   this gives a samples-to-params ratio of ~0.17, which is manageable with strong L2 regularization
   (weight_decay=1e-3 during fine-tuning).

**Why tanh activations?** Physical simulation responses are smooth, bounded functions of
geometry — tanh is smooth (infinite differentiability), symmetric around zero, and produces
bounded pre-activations that match well with standardized inputs. ReLU is prone to dead neurons
in small-data regimes; ELU/SELU add complexity without clear benefit here.

**Why not add a hidden layer during HF fine-tuning (as originally considered)?**
Adding a layer changes the model's parameter count and the shape of intermediate tensors,
making it impossible to initialize from the pre-trained weights directly. The gain (slightly
more capacity for HF-specific features) is outweighed by the loss of warm-start initialization.
The fine-tuning approach (freeze + retrain) achieves the same effect cleanly: the frozen layers
provide the learned geometric representation, and the retrained layers adapt that representation
to HF fidelity.

### Phase 1 — Low-Fidelity Pre-Training

```
Data:        750 LF train points
Loss:        MSE on [SEA_std, CFE_std]
Optimizer:   Adam  (lr=1e-3, weight_decay=1e-4)
Batch size:  64 (mini-batch; implicit noise regularization + more updates per epoch)
Epochs:      600 max, with early stopping (patience=50) on LF val loss
Saved as:    models/mlp_pretrained_lf.pt
```

Training monitors both SEA and CFE loss independently to detect if one output is being
systematically ignored. If CFE loss stagnates (CFE range is narrower → smaller MSE magnitude
than SEA), a loss weighting strategy (weight SEA loss vs CFE loss by inverse output variance)
is applied.

### Phase 2 — High-Fidelity Fine-Tuning

```
Data:        98 HF fine-tune points (LF scalers applied)
Frozen:      Hidden Layer 1, Hidden Layer 2  (set requires_grad=False)
Trainable:   Hidden Layer 3, Output Layer   (562 parameters)
Loss:        MSE on [SEA_std, CFE_std] using HF target values
Optimizer:   Adam  (lr=1e-4, weight_decay=1e-3 — stronger regularization for small HF set)
Batch size:  Full-batch or 32 (whichever is smaller for stability)
Epochs:      300 max, with early stopping (patience=30)
Saved as:    models/mlp_finetuned_hf.pt
```

The lower learning rate (1e-4 vs 1e-3) prevents catastrophic forgetting of the pre-trained
representation in the unfrozen layers while still allowing meaningful adaptation. The stronger
weight decay (1e-3 vs 1e-4) controls overfitting on the small HF set.

**Rationale for freezing layers 1 and 2:**

- Layer 1 (4→64) captures broad feature interactions from the raw inputs — this is domain
  knowledge about the geometry that is fidelity-independent. Whether a simulation uses shell or
  solid elements, the geometric relationships (R vs CC via constraint C1, VC vs R via C2) are
  the same. This layer should be preserved.

- Layer 2 (64→32) compresses and cross-correlates features — again, fidelity-agnostic
  geometric knowledge. Retraining it on 98 points risks degrading the LF pre-trained signal.

- Layer 3 (32→16) is the most task-specific representation before the output. Retraining this
  layer allows the model to refocus its compressed representation toward HF simulation physics.

- The output layer (16→2) maps directly to (SEA, CFE) — the HF values are systematically
  higher for SEA and slightly different for CFE, so this must be retrained.

This mirrors the professor's lecture exactly (freeze early layers, fine-tune output layers)
and is the standard approach for low-data target domain adaptation.

### Saving and Reloading

```
models/
├── mlp_pretrained_lf.pt      # Phase 1 state_dict
├── mlp_finetuned_hf.pt       # Phase 2 state_dict
├── mlp_metadata.json         # Architecture, training config, metrics, date
└── scalers.pkl               # StandardScaler objects (fit on LF train)
```

The metadata JSON records: layer sizes, activation, optimizer hyperparameters, number of LF/HF
training samples, final train/val loss, test R², and timestamp. This allows reproducibility
and comparison across future fine-tuning sessions.

---

## Model 2: Multi-Output Gaussian Process (GP)

### Framework

**scikit-learn** `GaussianProcessRegressor` — already in requirements, no new dependencies.
Two independent GPs are used (one for SEA, one for CFE). A true multi-output GP (ICM/LMC)
would be more principled but requires GPy or GPflow and becomes computationally expensive for
n > 500. With scikit-learn, full GP training on 938 points is O(n³) ≈ O(938³/3) ≈ 275M
operations, which runs in seconds to a minute — acceptable for a one-time fit.

### Why two independent GPs over a scalarized single GP?

The user mentioned using `Obj = SEA - k*(1-CFE)` as the GP target. This is valid but collapses
two objectives into one: you lose the ability to inspect SEA and CFE separately, and the
objective function uses `k = mean(SEA_HF)` which is defined only after seeing the HF data —
creating a subtle leakage of HF statistics into the model structure.

Two independent GPs provide:
1. Separate uncertainty estimates for SEA and CFE
2. Flexibility to define different k values post-hoc without retraining
3. Cleaner comparison with the MLP (which also outputs both SEA and CFE separately)

The scalarized Obj can always be computed from the two GP predictions: `Obj = GP_SEA(x) - k*(1 - GP_CFE(x))`.

### Kernel

```
kernel = C(1.0) * RBF(length_scale=[1.0]*4, length_scale_bounds=(1e-2, 1e2)) 
         + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1.0))
```

**ARD RBF (Automatic Relevance Determination):** The kernel learns a separate length scale
per input dimension. This is important because:
- R (float, range ~6.8) and A (float, range ~60) have very different physical scales
- CC (integer, range 4–22) and VC (integer, range 4–10) are discrete but treated as
  continuous — ARD naturally learns that integer-valued inputs have shorter effective
  length scales

**WhiteKernel:** Captures stochastic simulation noise (FEA runs can have minor numerical
variability from mesh generation). Setting a noise floor prevents overconfident predictions.

### Multi-Fidelity Transfer: Kennedy-O'Hagan Additive Correction

For each output (SEA and CFE independently), the multi-fidelity approach follows the
Kennedy & O'Hagan (2000) paradigm:

```
Step 1: Fit GP_LF on all 938 LF (shell) training points.

Step 2: For each of the 123 HF fine-tune points x_i:
        delta_i = Y_HF(x_i) - GP_LF.predict(x_i)
        [For the 5 non-overlapping HF points, GP_LF.predict() is used
         since there is no exact LF counterpart — this is valid because
         GP_LF has already generalized across the input space.]

Step 3: Fit GP_delta on the 123 (x_i, delta_i) pairs.
        GP_delta uses the same ARD-RBF + WhiteKernel structure.

Step 4: Final prediction at any x:
        mu_HF(x)  = mu_LF(x)  + mu_delta(x)
        var_HF(x) = var_LF(x) + var_delta(x)
        [Variances add under independence assumption of LF and delta GPs]
```

**Why this approach?**

1. **Principled**: The delta represents the systematic physical bias between shell and solid
   FEA models. The fidelity delta statistics show mean SEA delta of +0.67 with std 0.54 —
   significant structure worth modeling rather than ignoring.

2. **Data-efficient**: GP_LF leverages all 938 LF points. GP_delta only needs to learn the
   (smaller, smoother) correction from 123 HF points. A GP trained purely on 123 HF points
   would miss the broad structure learned from 938 samples.

3. **Naturally handles non-overlapping points**: The 5 HF points without an exact LF match
   are handled seamlessly — we evaluate GP_LF (not look up a table) to get the LF prediction.

4. **Uncertainty propagation**: Unlike fine-tuning an MLP (which gives a point prediction),
   the GP framework naturally propagates uncertainty from both the LF fit and the correction
   model, giving honest confidence intervals on HF predictions.

**Why K-O'Hagan over direct HF-only GP?**

Training a GP only on the 123 HF points ignores 87.8% of the available data (938 LF points).
The LF and HF responses are highly correlated (same physics, different mesh fidelity) — the
LF data provides a strong prior on the response surface shape, which the K-O'Hagan approach
explicitly exploits. The delta GP is much easier to learn because deltas (std ≈ 0.54 for SEA)
are smaller than raw SEA values (std ≈ 1.27 for LF, ≈ 1.77 for HF).

### Saving and Reloading

```
models/
├── gp_sea_lf.pkl           # GP_LF for SEA
├── gp_cfe_lf.pkl           # GP_LF for CFE
├── gp_sea_delta.pkl        # GP_delta for SEA correction
├── gp_cfe_delta.pkl        # GP_delta for CFE correction
├── gp_metadata.json        # Kernel hyperparameters, training config, metrics, date
└── scalers.pkl             # shared with MLP (same LF-fit scalers)
```

`joblib.dump` / `joblib.load` are used for all `.pkl` files (sklearn's recommended serialization).
Kernel hyperparameters (optimized length scales, amplitude, noise) are also recorded in the
metadata JSON for interpretability.

---

## Transfer Learning Comparison Protocol

### Ablation Study (4 model variants per architecture)

| Variant | Training data | Transfer? | Notation |
|---------|--------------|-----------|----------|
| LF-only baseline | 938 LF shell | None | MLP-LF / GP-LF |
| HF-only baseline | 123 HF solid (from scratch) | None | MLP-HF0 / GP-HF0 |
| Transfer (LF→HF) | 938 LF + 98 HF fine-tune | Yes | MLP-TL / GP-TL |

This gives a complete picture: how much does LF pre-training help vs. HF-from-scratch? How
much does HF transfer improve over LF-only?

### Evaluation Metrics

All models evaluated on the same held-out **HF test set** (25 points):

| Metric | Formula |
|--------|---------|
| R² | 1 - SS_res/SS_tot |
| RMSE | √( mean((y_pred - y_true)²) ) |
| MAE | mean(\|y_pred - y_true\|) |

Reported separately for SEA and for CFE. R² is the primary metric.

### Visualizations

1. **Parity plots** (predicted vs actual) — one per model variant per output
2. **R² bar chart** — MLP vs GP, LF-only vs Transfer, for SEA and CFE
3. **Learning curves** — MLP train/val loss over epochs for Phase 1 and Phase 2
4. **GP delta visualization** — scatter of delta vs each input to show spatial structure
5. **Residual plots** — error distribution before and after transfer learning

---

## Implementation File Structure

```
src/
├── ml_models/
│   ├── __init__.py
│   ├── data_loader.py      # Load _PD CSVs, create LF/HF splits, fit scalers
│   ├── mlp_model.py        # PyTorch MLP class (architecture, forward pass)
│   ├── gp_model.py         # sklearn GP wrapper + K-O'Hagan correction logic
│   ├── train_mlp.py        # Phase 1 + Phase 2 training pipeline, early stopping
│   ├── train_gp.py         # GP fitting + delta computation + correction fitting
│   ├── evaluate.py         # Metrics (R², RMSE, MAE), parity plots, comparison charts
│   └── predict.py          # Unified prediction interface for both models
├── models/                 # Saved model files (.pt, .pkl, .json)
│   └── .gitkeep
```

### `data_loader.py` responsibilities

- Load `Shell_Fixed_PD.csv` and `Solid_PD.csv` from `data_folder/1_param/`
- Select columns: `['R', 'A', 'CC', 'VC']` as X, `['SEA', 'CFE']` as Y
- Drop any rows with NaN in X or Y
- Create train/val/test splits (stratified by CC and VC for representativeness)
- Fit `StandardScaler` on LF train split; export for reuse

### `mlp_model.py` responsibilities

- Define `SurrogateNet(nn.Module)` with configurable layer sizes
- `freeze_until(layer_idx)` method: sets `requires_grad=False` on layers 0..idx-1
- `unfreeze_all()` method: resets all layers to trainable
- `save(path, metadata)` and `load(path)` classmethods

### `gp_model.py` responsibilities

- `MultiFidelityGP` class wrapping two independent scikit-learn GPs
- `fit_lf(X_lf, Y_lf)`: fits GP_LF_SEA and GP_LF_CFE
- `compute_deltas(X_hf, Y_hf)`: evaluates GP_LF at HF locations, computes delta
- `fit_hf_correction(X_hf, deltas_hf)`: fits GP_delta_SEA and GP_delta_CFE
- `predict(X, return_std=False)`: returns (SEA_pred, CFE_pred) + optional std
- `objective(X, k=None)`: returns scalarized Obj = SEA_pred - k*(1-CFE_pred)
- `save(dir)` and `load(dir)` classmethods using joblib

### `predict.py` responsibilities

- `load_mlp(phase)`: loads MLP (LF or HF phase) from `models/`
- `load_gp()`: loads multi-fidelity GP from `models/`
- `predict_mlp(X, phase='hf')`: returns (SEA, CFE) in original units
- `predict_gp(X)`: returns (SEA_mean, SEA_std, CFE_mean, CFE_std) in original units
- Input validation: check X shape (n, 4), clip R/A to domain, round CC/VC to nearest int

---

## Questions / Open Items

1. **Loss weighting for SEA vs CFE**: SEA variance (~1.27²) is much larger than CFE variance
   (~0.07²). After standardization this equalizes, but if the model consistently under-learns
   CFE, a weighted loss `loss = w_SEA * MSE_SEA + w_CFE * MSE_CFE` can be applied (weights
   ≈ inverse of output variance before standardization). This will be checked during Phase 1
   training and added if CFE R² < 0.5 at convergence.

2. **GP scalability**: With n=938, sklearn GP is O(n³) ≈ feasible in under a minute. If HF
   data grows (additional simulation campaigns), sparse approximations (Nyström, SOR) may be
   needed. This is noted but not pre-implemented.

3. **Constraint enforcement at prediction time**: The `predict.py` interface will warn (not
   error) if input X violates constraints C1/C2/C3, since extrapolation into the infeasible
   region may still be of research interest.

4. **Random seed**: All splits, weight initializations, and GP optimizations use `seed=42` by
   default for reproducibility.

---

## Dependency Additions

Add to `requirements.txt`:
```
torch          # MLP framework (PyTorch)
joblib         # GP serialization (already available via scikit-learn, but explicit)
```

---

## References

- Kennedy, M.C. & O'Hagan, A. (2000). Predicting the output from a complex computer code when
  fast approximations are available. *Biometrika*, 87(1), 1-13. [Fidelity correction framework]
- Professor's transfer learning lecture (transfer_learning_lecture/): Layer-freezing fine-tuning
  approach for MLP target domain adaptation
- Rasmussen & Williams (2006). *Gaussian Processes for Machine Learning*. MIT Press.
  [GP kernel design and ARD]
