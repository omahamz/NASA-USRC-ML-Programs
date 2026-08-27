# MLP Surrogate Model — Design Document

## 1. Problem Framing

The multi-layer perceptron (MLP) is trained to approximate the mapping:

```
f : (R, A, CC, VC) → (SEA, CFE)
```

from a finite set of finite-element analysis (FEA) simulation results, avoiding the need to
run expensive simulations for every candidate design during optimization.

**Why an MLP?** Universal approximation theory guarantees that a feed-forward network with
a single hidden layer and enough neurons can approximate any continuous function on a compact
domain to arbitrary precision (Hornik et al., 1989). In practice, moderate-depth networks
generalize far better than wide-shallow networks on physical simulation data (Bengio, 2012).

---

## 2. Architecture

```
Input  (4)  →  Hidden-1 (64) + tanh
            →  Hidden-2 (32) + tanh
            →  Hidden-3 (16) + tanh
            →  Output  (2,  linear)
```

| Layer     | Shape   | Parameters | Role                                                     |
| --------- | ------- | ---------- | -------------------------------------------------------- |
| Input     | 4       | —          | R, A, CC, VC (standardized)                              |
| Hidden-1  | 4 → 64  | 320        | Broad feature extraction; encodes geometric interactions |
| Hidden-2  | 64 → 32 | 2,080      | Cross-parameter relationship compression                 |
| Hidden-3  | 32 → 16 | 528        | Compact pre-output representation                        |
| Output    | 16 → 2  | 34         | SEA and CFE simultaneously (linear, no activation)       |
| **Total** |         | **2,962**  |                                                          |

### 2.1 Why three hidden layers?

Physical crashworthiness simulation responses involve interactions at multiple scales:

- **Local scale**: R and CC are coupled through constraint C1; small changes in R require
  large changes in CC to remain geometrically feasible.
- **Global scale**: The combined effect of all four parameters determines the volume fraction
  and ultimately SEA and CFE.

A single hidden layer is theoretically sufficient for universal approximation, but deeper networks often represent complex compositional relationships more efficiently and with fewer parameters. Two
hidden layers begin capturing pairwise interactions. Three hidden layers can represent
higher-order interactions and the multi-scale structure of the response surface, which is
important for non-linear simulation data (Bengio, 2012; Mhaskar & Poggio, 2016).

Four or more hidden layers were not used because:

- With n=750 LF training samples, deeper networks risk overfitting
- The response is smooth (FEA physics are continuous) and doesn't require extreme depth
- Gradient flow degrades with depth when using tanh (Glorot & Bengio, 2010)

### 2.2 Why these layer widths (64 → 32 → 16)?

The exponentially decreasing width pattern (halving each layer) is a common and well-motivated
choice for regression surrogates (Goodfellow et al., 2016, §6.4.3):

- **Encoder perspective**: Wide first layer extracts many candidate features; subsequent layers
  select and compress into a low-dimensional representation sufficient for the two outputs.
- **Parameter efficiency**: Total parameters (~2,962) give a samples-to-parameters ratio of
  ~0.25 for 750 LF training samples. With L2 regularization (weight decay) and early stopping,
  this is well-regularized. Standard recommendation is >10 samples per parameter for
  unregularized models, but regularized networks can achieve good generalization with ratios
  as low as 0.1–0.5 (Goodfellow et al., 2016).

---

## 3. Activation Function: tanh

**tanh** (hyperbolic tangent) is chosen over ReLU and ELU for this problem:

| Property                   | tanh      | ReLU                        | ELU             |
| -------------------------- | --------- | --------------------------- | --------------- |
| Smooth (C∞)                | ✓         | ✗ (non-differentiable at 0) | ✓               |
| Bounded output             | ✓ (−1, 1) | ✗                           | ✗               |
| Zero-centered              | ✓         | ✗                           | approximately ✓ |
| Dying neuron risk          | Low       | High                        | Low             |
| Matches standardized input | Well      | Moderate                    | Well            |

Physical simulation responses are smooth, bounded functions of geometry — tanh's smooth,
bounded activations create intermediate features that match these properties. Because the inputs and outputs are standardized and the underlying response surface is expected to be smooth, tanh was selected as an initial activation function. Alternative activations such as ReLU and ELU may be explored during ablation studies.

**Reference**: LeCun et al. (1998) showed that tanh outperforms hard-limiting activations for
regression on physical data; Glorot & Bengio (2010) confirmed that tanh is the preferred
activation for networks initialized with Xavier/Glorot initialization.

---

## 4. Weight Initialization: Xavier Uniform

Xavier (Glorot) uniform initialization draws weights from:

```
W ~ Uniform(-√(6 / (fan_in + fan_out)), √(6 / (fan_in + fan_out)))
```

This is derived to preserve the variance of activations across layers during the forward pass
and the variance of gradients during backpropagation, preventing vanishing or exploding
gradients from the first iteration. It is mathematically optimal for tanh activations
(Glorot & Bengio, 2010).

---

## 5. Optimizer: Adam

**Adam** (Adaptive Moment Estimation, Kingma & Ba, 2014) is used for both training phases.

Adam combines momentum (first-moment estimate) and per-parameter adaptive learning rates
(second-moment estimate), providing:

1. Faster convergence than SGD for non-convex loss surfaces
2. Robustness to noisy gradient estimates (relevant for mini-batch training)
3. Implicit per-parameter learning rate scaling — useful when input features have very
   different scales even after standardization (e.g., CC range 4–22 vs VC range 4–10)

| Hyperparameter    | Phase 1 | Phase 2 | Justification                                        |
| ----------------- | ------- | ------- | ---------------------------------------------------- |
| Learning rate     | 1e-3    | 1e-4    | Lower LR in Phase 2 prevents catastrophic forgetting |
| Weight decay (L2) | 1e-4    | 1e-3    | Stronger regularization on small HF dataset          |
| β₁                | 0.9     | 0.9     | Default (Kingma & Ba, 2014)                          |
| β₂                | 0.999   | 0.999   | Default                                              |

---

## 6. Transfer Learning: Layer Freezing

### 6.1 Motivation

Yosinski et al. (2014) demonstrated in a large-scale study on image networks that:

- Features in early layers are **general** (transferable across tasks)
- Features in later layers are **task-specific** (need adaptation for the target domain)

In this problem, the "source domain" is shell FEA and the "target domain" is solid FEA.
The underlying physics (geometry → energy absorption) is the same; only the mesh discretization
(and thus the numerical accuracy) differs. Early layers encoding geometric relationships
(R-CC coupling via constraint C1, VC-R coupling via constraint C2) are fully transferable.
Later layers that map the learned representation to specific (SEA, CFE) values need adaptation.

### 6.2 Freeze strategy

In Phase 2, **Hidden-1 and Hidden-2 are frozen** (requires_grad=False):

```
Frozen (Phase 2):     Hidden-1 (4→64)   Hidden-2 (64→32)
                      [2,400 params frozen]

Trainable (Phase 2):  Hidden-3 (32→16)  Output (16→2)
                      [562 params trainable]
```

This leaves 562 trainable parameters to adapt on 78 HF training samples
(98 total × 80% inner train split), giving a samples-to-trainable-params ratio of ~0.14.
The stronger L2 weight decay (1e-3) and early stopping (patience=30) provide sufficient
regularization at this ratio.

**Why not freeze Hidden-3 as well (only update the output layer)?**
Output-layer-only fine-tuning (32 trainable parameters) is very conservative and risks
underfitting the systematic fidelity shift (mean SEA delta = +0.67, std = 0.54 in original
units). The observed delta variance requires learning a non-trivial correction, which
a single linear output layer cannot represent.

**Why not unfreeze all layers?**
With only 98 HF samples, fine-tuning all 2,962 parameters would lead to catastrophic
forgetting of the LF pre-training (McCloskey & Cohen, 1989; Kirkpatrick et al., 2017).

### 6.3 Lower learning rate in Phase 2

Using lr=1e-4 (10× lower than Phase 1) ensures that even the trainable layers update slowly
enough to preserve the pretrained representation in adjacent frozen layers' neighborhood.
This is the standard practice from domain adaptation literature (Pan & Yang, 2010).

---

## 7. Loss Function

```
L = MSE(SEA_pred_std, SEA_true_std) + MSE(CFE_pred_std, CFE_true_std)
```

**Why sum (not weight)?** After StandardScaler normalization, both outputs have zero mean and
unit variance. Equal-weight MSE is therefore appropriate — neither output dominates the
gradient. The per-output losses are logged separately during training to detect any
systematic imbalance.

---

## 8. Data Normalization

All inputs and outputs are standardized using `StandardScaler` (zero mean, unit variance):

- **Inputs**: prevents any single feature from dominating the gradient signal due to scale
  differences (R range ≈ 6.8 vs A range ≈ 60)
- **Outputs**: prevents SEA (larger values) from dominating over CFE in the joint loss

Scalers are fit on the **LF training split only** and applied (transform-only) to all other
splits. This prevents data leakage — a common source of optimistic evaluation bias
(Cawley & Talbot, 2010).

---

## 9. Training Protocol

### 9.1 Data splits

| Split                     | n   | Purpose                                            |
| ------------------------- | --- | -------------------------------------------------- |
| LF train                  | 750 | Phase 1 gradient updates                           |
| LF val                    | 188 | Phase 1 early stopping                             |
| HF finetune (inner train) | ~78 | Phase 2 gradient updates                           |
| HF finetune (inner val)   | ~20 | Phase 2 early stopping                             |
| HF test                   | 25  | Final evaluation only (never seen during training) |

### 9.2 Early stopping

Early stopping terminates training when the validation loss does not improve for `patience`
consecutive monitoring epochs, and restores the best-validation-loss weights. This prevents
overfitting while avoiding under-training (Prechelt, 1998).

| Phase | Monitoring interval | Patience                      |
| ----- | ------------------- | ----------------------------- |
| 1     | Every 25 epochs     | 50 epochs without improvement |
| 2     | Every 25 epochs     | 30 epochs without improvement |

### 9.3 Batch training

| Phase | Batch size       | Batches/epoch | Reasoning                                                                     |
| ----- | ---------------- | ------------- | ----------------------------------------------------------------------------- |
| 1     | 64               | 750/64 ≈ 12   | Mini-batch noise provides implicit regularization; 12 updates/epoch is enough |
| 2     | Full batch (~78) | 1             | Eliminates noisy gradient updates that could corrupt pretrained layers        |

---

## 10. Testing and Validation Plan

### 10.1 Metrics

All metrics are computed on the **HF test set** (25 points, held out throughout training):

| Metric | Formula             | Notes                         |
| ------ | ------------------- | ----------------------------- | --- | ------------------ |
| R²     | 1 − SS_res / SS_tot | Primary metric; 1.0 = perfect |
| RMSE   | √mean((ŷ−y)²)       | Same units as output          |
| MAE    | mean(               | ŷ−y                           | )   | Robust to outliers |

### 10.2 Ablation study

Four model variants are evaluated on the same HF test set:

| Variant | Training data             | Notes                           |
| ------- | ------------------------- | ------------------------------- |
| MLP-LF  | 750 LF only               | Baseline — no transfer          |
| MLP-HF0 | 98 HF from scratch        | Shows HF-only information limit |
| MLP-TL  | 750 LF + 98 HF (transfer) | Full pipeline                   |

### 10.3 Expected performance range

Based on literature on surrogate modeling for crashworthiness simulations
(Fang et al., 2005; Gu et al., 2001):

- SEA R² > 0.85 is considered a good surrogate fit for tube-crushing problems
- CFE R² > 0.75 is acceptable given its lower variance and more complex non-linearity
- Transfer learning typically yields 10–30% improvement in R² on low-data HF targets
  (Yosinski et al., 2014; Pan & Yang, 2010)

---

## 11. Fine-Tuning with New HF Data

To incorporate additional solid simulation results into the model:

```python
from src.ml_models.train_mlp import run_phase2
from src.ml_models.data_loader import load_data

# 1. Re-run data loading with the updated CSV (new rows appended to HF CSV)
splits = load_data(seed=42)

# 2. Fine-tune from the existing Phase-1 checkpoint
model = run_phase2(splits, source_model=None)  # loads mlp_pretrained_lf.pt automatically
```

No changes to Phase 1 are needed — the pre-trained LF weights are preserved. Only the
HF correction (hidden-3 + output layer) is re-learned on the expanded HF dataset.

---

## 12. Active Learning Integration

### 12.1 GP-Guided Sampling Benefits the MLP Too

The active learning acquisition signal is derived entirely from the GP's posterior
uncertainty — not from the MLP.  However, the same new high-fidelity simulation
data acquired through GP-guided sampling also improves the MLP.

The GP identifies *where* the design space is under-characterized; the resulting
new solid simulations reduce uncertainty for both models simultaneously.  No
separate MLP-specific acquisition mechanism is required.  The workflow is:

```
propose (GP acquisition) → simulate (new HF points) → append to HF CSV
  → retrain GP correction (train_gp --phase correction)
  → retrain MLP Phase 2 (train_mlp --phase 2)
```

### 12.2 No Separate MLP Uncertainty Is Required

The MLP is a point predictor — it returns a single predicted value, not a posterior
distribution.  Principled MLP uncertainty would require architectural changes such
as MC Dropout (Gal & Ghahramani, 2016) or Deep Ensembles (Lakshminarayanan et al.,
2017), both of which increase training and inference cost.

The GP's posterior standard deviation is a sufficient proxy for MLP uncertainty:
both models are trained on the same HF data, so regions where the GP is uncertain
are also regions where the MLP is likely to generalize poorly.  Using the GP as
the uncertainty oracle avoids doubling the model complexity.

### 12.3 When to Retrigger Phase 2

**Rule**: retrigger Phase 2 when the number of new HF points added satisfies

```
n_new_hf >= 0.10 * n_current_hf
```

For typical active sampling batches of k = 32–64 new points and a current HF
dataset of ~123 points, this threshold (~12 points) is always exceeded by a
single batch.  Always retrigger after each active sampling round.

Commands after appending new simulation results to the HF CSV:

```bash
# 1. Refit only the correction GPs (fast — O(n_HF³) not O(n_LF³))
python -m src.ml_models.train_gp --phase correction

# 2. Re-run MLP Phase 2 on the expanded HF dataset
python -m src.ml_models.train_mlp --phase 2
```

Phase 1 (LF pre-training) does not need to be re-run — the shell simulation data
is fixed and the pre-trained weights are frozen during Phase 2.

### 12.4 Catastrophic Forgetting Risk with Small Batches

With a current HF fine-tune set of ~98 points, adding k = 32–64 new points
represents a 33–65% increase in the fine-tune dataset.  This is a significant
regime change for gradient-based updating.

**Mitigation already in place**: Phase 2 freezes the first two hidden layers
(only 562 out of 2962 total parameters are updated).  This strongly limits
catastrophic forgetting of the LF pre-training because the bulk of the network's
representational capacity is preserved (Kirkpatrick et al., 2017).

Practical guidance:
- If CFE R² *degrades* after Phase 2 following a new batch, reduce the Phase 2
  learning rate (default `lr=1e-4`; try `lr=5e-5`).
- If SEA R² degrades, the new HF points may contain outliers — inspect the
  simulation outputs before appending.
- Monitor both outputs after every round; the goal is monotonically improving
  CFE R² until it reaches ≥ 0.80.

---

## 13. Surrogate Optimization via Differential Evolution

### 13.1 Why Differential Evolution for the MLP

The MLP is a deterministic point predictor — it has no posterior variance σ, so
acquisition functions (EI, UCB) that rely on uncertainty estimates are inapplicable.
What is needed instead is a global search method that can find the maximum of the
scalarized objective `α·SEA_norm + (1−α)·CFE` over the constrained design space.

The tanh activation produces smooth but potentially multimodal objective landscapes:
multiple local optima can exist at different (CC, VC) combinations. Gradient-based
methods from random restarts can miss many of these.

**Differential Evolution** (Storn & Price, 1997) addresses this:
- Population-based global optimizer — no gradient needed.
- Handles mixed-integer variables natively via the `integrality` flag (scipy ≥ 1.9).
- Supports nonlinear constraint enforcement via `scipy.optimize.NonlinearConstraint`.
- Robust to multimodal landscapes due to population diversity.

### 13.2 Mixed-Integer Problem Formulation

The design vector is `x = [R, A, CC, VC]` where R and A are continuous, CC and VC are
integer-valued:

```
minimize    −Obj(x; α)          [scipy minimizes; negate to maximize]
subject to:
  C1: CC · √3 · R  <  π · OD
  C2: VC  <  (L − 2(G+R)) / |2R − √3·π·OD/(6·CC)|  +  1
  C3: VC  <  (L − 2(R+G)) / R  +  1
bounds:       R ∈ [2, 8.8],  A ∈ [30, 90],  CC ∈ [4, 22],  VC ∈ [4, 10]
integrality:  [False, False, True, True]
```

Using `integrality=[False, False, True, True]` enforces integer-valued CC and VC
*throughout the search*, not just at the end. This is superior to post-hoc rounding:

- Rounding a continuous solution can violate constraints, requiring a feasibility re-check.
- A rounded point is not necessarily locally optimal in the integer sense.
- DE's integrality flag maintains a valid mixed-integer population at every generation.

### 13.3 Obtaining Multiple Diverse Proposals

DE maintains a *population* of `pop_size × 4` individuals throughout the search. By the
final generation this population has converged toward the best regions of the landscape,
but the individuals are not all identical — they spread across nearby high-quality designs.

The optimizer extracts the **full final population** after a single DE run, evaluates and
filters each individual for feasibility, then ranks by objective and applies the maxmin
diversity filter to select `n_proposals` spread-out solutions.

This is more reliable than running `n_proposals` independent DE runs: independent runs
all converge to the same dominant global optimum when one exists, collapsing to a single
unique solution after deduplication. The final population of a single run naturally
contains diversity because DE's mutation and crossover operators maintain population
spread as a core mechanism of the algorithm.

### 13.4 Multi-Start L-BFGS-B as Alternative (`method=lbfgsb`)

The MLP with tanh activations is continuously differentiable everywhere. The full
gradient of the objective with respect to raw inputs can be computed analytically via
PyTorch autograd, chaining through the input scaler, the network, and the output scaler:

```
∂Obj/∂X = [α/SEA_ref, 1−α] · diag(σ_y) · J_MLP · diag(1/σ_x)
```

where `J_MLP = ∂Y_std/∂X_std` is the network Jacobian computed by `.backward()`.

This gradient is passed to `scipy.optimize.minimize(method='L-BFGS-B', jac=True)` for
fast gradient-based local convergence. Multiple random starting points from a Sobol pool
are used to explore the landscape.

**Continuous relaxation:** CC and VC are treated as continuous during optimization and
rounded to the nearest integer post-convergence. Solutions that become infeasible after
rounding are discarded.

**Trade-offs vs DE:**
- L-BFGS-B is faster per iteration and per restart.
- DE is more robust to multimodal landscapes (global vs local search).
- L-BFGS-B may miss the global optimum when many restarts converge to the same basin.
- Recommended for quick exploration; use DE for production optimization.

### 13.5 Constraint Handling in DE

The constraints C1, C2, C3 are implemented as `scipy.optimize.NonlinearConstraint`
objects. Each returns a value that must be ≥ 0 (strict feasibility):

```python
NonlinearConstraint(c1_fn, lb=1e-6, ub=inf)   # π·OD − CC·√3·R > 0
NonlinearConstraint(c2_fn, lb=1e-6, ub=inf)   # C2 slack > 0
NonlinearConstraint(c3_fn, lb=1e-6, ub=inf)   # C3 slack > 0
```

DE uses COBYLA-like penalty enforcement internally to steer the population toward
feasible regions. Contrast with the GP path, where all candidates are pre-filtered
through `generate_sobol()` before any evaluation — constraint enforcement is structural
rather than penalty-based.

### 13.6 References

| Citation | Relevance |
|----------|----------|
| Storn, R., & Price, K. (1997). Differential Evolution — a simple and efficient heuristic for global optimization over continuous spaces. *J. Global Optimization*, 11, 341–359. | DE algorithm |
| Liu, D.C., & Nocedal, J. (1989). On the limited memory BFGS method for large scale optimization. *Mathematical Programming*, 45, 503–528. | L-BFGS-B algorithm |
| Marler, R.T., & Arora, J.S. (2004). Survey of multi-objective optimization methods for engineering. *Structural and Multidisciplinary Optimization*, 26(6), 369–395. | Scalarization, normalization rationale |
| Deb, K., et al. (2002). NSGA-II. *IEEE Trans. Evolutionary Computation*, 6(2), 182–197. | Non-dominated sorting |
| Miettinen, K. (1999). *Nonlinear Multiobjective Optimization*. Kluwer. | Chebyshev scalarization |

---

## 14. References

| Citation                                                                                                                                          | Relevance                                         |
| ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| Hornik, K., Stinchcombe, M., & White, H. (1989). Multilayer feedforward networks are universal approximators. _Neural Networks_, 2(5), 359–366.   | Universal approximation justification             |
| Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. _AISTATS_.                            | Xavier initialization, tanh analysis              |
| LeCun, Y., et al. (1998). Efficient backprop. _Neural Networks: Tricks of the Trade_. Springer.                                                   | Activation function selection                     |
| Kingma, D.P., & Ba, J. (2014). Adam: A method for stochastic optimization. _ICLR 2015_. arXiv:1412.6980.                                          | Adam optimizer                                    |
| Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How transferable are features in deep neural networks? _NeurIPS 27_.                    | Layer freezing strategy                           |
| Pan, S.J., & Yang, Q. (2010). A survey on transfer learning. _IEEE TKDE_, 22(10), 1345–1359.                                                      | Transfer learning framework                       |
| Bengio, Y. (2012). Practical recommendations for gradient-based training of deep architectures. _Neural Networks: Tricks of the Trade_. Springer. | Depth, width, regularization                      |
| Goodfellow, I., Bengio, Y., & Courville, A. (2016). _Deep Learning_. MIT Press. §6.4, §7.8                                                        | Architecture design, regularization               |
| Prechelt, L. (1998). Early stopping — but when? _Neural Networks: Tricks of the Trade_.                                                           | Early stopping justification                      |
| Cawley, G.C., & Talbot, N.L.C. (2010). On over-fitting in model selection. _JMLR_, 11.                                                            | Scaler-fit-on-train-only protocol                 |
| Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. _PNAS_, 114(13).                                           | Catastrophic forgetting — why full unfreeze fails |
| McCloskey, M., & Cohen, N.J. (1989). Catastrophic interference in connectionist networks. _Psychology of Learning and Motivation_, 24.            | Original catastrophic forgetting paper            |
| Fang, H., et al. (2005). Numerical simulations and crash optimizations of cubic foam-filled thin-walled structures. _IJCV_.                       | Crashworthiness surrogate benchmarks              |
| Mhaskar, H., & Poggio, T. (2016). Deep vs. shallow networks: An approximation theory perspective. _Analysis and Applications_, 14(06).            | Depth advantage for composition                   |
