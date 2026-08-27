# Gaussian Process Surrogate Model — Design Document

## 1. Problem Framing

The multi-fidelity Gaussian Process (GP) provides a **probabilistic** surrogate that maps:

```
f : (R, A, CC, VC) → (SEA, CFE)
```

Unlike the MLP, which returns a point estimate, the GP returns a full posterior distribution
over the output — including a **posterior mean** (the prediction) and a **posterior standard
deviation** (the uncertainty).  This is critical for material design, where decisions should
be informed by how confident the model is at a given input location.

---

## 2. Gaussian Process Fundamentals

A Gaussian Process is a distribution over functions (Rasmussen & Williams, 2006, §2.2):

```
f(x) ~ GP(m(x), k(x, x'))
```

where:
- `m(x)` is the prior mean function (set to 0 after standardization)
- `k(x, x')` is the kernel (covariance function) encoding similarity between points

Given n training observations **X** and **y**, the posterior at a new input x* is:

```
P(f(x*) | X, y, x*) = N(μ*, σ²*)
μ*  = k(x*, X) [k(X, X) + σ²ₙI]⁻¹ y
σ²* = k(x*, x*) − k(x*, X) [k(X, X) + σ²ₙI]⁻¹ k(X, x*)
```

The posterior mean is the prediction; the posterior variance quantifies uncertainty.
Regions far from training data have high variance; regions near training data have low variance.

**Computational complexity**: Fitting requires computing the Cholesky factorization of the
n×n Gram matrix — O(n³/3) in time, O(n²) in memory.  With n ≈ 938 LF training points,
this is feasible on modern hardware in ≈ 30s–3min per GP.

---

## 3. Why Two Independent GPs (not one scalarized GP)?

Three options for the multi-output formulation were considered:

| Option | Description | Verdict |
|--------|-------------|---------|
| **A: Two independent GPs** | Separate GP for SEA, separate GP for CFE | ✓ Chosen |
| B: Single scalarized GP | Predict Obj = SEA − k(1−CFE) directly | ✗ Rejected |
| C: Multi-output GP (ICM/LMC) | Joint GP with inter-output covariance | ✗ Too expensive |

**Against B (scalarized)**: Predicting the objective directly conflates two physically distinct
quantities with different noise structures and different response shapes.  It requires
committing to a specific k value at training time — using k = mean(SEA_HF) introduces
a subtle form of data leakage where HF statistics influence the model structure.  Additionally,
once an objective-only surrogate is trained, SEA and CFE cannot be recovered separately for
multi-objective analysis.

**Against C (multi-output GP)**: True multi-output GPs using the Intrinsic Coregionalization
Model (ICM) or Linear Model of Coregionalization (LMC) scale as O((nP)³) where P is the
number of outputs — computationally expensive for n ≈ 938 and beyond the scope of
scikit-learn's built-in implementation.  Furthermore, Álvarez et al. (2012) note that ICM
benefits diminish when output correlations are moderate, which is likely here (SEA and CFE
have different physical drivers).

**For A (two independent GPs)**: The objective is always computable as a derived quantity:

```
Obj(x) = GP_SEA.predict(x) − k × (1 − GP_CFE.predict(x))
```

This provides full flexibility in k selection post-hoc and enables separate uncertainty
bounds for each physical quantity.

---

## 4. Kernel Design

### 4.1  Chosen kernel

```python
C(amplitude) * RBF(length_scale=[l_R, l_A, l_CC, l_VC]) + WhiteKernel(noise)
```

### 4.2  Why RBF?

The Radial Basis Function (squared exponential) kernel:

```
k_RBF(x, x') = exp(−||x − x'||² / (2 l²))
```

corresponds to a GP prior over **infinitely differentiable** (analytic) functions
(Rasmussen & Williams, 2006, §4.2).  FEA simulation responses are smooth functions of
geometry — there are no discontinuities in SEA or CFE as R, A, CC, or VC change continuously.
The RBF kernel is therefore a physically motivated choice.

Alternative kernels considered:

| Kernel | Why not |
|--------|---------|
| Matérn 5/2 | Allows rougher functions; not needed for smooth FEA responses |
| Linear | Assumes global linearity — clearly insufficient |
| Periodic | No periodic structure expected in (R, A, CC, VC) space |

### 4.3  ARD (Automatic Relevance Determination)

Using a **separate length scale per input dimension** (`length_scale=[l_R, l_A, l_CC, l_VC]`)
allows the kernel to discover that:
- R (range ~6.8 after standardization: ≈ ±2σ) has a different effective correlation length than
  CC (integer-valued, range 4–22) or VC (integer-valued, range 4–10)
- Some parameters may have weaker effect on SEA/CFE than others — ARD implements soft
  feature selection by learning very large length scales for irrelevant features

ARD was introduced in the context of GPs by MacKay (1992) and Neal (1996) and is standard
practice for surrogate modeling with heterogeneous inputs (Rasmussen & Williams, 2006, §5.1).

### 4.4  WhiteKernel (noise term)

```
k_noise(x, x') = σ²ₙ δ(x, x')
```

This Kronecker delta term (non-zero only when x = x') models stochastic output noise —
arising from numerical variability in the FEA solver (mesh generation randomness, solver
tolerances).  The noise variance σ²ₙ is optimized jointly with the RBF hyperparameters.

### 4.5  Hyperparameter optimization

All kernel hyperparameters (amplitude, 4 length scales, noise variance) are estimated by
maximizing the **log marginal likelihood**:

```
log p(y | X, θ) = −½ yᵀ K⁻¹ y − ½ log|K| − n/2 log(2π)
```

This provides a principled trade-off between data fit and model complexity (Occam's razor
is built in via the determinant term).  Optimization uses L-BFGS-B with `n_restarts=5`
random starting points to mitigate local optima (Rasmussen & Williams, 2006, §5.4.1).

---

## 5. Multi-Fidelity Framework: Kennedy-O'Hagan (2000)

### 5.1  Motivation

Two simulation fidelities are available:
- **Low-fidelity (LF)**: shell FEA — 938 points, fast to compute
- **High-fidelity (HF)**: solid FEA — 123 points, slow, higher physical accuracy

The measured fidelity gap (computed at the 118 overlapping design points):

| Output | LF mean | HF mean | Mean delta | Delta std |
|--------|---------|---------|-----------|-----------|
| SEA | 2.208 | 2.748 | **+0.669** | 0.544 |
| CFE | 0.638 | 0.648 | **+0.011** | 0.094 |

SEA shows a **systematic positive bias** of ~30% between shell and solid simulations.
This bias is not constant — it varies with input parameters (std = 0.544), meaning a simple
global offset correction would be insufficient.  The delta must be modeled as a function of x.

### 5.2  The additive correction model

For each output (SEA and CFE independently), the Kennedy-O'Hagan (2000) model decomposes:

```
f_HF(x) = f_LF(x) + δ(x)
```

where δ(x) = f_HF(x) − f_LF(x) is the fidelity bias function.  Training proceeds in two
phases:

**Phase 1 — Fit GP_LF on 938 LF samples:**
```
GP_LF: (R, A, CC, VC) → P(SEA_LF | X_LF, y_LF)
```

**Phase 2 — Compute residuals and fit GP_delta on 98 HF samples:**
```
δᵢ = SEA_HF(xᵢ) − GP_LF.predict(xᵢ)   for i = 1, …, 98
GP_delta: (R, A, CC, VC) → P(δ | X_HF, δ)
```

**Prediction at any new x:**
```
μ_HF(x)  = μ_LF(x)  + μ_delta(x)
σ²_HF(x) = σ²_LF(x) + σ²_delta(x)
```

Variances add under the assumption that the LF and delta GPs are conditioned on independent
data (the LF data and the HF residuals respectively) — the standard independence assumption
in the multi-fidelity surrogate literature (Forrester et al., 2007).

### 5.3  Why K-O'Hagan over alternatives?

| Approach | Pros | Cons |
|----------|------|------|
| **K-O'Hagan (chosen)** | Uses all 938 LF + 98 HF; principled uncertainty; handles non-overlapping points | Assumes approximate independence |
| HF-only GP | Simpler | Discards 938 LF samples; poor generalization with only 98 points |
| Co-kriging | Handles correlations between fidelities jointly | More complex; requires overlapping points for efficient estimation |
| Simple mean-shift | Trivial to implement | Cannot model spatially varying bias |

The K-O'Hagan framework is also consistent with the multi-fidelity GP framework validated
on engineering design problems by Forrester et al. (2007), who demonstrated 60–90% reduction
in required HF simulations compared to using HF data alone.

### 5.4  Handling non-overlapping HF points

5 of the 123 HF points have no exact counterpart in the LF dataset.  This is not a problem:
- Phase 1 fits GP_LF on all 938 LF points as a regression, not a table lookup
- GP_LF.predict() can be evaluated at any input location, including the 5 non-overlapping HF points
- The delta δᵢ = Y_HF(xᵢ) − GP_LF.predict(xᵢ) is valid for all 123 HF points

---

## 6. Uncertainty Quantification

Unlike the MLP, the GP provides a calibrated uncertainty estimate at every prediction location.
The posterior standard deviation:

```
σ_HF(x) = √(σ²_LF(x) + σ²_delta(x))
```

has a natural interpretation:
- High σ at sparse regions of input space (far from any training point)
- Low σ near training data (well-characterized regions)
- 95% credible interval: μ_HF(x) ± 1.96 σ_HF(x)

This enables **uncertainty-aware design exploration**: rather than selecting designs by
predicted performance alone, one can use acquisition functions (Expected Improvement,
Upper Confidence Bound) to balance predicted performance and uncertainty — the foundation
of Bayesian Optimization (Snoek et al., 2012).

---

## 7. Data Normalization

**Inputs (X)**: StandardScaler fit on LF train split.  After standardization, all features
have zero mean and unit variance.  The RBF kernel's `length_scale_bounds=(1e-2, 1e2)` covers
the expected correlation lengths in standardized space.

**Outputs (Y)**: StandardScaler fit on LF train split (separately for SEA and CFE).
`normalize_y=False` is set in sklearn to avoid double normalization.  All GPs operate in
standardized output space; predictions are inverse-transformed for reporting.

**Standard deviation inverse-transform**: GP σ is in standardized units.  Converting to
original units: `σ_orig = σ_std × scaler.scale_` (the training standard deviation).
This is correct because the StandardScaler maps y → (y − μ) / σ_train, so the derivative
of the inverse is σ_train.

---

## 8. Testing and Validation Plan

### 8.1  Metrics

Same as MLP: R², RMSE, MAE on the HF test set (25 points, never seen during training).

### 8.2  Ablation study

| Variant | Training | Notes |
|---------|----------|-------|
| GP-LF | 938 LF only | Baseline |
| GP-TL | 938 LF + 98 HF correction | Full K-O'Hagan pipeline |

### 8.3  Calibration check

A well-calibrated GP should have approximately:
- 68% of test points within ±1σ of the posterior mean
- 95% within ±2σ
- 99.7% within ±3σ

This is checked by computing the **empirical coverage** of the confidence intervals on the
25-point test set.  Miscalibration (too wide or too narrow intervals) indicates kernel
misspecification and guides kernel redesign.

### 8.4  Expected performance

Based on Kennedy & O'Hagan (2000) and Forrester et al. (2007) benchmarks on engineering
surrogate problems with similar sample sizes:
- K-O'Hagan GPs typically outperform HF-only GPs by 15–40% in RMSE when the LF/HF
  ratio is ≥ 5:1 (here it is 938:98 ≈ 9.6:1)
- SEA R² > 0.80 is expected given the systematic delta structure
- CFE R² may be lower due to its narrower variance range

---

## 9. Fine-Tuning with New HF Data

To incorporate additional solid simulation results:

```python
from src.ml_models.train_gp import run_correction_fit
from src.ml_models.data_loader import load_data

# 1. Re-run data loading (updated HF CSV with new rows appended)
splits = load_data(seed=42)

# 2. Re-fit the correction GPs only (LF GP is unchanged)
from src.ml_models.gp_model import MultiFidelityGP
gp = MultiFidelityGP.load("models/gp")         # load existing LF GPs
gp.fit_hf_correction(splits.X_hf_finetune, splits.Y_hf_finetune)  # refit delta
gp.save("models/gp")
```

The LF GPs (trained on 938 shell points) are unchanged.  Only the correction GPs need
refitting when new HF data arrives — O(n_HF³) cost, not O((n_LF + n_HF)³).

---

## 10. Active Learning and Sequential Experimental Design

### 10.1 Why Random Sobol Sampling Is Suboptimal for Targeted Refinement

Random or quasi-random space-filling designs (including Sobol sequences) are optimal
for minimizing the *average* mean-squared error under a uniform prior over the input
space (Sacks et al., 1989).  However, once an initial GP surrogate has been trained,
we gain information about where the model is uncertain.  Continuing to add uniformly
distributed points wastes budget on well-characterized regions.

Acquisition-function-guided sampling instead targets the *maximum* expected reduction
in posterior variance — selecting points where the GP's posterior standard deviation
σ(x) is large.  This is equivalent to minimizing the maximum expected squared error
(integrated Bayes risk under a non-uniform, GP-informed distribution over x).

For our problem: SEA has R² ≈ 0.95 (GP is confident everywhere), while CFE has
R² ≈ 0.2–0.4 (GP uncertainty is large over most of the domain).  Uniform Sobol
sampling would add points proportionally to volume, whereas acquisition-guided
sampling directs the budget toward the regions where CFE uncertainty is highest.

### 10.2 Expected Improvement: Derivation from Truncated Gaussian Integral

Expected Improvement (EI) for a maximization problem (Jones et al., 1998) is the
expected amount by which a new point x exceeds the current best observed value y*:

```
EI(x) = E[max(f(x) - y*, 0)]
```

Since the GP posterior at x is Gaussian with mean μ(x) and std σ(x), this integral
has a closed form:

```
Z(x)  = (μ(x) - y*) / σ(x)        [when σ(x) > 0]

EI(x) = (μ(x) - y*) · Φ(Z) + σ(x) · φ(Z)
EI(x) = 0                           [when σ(x) = 0]

Φ = scipy.stats.norm.cdf   φ = scipy.stats.norm.pdf
```

The first term `(μ - y*) · Φ(Z)` captures **exploitation**: points with high
predicted mean relative to y* are preferred.  The second term `σ · φ(Z)` captures
**exploration**: points with high uncertainty are also preferred, even if their
predicted mean is below y*.

The plug-in estimator sets `y* = max(μ(x))` over the candidate pool (Frazier, 2018,
§2.2), avoiding the need to find the true maximum of the posterior mean.

EI is computed separately for SEA (maximize) and CFE (maximize; CFE closer to 1 is
better).

### 10.3 Weighted Composite EI for Multi-Output Problems

Combining EI_SEA and EI_CFE naively by addition is problematic because EI_SEA is
in units of N·mm/mm³ while EI_CFE is dimensionless — their magnitudes are
incomparable.

The Svenson & Santner (2016) approach normalizes each EI by its maximum over the
candidate pool before combining:

```
WCEI(x) = w_SEA · [EI_SEA(x) / max(EI_SEA)]
         + w_CFE · [EI_CFE(x) / max(EI_CFE)]
```

Edge cases:
- If `max(EI_SEA) = 0`, the SEA term collapses to 0 and the full weight transfers
  to CFE (GP is already certain about SEA — this is consistent with R² ≈ 0.95).
- If both maxima are 0, the function falls back to `uncertainty_cfe` with a warning.

Default weights: `w_SEA = 0.3, w_CFE = 0.7` (see §10.5 for the rationale).

### 10.4 Pure Uncertainty Sampling as Alternative Criterion

Uncertainty sampling (Sacks et al., 1989; Settles, 2012) selects the point where
the GP's posterior standard deviation is largest:

```
US(x)     = w_SEA · [σ_SEA(x) / max(σ_SEA)] + w_CFE · [σ_CFE(x) / max(σ_CFE)]
US_CFE(x) = σ_CFE(x)   (default; unnormalized, CFE only)
```

Uncertainty sampling is preferable to EI when the goal is **model improvement**
rather than optimization — it reduces the global uncertainty of the surrogate
regardless of where the current best y* is.  EI is better when there is a specific
optimization target.  Given that CFE R² ≈ 0.2–0.4, reducing uncertainty globally
is a higher priority than finding the maximum CFE point, which is why
`uncertainty_cfe` is the default acquisition function.

### 10.5 The CFE-Bias Rationale

The default weights `w_SEA = 0.3, w_CFE = 0.7` reflect the current performance gap:

| Output | Test R² | Model status |
|--------|---------|-------------|
| SEA    | ≈ 0.95  | Well-characterized; little budget needed |
| CFE    | ≈ 0.2–0.4 | Poorly characterized; dominant bottleneck |

The physical reason CFE is harder to model: CFE depends on the buckling initiation
mode — a high-frequency, spatially localized response that changes abruptly with
small variations in geometry.  The GP cannot interpolate this mode from sparse HF
data because the response is not smooth in the input space.

The 70% budget toward CFE is appropriate until CFE R² ≥ 0.80, at which point weights
can be balanced (e.g., `w_SEA = 0.5, w_CFE = 0.5`) or shifted toward optimization
(switching from `uncertainty_cfe` to `ei_weighted`).

### 10.6 Practical Pipeline

The active sampling pipeline in `src/active_sampler.py` proceeds as:

1. **Candidate pool**: generate ~4096 constraint-satisfying Sobol points using
   `generate_sobol(k=pool_size, OD=40, L=50)` without a named instance (avoids
   polluting the project's persistent Sobol state).
2. **GP query**: evaluate the GP posterior at all candidates → `(μ_SEA, σ_SEA, μ_CFE, σ_CFE)`.
3. **Acquisition**: compute acquisition values for each candidate using the selected function.
4. **Pre-filter**: select top 8k candidates by acquisition value.
5. **Diversity filter**: apply greedy farthest-point (maxmin) subsampling to select
   k spatially diverse proposals from the top 8k.
6. **Near-duplicate check**: remove any proposed point within 5% of the normalized
   space diagonal from existing LF or HF data.
7. **Output**: print summary table; save CSV and diagnostic plots.

### 10.7 References

| Citation | Relevance |
|----------|----------|
| Jones, D.R., Schonlau, M., & Welch, W.J. (1998). Efficient global optimization of expensive black-box functions. *J. Global Optimization*, 13(4), 455–492. | EI formula and derivation |
| Sacks, J., Welch, W.J., Mitchell, T.J., & Wynn, H.P. (1989). Design and analysis of computer experiments. *Statistical Science*, 4(4), 409–435. | Uncertainty sampling, space-filling rationale |
| Frazier, P.I. (2018). A tutorial on Bayesian optimization. arXiv:1807.02811. | EI plug-in estimator, general BO framing |
| Svenson, J.D., & Santner, T.J. (2016). Multiobjective optimization of expensive black-box functions via nonstationary Gaussian process models. *J. Global Optimization*, 64(2), 297–310. | Weighted composite EI normalization |
| Srinivas, N., Krause, A., Kakade, S., & Seeger, M. (2012). Information-theoretic regret bounds for Gaussian process optimization. *IEEE Trans. Information Theory*, 58(5), 3250–3265. | UCB acquisition |
| Settles, B. (2012). *Active Learning*. Synthesis Lectures on AI and Machine Learning. Morgan & Claypool. | Uncertainty sampling framing, model improvement vs optimization |
| Forrester, A.I.J., Sóbester, A., & Keane, A.J. (2007). Multi-fidelity optimization via surrogate modelling. *Proc. Royal Society A*, 463, 3251–3269. | Campaign size guidance, multi-fidelity context |

---

## 11. Surrogate-Based Bayesian Optimization

### 11.1 From Forward Model to Design Optimization

The GP surrogate was trained as a *forward model*: given a design (R, A, CC, VC), predict
(SEA, CFE). Optimization inverts this relationship: given the surrogate, *search* for the
design that produces the best outputs.

Running optimization directly on FEA would require a new simulation per candidate evaluation —
each taking minutes to hours. The surrogate replaces FEA during the search, reducing evaluation
cost to microseconds. The result is a ranked list of recommended designs to *then* simulate
(single-shot surrogate optimization; Koziel & Leifsson, 2016).

The GP does **not** re-fit during optimization. It is used as a fixed, trained predictor.

### 11.2 Multi-Objective Scalarization: Weighted Sum vs Penalty Formula

Two objectives must be balanced: maximize SEA (N·mm/mm³) and maximize CFE (dimensionless,
target = 1). These live on different scales, so direct summation gives an arbitrary weighting
that depends on units.

**Normalized weighted sum** (default):

```
SEA_ref = max(SEA_pred)  over candidate pool
Obj(x; α) = α · (SEA(x) / SEA_ref)  +  (1−α) · CFE(x)
```

Dividing by `SEA_ref` maps SEA into [0, 1], matching the natural [0, 1] range of CFE.
Under this normalization, `α = 0.5` gives equal weight regardless of the absolute magnitude
of SEA. This is the recommended approach (Marler & Arora, 2004).

**Penalty formula** (existing project convention):

```
Obj(x; k) = SEA(x) − k · (1 − CFE(x))
```

where `k = mean(SEA_pred)` over the pool. This is the formula already used throughout
`predict_objective()`. It gives the same rank ordering as the weighted sum at `k =
SEA_ref · (1−α)/α`, but scale-sensitivity makes the default penalty formula harder to
interpret across datasets.

### 11.3 Acquisition-Guided Optimization (EI, UCB)

Pure exploitation (rank by `μ_Obj`) can get stuck at the current mean-maximum and miss
designs where `σ_CFE` is large — places where the true CFE might be substantially higher
than the GP's conservative mean estimate. Acquisition functions trade off exploitation
against exploration.

**EI on the composite objective** (`mode=ei`):

Since SEA and CFE GPs are fitted independently, their correlation is unknown. We apply the
independence approximation, which gives a conservative (upper-bound) estimate of composite
variance under positive correlation (Frazier, 2018):

```
μ_Obj(x; α)  =  α · μ_SEA(x) / SEA_ref  +  (1−α) · μ_CFE(x)
σ_Obj(x; α)  =  sqrt( (α/SEA_ref)² · σ²_SEA  +  (1−α)² · σ²_CFE )
y*           =  max(μ_Obj)  over pool   [plug-in estimator]
Z(x)         =  (μ_Obj(x) − y*) / σ_Obj(x)
EI(x)        =  (μ_Obj(x) − y*) · Φ(Z)  +  σ_Obj(x) · φ(Z)   [σ_Obj > 0]
EI(x)        =  0                                                [σ_Obj = 0]
```

EI balances regions with high mean objective against regions where the GP is uncertain
(Jones et al., 1998).

**UCB on the composite** (`mode=ucb`):

```
UCB(x; α, β)  =  μ_Obj(x; α)  +  β · σ_Obj(x; α)
```

`β = 2.0` corresponds to approximately a 95% upper confidence bound (Srinivas et al., 2012).
Larger β increases exploration; β = 0 reduces to exploitation.

**When to prefer EI or UCB over exploitation:**
- High `σ_CFE` regions may contain designs with true CFE substantially above the GP mean.
- After adding new HF points, EI/UCB naturally shifts focus to unexplored regions.
- For small HF datasets (< 50 points), GP uncertainty is significant and EI/UCB add real value.

### 11.4 Pareto Front Generation: Non-Dominated Sorting and Chebyshev Scalarization

**Non-dominated sorting** (Deb et al., 2002):

```
x dominates y  iff  SEA(x) ≥ SEA(y)  AND  CFE(x) ≥ CFE(y),  with at least one strict.
Pareto front P = { x : no y in pool dominates x }
```

Applied to all pool candidates, this gives a sample approximation of the true Pareto front.
Complexity is O(n²); adequate for pool sizes up to ~10,000.

**Chebyshev scalarization** (Miettinen, 1999):

The weighted sum `α·SEA_norm + (1−α)·CFE` traces only the *convex hull* of the Pareto
front — it cannot find solutions on concave regions. The Chebyshev approach is guaranteed
to find Pareto-optimal solutions for any front shape:

```
Obj_cheby(x; α)  =  max( α · (1 − SEA_norm(x)),   (1−α) · (1 − CFE(x)) )
```

For each `α ∈ linspace(0, 1, n_weights)`, find `argmin_x Obj_cheby` over the pool.
Collect all solutions → deduplicate → keep non-dominated subset → Pareto front.

The optimizer returns the **full Pareto front** in `mode=pareto` (the `--n` flag is
ignored). Each point on the front represents a different SEA/CFE trade-off. Points are
labeled with `(CC, VC)` — the discrete parameters that most strongly drive the trade-off.

### 11.5 Constraint Handling: Pre-Filtered Sobol Pool

All candidates in the optimization pool are generated through `generate_sobol()` with
`OD=40, L=50, G=3`, which calls `apply_constraints()` internally. Every candidate in the
pool satisfies C1, C2, C3 exactly before any GP evaluation.

This means no penalty method or constraint-handling machinery is needed on the GP side —
constraints are enforced structurally at the pool-generation step. This contrasts with the
MLP path (Section 13 in MLP_DESIGN.md), which uses `scipy.optimize.NonlinearConstraint`
for explicit enforcement during iterative search.

### 11.6 References

| Citation | Relevance |
|----------|----------|
| Jones, D.R., Schonlau, M., & Welch, W.J. (1998). Efficient global optimization of expensive black-box functions. *J. Global Optimization*, 13(4), 455–492. | EI acquisition function |
| Srinivas, N., Krause, A., Kakade, S., & Seeger, M. (2012). Information-theoretic regret bounds for Gaussian process optimization. *IEEE Trans. Information Theory*, 58(5). | UCB acquisition, exploration–exploitation |
| Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Trans. Evolutionary Computation*, 6(2), 182–197. | Non-dominated sorting definition and complexity |
| Miettinen, K. (1999). *Nonlinear Multiobjective Optimization*. Kluwer. | Chebyshev scalarization, Pareto theory |
| Marler, R.T., & Arora, J.S. (2004). Survey of multi-objective optimization methods for engineering. *Structural and Multidisciplinary Optimization*, 26(6), 369–395. | Normalized weighted sum rationale |
| Frazier, P.I. (2018). A tutorial on Bayesian optimization. *arXiv:1807.02811*. | Plug-in y* estimator, BO framing, independence approximation |
| Koziel, S., & Leifsson, L. (2016). *Surrogate-Based Modeling and Optimization*. Springer. | Single-shot surrogate optimization in engineering design |

---

## 12. References

| Citation | Relevance |
|----------|----------|
| Rasmussen, C.E., & Williams, C.K.I. (2006). *Gaussian Processes for Machine Learning*. MIT Press. | GP theory, kernel selection, ARD, marginal likelihood |
| Kennedy, M.C., & O'Hagan, A. (2000). Predicting the output from a complex computer code when fast approximations are available. *Biometrika*, 87(1), 1–13. | Multi-fidelity additive correction framework |
| Forrester, A.I.J., Sóbester, A., & Keane, A.J. (2007). Multi-fidelity optimization via surrogate modelling. *Proc. Royal Society A*, 463, 3251–3269. | K-O'Hagan validation on engineering problems |
| Álvarez, M.A., Rosasco, L., & Lawrence, N.D. (2012). Kernels for vector-valued functions: A review. *Foundations and Trends in ML*, 4(3), 195–266. | Multi-output GP theory (ICM, LMC) |
| MacKay, D.J.C. (1992). Bayesian interpolation. *Neural Computation*, 4(3), 415–447. | ARD (Automatic Relevance Determination) |
| Neal, R.M. (1996). *Bayesian Learning for Neural Networks*. Springer. | ARD in GP context |
| Snoek, J., Larochelle, H., & Adams, R.P. (2012). Practical Bayesian optimization of machine learning algorithms. *NeurIPS*. | Bayesian optimization with GP |
| Cawley, G.C., & Talbot, N.L.C. (2010). On over-fitting in model selection. *JMLR*, 11. | Scaler-fit-on-train-only protocol |
| Gu, L., et al. (2001). A comparison of metamodeling methods for crashworthiness optimization. *AIAA Paper 2001-1623*. | GP surrogates for crashworthiness |
