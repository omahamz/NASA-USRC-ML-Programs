"""
Surrogate-Based Optimization for crashworthiness thin-walled tube design.

Optimizes (R, A, CC, VC) to maximize SEA while bringing CFE toward 1,
using the trained GP and MLP surrogates as cheap stand-ins for FEA.

Usage
-----
    python src/optimizer.py --model gp --mode exploitation --n 10 --dry-run
    python src/optimizer.py --model gp --mode ei --n 10 --dry-run
    python src/optimizer.py --model gp --mode ucb --beta 3.0 --n 10 --dry-run
    python src/optimizer.py --model gp --mode pareto --n-weights 50 --dry-run
    python src/optimizer.py --model mlp --method de --n 10 --dry-run
    python src/optimizer.py --model mlp --method lbfgsb --n-restarts 50 --n 10 --dry-run
    python src/optimizer.py --model mlp --method sobol --n 10 --dry-run
    python src/optimizer.py --model gp --mode exploitation --n 10 --compare --save
    python src/optimizer.py --list-modes

References
----------
Jones, D.R., et al. (1998). Efficient global optimization. J. Global Opt., 13(4), 455-492.
Srinivas, N., et al. (2012). Information-theoretic regret bounds for GP optimization. IEEE Trans. IT.
Storn, R., & Price, K. (1997). Differential Evolution. J. Global Optimization, 11, 341-359.
Deb, K., et al. (2002). NSGA-II. IEEE Trans. Evolutionary Computation, 6(2), 182-197.
Miettinen, K. (1999). Nonlinear Multiobjective Optimization. Kluwer.
Marler, R.T., & Arora, J.S. (2004). Survey of multi-objective optimization methods. Struct. Multidisc. Opt.
Frazier, P.I. (2018). A tutorial on Bayesian optimization. arXiv:1807.02811.
Liu, D.C., & Nocedal, J. (1989). L-BFGS. Mathematical Programming, 45, 503-528.
Koziel, S., & Leifsson, L. (2016). Surrogate-Based Modeling and Optimization. Springer.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import differential_evolution, minimize, NonlinearConstraint
import torch

from sample import generate_sobol, apply_constraints, stratified_subsample
from ml_models.predict import predict_gp, predict_mlp
from ml_models.data_loader import MODELS_DIR, HF_CSV, FEATURE_COLS, load_scalers
from ml_models.mlp_model import SurrogateNet
from active_sampler import (
    build_candidate_pool,
    query_gp,
    apply_diversity_filter,
    check_near_duplicates,
    select_top_k_raw,
)


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

OBJ_REGISTRY: dict[str, str] = {
    "weighted_sum": "Normalized weighted sum: alpha*SEA_norm + (1-alpha)*CFE (default)",
    "penalty":      "Penalty formula: SEA - k*(1-CFE)",
}

TARGET_REGISTRY: dict[str, str] = {
    "obj": "Combined objective — alpha*SEA_norm + (1-alpha)*CFE (default; respects --alpha and --formula)",
    "sea": "Maximize SEA only  — sets alpha=1.0 (weighted_sum) or k_penalty=0 (penalty)",
    "cfe": "Maximize CFE only  — sets alpha=0.0, forces formula=weighted_sum",
}

GP_MODE_REGISTRY: dict[str, str] = {
    "exploitation": "Rank by GP posterior mean of scalarized objective (pure surrogate maximization)",
    "ei":           "Expected Improvement on scalarized objective — conservative independence approx",
    "ucb":          "Upper Confidence Bound on scalarized objective (Srinivas et al. 2012)",
    "pareto":       "Pareto front via Chebyshev scalarization + non-dominated sorting",
}

MLP_METHOD_REGISTRY: dict[str, str] = {
    "de":     "Differential Evolution — global, constraint-aware, repeated runs (Storn & Price 1997)",
    "lbfgsb": "Multi-start L-BFGS-B — gradient-based via PyTorch autograd, continuous relaxation",
    "sobol":  "Sobol pool evaluation — fastest, no iterative optimization",
}


# ---------------------------------------------------------------------------
# Objective functions (pure, no I/O)
# ---------------------------------------------------------------------------

def compute_scalarized_obj(
    sea: np.ndarray,
    cfe: np.ndarray,
    alpha: float = 0.5,
    sea_ref: float | None = None,
    formula: str = "weighted_sum",
    k_penalty: float | None = None,
) -> tuple[np.ndarray, float]:
    """
    Compute scalarized objective for an array of (sea, cfe) predictions.
    Returns (obj_values, sea_ref_used).
    sea_ref is computed as max(sea) if None.
    """
    sea = np.asarray(sea, dtype=float)
    cfe = np.asarray(cfe, dtype=float)

    if sea_ref is None:
        sea_ref = float(np.max(sea)) if len(sea) > 0 else 1.0
    if sea_ref == 0.0:
        sea_ref = 1.0

    if formula == "weighted_sum":
        obj = alpha * (sea / sea_ref) + (1.0 - alpha) * cfe
    elif formula == "penalty":
        if k_penalty is None:
            k_penalty = float(np.mean(sea))
        obj = sea - k_penalty * (1.0 - cfe)
    else:
        raise ValueError(f"Unknown formula '{formula}'. Choose from: {list(OBJ_REGISTRY)}")

    return obj, float(sea_ref)


def compute_composite_gp_posterior(
    sea_mean: np.ndarray,
    sea_std: np.ndarray,
    cfe_mean: np.ndarray,
    cfe_std: np.ndarray,
    alpha: float,
    sea_ref: float,
    formula: str = "weighted_sum",
    k_penalty: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mu_Obj and sigma_Obj for the scalarized composite output.

    Independence approximation for sigma_Obj: since SEA and CFE GPs are fitted
    independently, their correlation is unknown. We use the independence
    assumption, which gives a conservative (upper-bound) estimate of sigma_Obj
    under positive correlation (Frazier 2018).

    Returns (mu_obj, sigma_obj).
    """
    if sea_ref == 0.0:
        sea_ref = 1.0

    if formula == "weighted_sum":
        mu_obj = alpha * (sea_mean / sea_ref) + (1.0 - alpha) * cfe_mean
        sigma_obj = np.sqrt(
            (alpha / sea_ref) ** 2 * sea_std ** 2
            + (1.0 - alpha) ** 2 * cfe_std ** 2
        )
    elif formula == "penalty":
        if k_penalty is None:
            k_penalty = float(np.mean(sea_mean))
        mu_obj = sea_mean - k_penalty * (1.0 - cfe_mean)
        # Gradient of penalty w.r.t. (sea, cfe) = [1, k_penalty]
        sigma_obj = np.sqrt(sea_std ** 2 + k_penalty ** 2 * cfe_std ** 2)
    else:
        raise ValueError(f"Unknown formula '{formula}'.")

    return mu_obj, sigma_obj


# ---------------------------------------------------------------------------
# Pareto utilities
# ---------------------------------------------------------------------------

def nondominated_sort(sea: np.ndarray, cfe: np.ndarray) -> np.ndarray:
    """
    Boolean mask of non-dominated solutions. O(n²).
    x dominates y iff SEA(x)>=SEA(y) and CFE(x)>=CFE(y), at least one strict.
    """
    n = len(sea)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            if sea[j] >= sea[i] and cfe[j] >= cfe[i]:
                if sea[j] > sea[i] or cfe[j] > cfe[i]:
                    dominated[i] = True
                    break
    return ~dominated


def compute_pareto_front(
    candidates_df: pd.DataFrame,
    sea_pred: np.ndarray,
    cfe_pred: np.ndarray,
    n_weights: int = 50,
) -> pd.DataFrame:
    """
    Chebyshev scalarization sweep over alpha in linspace(0, 1, n_weights).

    For each weight: find argmin Obj_cheby over pool, add to candidate set.
    Then keep the non-dominated subset.

    Obj_cheby(x; alpha) = max(alpha*(1 - SEA_norm(x)), (1-alpha)*(1 - CFE(x)))

    Returns DataFrame with [R, A, CC, VC, SEA_pred, CFE_pred] sorted by SEA.
    """
    sea_ref = float(np.max(sea_pred)) if np.max(sea_pred) > 0 else 1.0
    sea_norm = sea_pred / sea_ref

    weights = np.linspace(0.0, 1.0, n_weights)
    candidate_indices: set[int] = set()

    for alpha in weights:
        cheby_vals = np.maximum(
            alpha * (1.0 - sea_norm),
            (1.0 - alpha) * (1.0 - cfe_pred),
        )
        candidate_indices.add(int(np.argmin(cheby_vals)))

    if not candidate_indices:
        candidate_indices.add(int(np.argmax(sea_pred)))

    idx_list = sorted(candidate_indices)
    sea_cands = sea_pred[idx_list]
    cfe_cands = cfe_pred[idx_list]

    nd_mask = nondominated_sort(sea_cands, cfe_cands)
    final_idx = [idx_list[i] for i in range(len(idx_list)) if nd_mask[i]]

    result = candidates_df.iloc[final_idx].copy().reset_index(drop=True)
    result["SEA_pred"] = sea_pred[final_idx]
    result["CFE_pred"] = cfe_pred[final_idx]
    return (
        result[["R", "A", "CC", "VC", "SEA_pred", "CFE_pred"]]
        .sort_values("SEA_pred")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# GP optimizer
# ---------------------------------------------------------------------------

def _ei_composite(mu_obj: np.ndarray, sigma_obj: np.ndarray) -> np.ndarray:
    """EI on the composite scalarized objective (plug-in y* estimator)."""
    y_star = float(np.max(mu_obj))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sigma_obj > 0, (mu_obj - y_star) / sigma_obj, 0.0)
        ei = np.where(
            sigma_obj > 0,
            (mu_obj - y_star) * norm.cdf(z) + sigma_obj * norm.pdf(z),
            0.0,
        )
    return np.maximum(ei, 0.0)


def optimize_gp(
    n_proposals: int = 10,
    mode: str = "exploitation",
    alpha: float = 0.5,
    beta: float = 2.0,
    formula: str = "weighted_sum",
    k_penalty: float | None = None,
    pool_size: int = 4096,
    top_k_multiplier: int = 8,
    dup_threshold: float = 0.05,
    n_weights: int = 50,
    OD: float = 40.0,
    L: float = 50.0,
    G: float = 3.0,
    models_dir: str | None = None,
) -> pd.DataFrame:
    """
    GP surrogate optimization pipeline.

    Non-pareto modes:
      1. build_candidate_pool
      2. query GP → (sea_mean, sea_std, cfe_mean, cfe_std)
      3. compute composite posterior → (mu_obj, sigma_obj)
      4. dispatch acquisition: exploitation→mu_obj, ei→EI, ucb→UCB
      5. select_top_k_raw → apply_diversity_filter → check_near_duplicates
      6. attach GP predictions + obj_value to result

    Pareto mode:
      1-2. same
      3. compute_pareto_front via Chebyshev sweep
      4. return full Pareto front (n_proposals ignored)

    Returns DataFrame: [R, A, CC, VC, SEA_pred, SEA_std, CFE_pred, CFE_std, obj_value]
    Pool arrays are attached to df.attrs for the orchestrator to use in plots.
    """
    if mode not in GP_MODE_REGISTRY:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(GP_MODE_REGISTRY)}")
    if formula not in OBJ_REGISTRY:
        raise ValueError(f"Unknown formula '{formula}'. Choose from: {list(OBJ_REGISTRY)}")
    if alpha == 0.0:
        print("[optimizer] Note: alpha=0.0 — optimizing CFE only (SEA contribution = 0).")
    elif alpha == 1.0:
        print("[optimizer] Note: alpha=1.0 — optimizing SEA only (CFE contribution = 0).")

    candidates_df = build_candidate_pool(pool_size=pool_size, OD=OD, L=L, G=G)
    sea_mean, sea_std, cfe_mean, cfe_std = query_gp(candidates_df, models_dir=models_dir)

    _, sea_ref = compute_scalarized_obj(
        sea_mean, cfe_mean, alpha=alpha, formula=formula, k_penalty=k_penalty
    )
    if formula == "penalty" and k_penalty is None:
        k_penalty = float(np.mean(sea_mean))

    # ── Pareto path ──────────────────────────────────────────────────────
    if mode == "pareto":
        pareto_df = compute_pareto_front(
            candidates_df, sea_mean, cfe_mean, n_weights=n_weights
        )
        n_front = len(pareto_df)
        print(f"[optimizer] Pareto front: {n_front} non-dominated solutions "
              f"(n_proposals ignored).")
        if n_front == 1:
            print("[optimizer] Warning: Pareto front has only 1 solution — "
                  "pool candidates may all be dominated or equivalent.")
        pareto_df.attrs["pool_sea"] = sea_mean
        pareto_df.attrs["pool_cfe"] = cfe_mean
        pareto_df.attrs["pool_obj"] = sea_mean  # placeholder; pareto uses its own plot
        pareto_df.attrs["candidates_df"] = candidates_df
        return pareto_df

    # ── Acquisition dispatch ─────────────────────────────────────────────
    mu_obj, sigma_obj = compute_composite_gp_posterior(
        sea_mean, sea_std, cfe_mean, cfe_std,
        alpha=alpha, sea_ref=sea_ref, formula=formula, k_penalty=k_penalty,
    )

    if mode == "exploitation":
        acq_values = mu_obj
    elif mode == "ei":
        acq_values = _ei_composite(mu_obj, sigma_obj)
        if np.max(acq_values) == 0.0:
            print("[optimizer] Warning: all EI values are zero (GP may be very certain). "
                  "Falling back to exploitation.")
            acq_values = mu_obj
    else:  # ucb
        acq_values = mu_obj + beta * sigma_obj

    k_raw = min(n_proposals * top_k_multiplier, len(candidates_df))
    top_df, _ = select_top_k_raw(candidates_df, acq_values, k_raw)
    proposals_df = apply_diversity_filter(top_df, n_proposals)
    proposals_df = check_near_duplicates(proposals_df, threshold=dup_threshold)

    if len(proposals_df) == 0:
        raise ValueError(
            "All proposals removed as near-duplicates. "
            "Lower --dup-threshold or increase --pool-size."
        )

    # Attach per-proposal GP predictions
    X_prop = proposals_df[FEATURE_COLS].values.astype(float)
    sea_mu_p, sea_std_p, cfe_mu_p, cfe_std_p = predict_gp(
        X_prop, models_dir=models_dir or MODELS_DIR
    )
    obj_p, _ = compute_scalarized_obj(
        sea_mu_p, cfe_mu_p,
        alpha=alpha, sea_ref=sea_ref, formula=formula, k_penalty=k_penalty,
    )

    proposals_df = proposals_df.copy()
    proposals_df["SEA_pred"]  = sea_mu_p
    proposals_df["SEA_std"]   = sea_std_p
    proposals_df["CFE_pred"]  = cfe_mu_p
    proposals_df["CFE_std"]   = cfe_std_p
    proposals_df["obj_value"] = obj_p

    proposals_df.attrs["pool_sea"]      = sea_mean
    proposals_df.attrs["pool_cfe"]      = cfe_mean
    proposals_df.attrs["pool_obj"]      = acq_values
    proposals_df.attrs["candidates_df"] = candidates_df

    return proposals_df


# ---------------------------------------------------------------------------
# MLP optimizer helpers
# ---------------------------------------------------------------------------

def _build_de_constraints(OD: float, L: float, G: float) -> list:
    """
    Build scipy NonlinearConstraint list enforcing C1, C2, C3.
    Each function returns a value that must be >= 0 (strict feasibility).

    Mirrors the logic in sample.apply_constraints for exact consistency.
    """
    sqrt3 = math.sqrt(3)
    pi    = math.pi

    def c1(x):
        R, _A, CC, _VC = x
        return pi * OD - CC * sqrt3 * R

    def c2(x):
        R, _A, CC, VC = x
        numerator = L - 2.0 * (G + R)
        if numerator <= 0:
            return -1.0
        denom = abs(2.0 * R - (sqrt3 * pi * OD) / (6.0 * CC))
        if denom < 1e-12:
            return -1.0
        return numerator / denom + 1.0 - VC

    def c3(x):
        R, _A, CC, VC = x
        num = L - 2.0 * (R + G)
        if num <= 0:
            return -1.0
        return num / R + 1.0 - VC

    def c_num_pos(x):
        R, _A, _CC, _VC = x
        return L - 2.0 * (G + R)

    return [
        NonlinearConstraint(c1,        1e-6, np.inf),
        NonlinearConstraint(c2,        1e-6, np.inf),
        NonlinearConstraint(c3,        1e-6, np.inf),
        NonlinearConstraint(c_num_pos, 1e-6, np.inf),
    ]


def _mlp_neg_objective(
    x: np.ndarray,
    model: SurrogateNet,
    x_sc,
    y_sc,
    alpha: float,
    sea_ref: float,
    formula: str,
    k_penalty: float | None,
) -> float:
    """Scalar -Obj for scipy DE/Nelder-Mead (minimization interface)."""
    X_std = x_sc.transform(x[np.newaxis, :])
    with torch.no_grad():
        Y_std = model(torch.FloatTensor(X_std)).numpy()
    Y = y_sc.inverse_transform(Y_std)
    sea, cfe = float(Y[0, 0]), float(Y[0, 1])
    obj, _ = compute_scalarized_obj(
        np.array([sea]), np.array([cfe]),
        alpha=alpha, sea_ref=sea_ref, formula=formula, k_penalty=k_penalty,
    )
    return -float(obj[0])


def _mlp_neg_objective_and_grad(
    x: np.ndarray,
    model: SurrogateNet,
    x_sc,
    y_sc,
    alpha: float,
    sea_ref: float,
    formula: str,
    k_penalty: float | None,
) -> tuple[float, np.ndarray]:
    """
    -Obj and its gradient via PyTorch autograd. Used for L-BFGS-B.

    Gradient chain (chain rule):
      ∂Obj/∂X = [α/SEA_ref, 1-α] · diag(σ_y) · J_MLP · diag(1/σ_x)
    where J_MLP = ∂Y_std/∂X_std is computed by autograd.
    """
    model.zero_grad()

    X_std = x_sc.transform(x[np.newaxis, :].astype(float))
    x_tensor = torch.tensor(X_std, dtype=torch.float32, requires_grad=True)

    Y_std_t = model(x_tensor)  # (1, 2)

    # Inverse-scale outputs (affine — preserves autograd graph)
    scale_y = torch.tensor(y_sc.scale_, dtype=torch.float32)
    mean_y  = torch.tensor(y_sc.mean_,  dtype=torch.float32)
    Y_t = Y_std_t * scale_y + mean_y  # (1, 2)

    sea_t = Y_t[0, 0]
    cfe_t = Y_t[0, 1]

    if formula == "weighted_sum":
        _ref = sea_ref if sea_ref != 0.0 else 1.0
        obj_t = alpha * (sea_t / _ref) + (1.0 - alpha) * cfe_t
    else:  # penalty
        _k = k_penalty if k_penalty is not None else sea_t.item()
        obj_t = sea_t - _k * (1.0 - cfe_t)

    neg_obj = -obj_t
    neg_obj.backward()

    # Chain back through input scaler: ∂(-Obj)/∂X = grad_std / σ_x
    grad_std = x_tensor.grad.detach().numpy()[0]
    grad_x   = grad_std / x_sc.scale_

    return float(neg_obj.item()), grad_x.astype(np.float64)


def _mlp_sea_ref(
    model: SurrogateNet,
    x_sc,
    y_sc,
    OD: float,
    L: float,
    G: float,
    n_points: int = 128,
) -> float:
    """Quick Sobol evaluation to estimate sea_ref for MLP normalization."""
    _domains = {
        "R": (2.0, 8.8), "A": (30.0, 90.0),
        "CC": (4, 22),    "VC": (4, 10),
        "OD": OD, "L": L, "G": G,
    }
    pool = generate_sobol(k=n_points, OD=OD, L=L, domains=_domains)
    X_std = x_sc.transform(pool[FEATURE_COLS].values.astype(float))
    with torch.no_grad():
        Y_std = model(torch.FloatTensor(X_std)).numpy()
    Y = y_sc.inverse_transform(Y_std)
    return max(float(np.max(Y[:, 0])), 1.0)


def _load_mlp(models_dir: str | None) -> tuple[SurrogateNet, object, object]:
    """Load the fine-tuned MLP and scalers; exit cleanly if not trained."""
    md = models_dir or MODELS_DIR
    path = os.path.join(md, "mlp_finetuned_hf.pt")
    if not os.path.isfile(path):
        print("[optimizer] ERROR: MLP not trained — run:\n"
              "    python -m src.ml_models.train_mlp")
        sys.exit(1)
    model = SurrogateNet.load(path)
    model.eval()
    x_sc, y_sc = load_scalers(md)
    return model, x_sc, y_sc


_OPT_BOUNDS = {"R": (2.0, 8.8), "A": (30.0, 90.0), "CC": (4.0, 22.0), "VC": (4.0, 10.0)}
_OPT_RANGES = np.array([_OPT_BOUNDS[c][1] - _OPT_BOUNDS[c][0] for c in FEATURE_COLS])
_OPT_MINS   = np.array([_OPT_BOUNDS[c][0]                      for c in FEATURE_COLS])


def _dedup_proposals(df: pd.DataFrame, threshold: float = 0.02) -> pd.DataFrame:
    """Greedy removal of near-duplicate rows using normalized input distances."""
    if len(df) <= 1:
        return df
    vals = df[FEATURE_COLS].values.astype(float)
    norm = (vals - _OPT_MINS) / _OPT_RANGES
    keep: list[int] = [0]
    for i in range(1, len(norm)):
        dists = np.linalg.norm(norm[keep] - norm[i], axis=1)
        if dists.min() >= threshold:
            keep.append(i)
    return df.iloc[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# MLP: Differential Evolution
# ---------------------------------------------------------------------------

def _sobol_top_k(
    n: int,
    alpha: float,
    sea_ref: float,
    formula: str,
    k_penalty: float | None,
    exclude_df: pd.DataFrame | None,
    OD: float,
    L: float,
    G: float,
    models_dir: str,
    pool_size: int = 2048,
) -> pd.DataFrame:
    """
    Return the top-n diverse Sobol candidates by objective, excluding any point
    within normalized distance 0.02 of exclude_df.  Used to supplement DE when
    the converged population lacks diversity.
    """
    candidates_df = build_candidate_pool(pool_size=pool_size, OD=OD, L=L, G=G)
    X = candidates_df[FEATURE_COLS].values.astype(float)
    sea_p, cfe_p = predict_mlp(X, phase="hf", models_dir=models_dir)
    obj_vals, _ = compute_scalarized_obj(
        sea_p, cfe_p, alpha=alpha, sea_ref=sea_ref,
        formula=formula, k_penalty=k_penalty,
    )

    k_raw = min(n * 8, len(candidates_df))
    top_df, _ = select_top_k_raw(candidates_df, obj_vals, k_raw)
    fill_df = apply_diversity_filter(top_df, n)

    # Exclude points too close to already-selected proposals
    if exclude_df is not None and len(exclude_df) > 0:
        excl_norm = (
            (exclude_df[FEATURE_COLS].values.astype(float) - _OPT_MINS) / _OPT_RANGES
        )
        fill_norm = (
            (fill_df[FEATURE_COLS].values.astype(float) - _OPT_MINS) / _OPT_RANGES
        )
        keep = []
        for i, fv in enumerate(fill_norm):
            if np.linalg.norm(excl_norm - fv, axis=1).min() >= 0.02:
                keep.append(i)
        fill_df = fill_df.iloc[keep].reset_index(drop=True)

    if len(fill_df) == 0:
        return pd.DataFrame(columns=["R", "A", "CC", "VC", "SEA_pred", "CFE_pred", "obj_value"])

    X_fill = fill_df[FEATURE_COLS].values.astype(float)
    sea_f, cfe_f = predict_mlp(X_fill, phase="hf", models_dir=models_dir)
    obj_f, _ = compute_scalarized_obj(
        sea_f, cfe_f, alpha=alpha, sea_ref=sea_ref,
        formula=formula, k_penalty=k_penalty,
    )
    fill_df = fill_df.copy()
    fill_df["SEA_pred"]  = sea_f
    fill_df["CFE_pred"]  = cfe_f
    fill_df["obj_value"] = obj_f
    return fill_df[["R", "A", "CC", "VC", "SEA_pred", "CFE_pred", "obj_value"]]


def optimize_mlp_de(
    n_proposals: int = 10,
    alpha: float = 0.5,
    formula: str = "weighted_sum",
    k_penalty: float | None = None,
    pop_size: int = 20,
    max_iter: int = 1000,
    tol: float = 1e-6,
    seed: int = 42,
    OD: float = 40.0,
    L: float = 50.0,
    G: float = 3.0,
    models_dir: str | None = None,
) -> pd.DataFrame:
    """
    Differential Evolution over the MLP surrogate.

    Step 1 — Run a single DE to find the global optimum. Extract the final
    population and dedup to get as many distinct solutions as the converged
    population contains.

    Step 2 — If the converged population is tight (all members near the same
    optimum, which is common when the objective landscape is unimodal), the
    remaining slots are filled with diverse top candidates from a Sobol pool.
    This gives proposal #1 the DE-certified global optimum and fills the rest
    with the best alternatives across different (CC, VC) combinations.
    """
    md = models_dir or MODELS_DIR
    model, x_sc, y_sc = _load_mlp(md)

    sea_ref = _mlp_sea_ref(model, x_sc, y_sc, OD, L, G)
    if formula == "penalty" and k_penalty is None:
        k_penalty = sea_ref

    bounds      = [(2.0, 8.8), (30.0, 90.0), (4.0, 22.0), (4.0, 10.0)]
    integrality = [False, False, True, True]
    constraints = _build_de_constraints(OD, L, G)

    print(f"[optimizer] Running DE (pop_size={pop_size}, "
          f"population={pop_size * 4}, max_iter={max_iter})...")

    try:
        res = differential_evolution(
            _mlp_neg_objective,
            bounds=bounds,
            args=(model, x_sc, y_sc, alpha, sea_ref, formula, k_penalty),
            integrality=integrality,
            constraints=constraints,
            popsize=pop_size,
            maxiter=max_iter,
            tol=tol,
            seed=seed,
            polish=False,
        )
    except Exception as exc:
        raise RuntimeError(f"DE failed: {exc}")

    if not res.success:
        print(f"[optimizer] Warning: DE did not fully converge "
              f"(best obj={-res.fun:.4f}). Using best-found solution.")

    # --- Extract unique solutions from the final population ---
    population = res.population  # (pop_size*4, 4)
    results: list[dict] = []
    for x in population:
        x_eval = x.copy()
        x_eval[2] = int(round(x_eval[2]))
        x_eval[3] = int(round(x_eval[3]))
        row_df = pd.DataFrame([x_eval], columns=FEATURE_COLS)
        if len(apply_constraints(row_df, OD, L, G)) == 0:
            continue
        sea_p, cfe_p = predict_mlp(x_eval[np.newaxis, :], phase="hf", models_dir=md)
        obj_p, _ = compute_scalarized_obj(
            sea_p, cfe_p,
            alpha=alpha, sea_ref=sea_ref, formula=formula, k_penalty=k_penalty,
        )
        results.append({
            "R": float(x_eval[0]), "A": float(x_eval[1]),
            "CC": int(x_eval[2]),  "VC": int(x_eval[3]),
            "SEA_pred":  float(sea_p[0]),
            "CFE_pred":  float(cfe_p[0]),
            "obj_value": float(obj_p[0]),
        })

    if not results:
        raise RuntimeError("No feasible solutions in DE population.")

    de_df = (
        pd.DataFrame(results)
        .sort_values("obj_value", ascending=False)
        .reset_index(drop=True)
    )
    de_df = _dedup_proposals(de_df, threshold=0.02)
    print(f"[optimizer] DE found {len(de_df)} distinct solution(s) from population.")

    # --- Supplement with Sobol if population converged too tightly ---
    n_need = n_proposals - len(de_df)
    if n_need > 0:
        print(f"[optimizer] Supplementing {n_need} slot(s) from Sobol pool "
              "(DE population converged to a tight region).")
        supplement = _sobol_top_k(
            n=n_need,
            alpha=alpha, sea_ref=sea_ref, formula=formula, k_penalty=k_penalty,
            exclude_df=de_df,
            OD=OD, L=L, G=G, models_dir=md,
        )
        proposals_df = pd.concat([de_df, supplement], ignore_index=True)
    else:
        proposals_df = de_df

    proposals_df = _dedup_proposals(proposals_df, threshold=0.02)

    if len(proposals_df) > n_proposals:
        proposals_df = apply_diversity_filter(proposals_df, n_proposals)

    if len(proposals_df) < n_proposals:
        print(f"[optimizer] Note: returning {len(proposals_df)}/{n_proposals} proposals "
              "(design space may have fewer than n distinct high-quality regions).")

    return proposals_df[["R", "A", "CC", "VC", "SEA_pred", "CFE_pred", "obj_value"]]


# ---------------------------------------------------------------------------
# MLP: L-BFGS-B multi-start
# ---------------------------------------------------------------------------

def optimize_mlp_lbfgsb(
    n_proposals: int = 10,
    n_restarts: int = 50,
    alpha: float = 0.5,
    formula: str = "weighted_sum",
    k_penalty: float | None = None,
    OD: float = 40.0,
    L: float = 50.0,
    G: float = 3.0,
    models_dir: str | None = None,
) -> pd.DataFrame:
    """
    Multi-start L-BFGS-B via PyTorch autograd gradients.

    Treats CC and VC as continuous during optimization (continuous relaxation),
    rounds them post-convergence, then discards infeasible solutions.

    Faster per iteration than DE; may miss the global optimum without
    sufficient restarts on multimodal landscapes.
    """
    md = models_dir or MODELS_DIR
    model, x_sc, y_sc = _load_mlp(md)

    sea_ref = _mlp_sea_ref(model, x_sc, y_sc, OD, L, G)
    if formula == "penalty" and k_penalty is None:
        k_penalty = sea_ref

    _domains = {
        "R": (2.0, 8.8), "A": (30.0, 90.0),
        "CC": (4, 22),    "VC": (4, 10),
        "OD": OD, "L": L, "G": G,
    }
    start_pool = generate_sobol(k=max(n_restarts, 128), OD=OD, L=L, domains=_domains)
    starts = start_pool[FEATURE_COLS].values[:n_restarts].astype(float)
    bounds = [(2.0, 8.8), (30.0, 90.0), (4.0, 22.0), (4.0, 10.0)]

    print(f"[optimizer] Running {n_restarts} L-BFGS-B restarts...")

    feasible: list[dict] = []
    for x0 in starts:
        try:
            res = minimize(
                _mlp_neg_objective_and_grad,
                x0=x0.copy(),
                args=(model, x_sc, y_sc, alpha, sea_ref, formula, k_penalty),
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
            )
        except Exception:
            continue

        x_opt = res.x.copy()
        x_opt[2] = int(round(x_opt[2]))
        x_opt[3] = int(round(x_opt[3]))

        row_df = pd.DataFrame([x_opt], columns=FEATURE_COLS)
        if len(apply_constraints(row_df, OD, L, G)) == 0:
            continue

        sea_p, cfe_p = predict_mlp(x_opt[np.newaxis, :], phase="hf", models_dir=md)
        obj_p, _ = compute_scalarized_obj(
            sea_p, cfe_p,
            alpha=alpha, sea_ref=sea_ref, formula=formula, k_penalty=k_penalty,
        )
        feasible.append({
            "R": float(x_opt[0]), "A": float(x_opt[1]),
            "CC": int(x_opt[2]),  "VC": int(x_opt[3]),
            "SEA_pred":  float(sea_p[0]),
            "CFE_pred":  float(cfe_p[0]),
            "obj_value": float(obj_p[0]),
        })

    if not feasible:
        raise RuntimeError(
            "No feasible solutions found by L-BFGS-B. "
            "Try increasing --n-restarts."
        )

    proposals_df = (
        pd.DataFrame(feasible)
        .sort_values("obj_value", ascending=False)
        .reset_index(drop=True)
    )

    if len(proposals_df) < n_proposals:
        print(f"[optimizer] Warning: only {len(proposals_df)}/{n_proposals} "
              "feasible solutions found after rounding. Returning available solutions.")
    else:
        proposals_df = apply_diversity_filter(proposals_df, n_proposals)

    return proposals_df[["R", "A", "CC", "VC", "SEA_pred", "CFE_pred", "obj_value"]].head(n_proposals)


# ---------------------------------------------------------------------------
# MLP: Sobol pool evaluation
# ---------------------------------------------------------------------------

def optimize_mlp_sobol(
    n_proposals: int = 10,
    alpha: float = 0.5,
    formula: str = "weighted_sum",
    k_penalty: float | None = None,
    pool_size: int = 4096,
    top_k_multiplier: int = 8,
    dup_threshold: float = 0.05,
    OD: float = 40.0,
    L: float = 50.0,
    G: float = 3.0,
    models_dir: str | None = None,
) -> pd.DataFrame:
    """
    Evaluate MLP on Sobol pool, rank by objective, diversity-filter to n_proposals.
    Fastest method; no iterative optimization. Equivalent to GP exploitation path.
    """
    md = models_dir or MODELS_DIR
    model_path = os.path.join(md, "mlp_finetuned_hf.pt")
    if not os.path.isfile(model_path):
        print("[optimizer] ERROR: MLP not trained — run:\n"
              "    python -m src.ml_models.train_mlp")
        sys.exit(1)

    candidates_df = build_candidate_pool(pool_size=pool_size, OD=OD, L=L, G=G)
    X = candidates_df[FEATURE_COLS].values.astype(float)

    sea_pred, cfe_pred = predict_mlp(X, phase="hf", models_dir=md)
    obj_values, sea_ref = compute_scalarized_obj(
        sea_pred, cfe_pred, alpha=alpha, formula=formula, k_penalty=k_penalty
    )

    k_raw = min(n_proposals * top_k_multiplier, len(candidates_df))
    top_df, _ = select_top_k_raw(candidates_df, obj_values, k_raw)
    proposals_df = apply_diversity_filter(top_df, n_proposals)
    proposals_df = check_near_duplicates(proposals_df, threshold=dup_threshold)

    if len(proposals_df) == 0:
        raise ValueError(
            "All proposals removed as near-duplicates. "
            "Lower --dup-threshold or increase --pool-size."
        )

    X_prop = proposals_df[FEATURE_COLS].values.astype(float)
    sea_p, cfe_p = predict_mlp(X_prop, phase="hf", models_dir=md)
    obj_p, _ = compute_scalarized_obj(
        sea_p, cfe_p,
        alpha=alpha, sea_ref=sea_ref, formula=formula, k_penalty=k_penalty,
    )

    proposals_df = proposals_df.copy()
    proposals_df["SEA_pred"]  = sea_p
    proposals_df["CFE_pred"]  = cfe_p
    proposals_df["obj_value"] = obj_p

    proposals_df.attrs["pool_sea"]      = sea_pred
    proposals_df.attrs["pool_cfe"]      = cfe_pred
    proposals_df.attrs["pool_obj"]      = obj_values
    proposals_df.attrs["candidates_df"] = candidates_df

    return proposals_df[["R", "A", "CC", "VC", "SEA_pred", "CFE_pred", "obj_value"]]


# ---------------------------------------------------------------------------
# Cross-evaluation
# ---------------------------------------------------------------------------

def cross_evaluate(
    proposals_df: pd.DataFrame,
    primary_model: str,
    models_dir: str | None = None,
) -> pd.DataFrame:
    """
    Evaluate proposals from one model using the other model.

    Adds columns: [<other>_SEA, <other>_CFE, delta_SEA, delta_CFE, agree].
    agree = True when |delta_SEA| < 0.5 N·mm/mm³ and |delta_CFE| < 0.05.

    Disagreement flags regions where GP and MLP diverge — high-information
    simulation candidates that combine optimization and active-learning signals.

    Results are written to a separate comparison CSV when --save is used.
    """
    md = models_dir or MODELS_DIR
    X  = proposals_df[FEATURE_COLS].values.astype(float)

    if primary_model == "gp":
        mlp_path = os.path.join(md, "mlp_finetuned_hf.pt")
        if not os.path.isfile(mlp_path):
            print("[optimizer] Warning: MLP not trained — skipping cross-evaluation.")
            return proposals_df
        try:
            other_sea, other_cfe = predict_mlp(X, phase="hf", models_dir=md)
        except Exception as exc:
            print(f"[optimizer] Warning: MLP cross-eval failed ({exc}). Skipping.")
            return proposals_df
        col_sea, col_cfe = "mlp_SEA", "mlp_CFE"
        extra_cols: dict = {}
    else:
        gp_dir = os.path.join(md, "gp")
        if not os.path.isdir(gp_dir):
            print("[optimizer] Warning: GP not trained — skipping cross-evaluation.")
            return proposals_df
        try:
            gp_sea_mu, gp_sea_std, gp_cfe_mu, gp_cfe_std = predict_gp(X, models_dir=md)
            other_sea, other_cfe = gp_sea_mu, gp_cfe_mu
        except Exception as exc:
            print(f"[optimizer] Warning: GP cross-eval failed ({exc}). Skipping.")
            return proposals_df
        col_sea, col_cfe = "gp_SEA", "gp_CFE"
        extra_cols = {"gp_SEA_std": gp_sea_std, "gp_CFE_std": gp_cfe_std}

    primary_sea = proposals_df["SEA_pred"].values
    primary_cfe = proposals_df["CFE_pred"].values

    out = proposals_df.copy()
    out[col_sea]      = other_sea
    out[col_cfe]      = other_cfe
    for k, v in extra_cols.items():
        out[k] = v
    out["delta_SEA"] = np.abs(primary_sea - other_sea)
    out["delta_CFE"] = np.abs(primary_cfe - other_cfe)
    out["agree"]     = (out["delta_SEA"] < 0.5) & (out["delta_CFE"] < 0.05)

    n_agree = int(out["agree"].sum())
    n_total = len(out)
    print(f"[optimizer] Cross-evaluation: {n_agree}/{n_total} proposals agree "
          f"(|dSEA|<0.5, |dCFE|<0.05).")
    if n_total - n_agree > 0:
        print(f"[optimizer] {n_total - n_agree} disagreeing point(s) -- "
              "prioritize these for simulation (high model-divergence).")

    return out


# ---------------------------------------------------------------------------
# Save utilities
# ---------------------------------------------------------------------------

def _resolve_output_dir(output_dir: str | None) -> str:
    if output_dir:
        return output_dir
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "data_folder", "output", "optimization",
    )


def _save_csv(df: pd.DataFrame, path: str, label: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[optimizer] {label} saved -> {path}")


def save_proposals(
    proposals_df: pd.DataFrame,
    model_name: str,
    mode: str,
    n: int,
    output_dir: str | None = None,
) -> str:
    out_dir = _resolve_output_dir(output_dir)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    path    = os.path.join(out_dir, f"proposals_{model_name}_{mode}_n{n}_{ts}.csv")
    df_save = proposals_df.copy()
    df_save["model"] = model_name
    df_save["mode"]  = mode
    _save_csv(df_save, path, "Proposals")
    return path


def save_pareto(
    pareto_df: pd.DataFrame,
    model_name: str,
    output_dir: str | None = None,
) -> str:
    out_dir = _resolve_output_dir(output_dir)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    path    = os.path.join(out_dir, f"pareto_{model_name}_{ts}.csv")
    _save_csv(pareto_df, path, "Pareto front")
    return path


def save_comparison(
    compare_df: pd.DataFrame,
    model_name: str,
    mode: str,
    output_dir: str | None = None,
) -> str:
    out_dir = _resolve_output_dir(output_dir)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    path    = os.path.join(out_dir, f"comparison_{model_name}_{mode}_{ts}.csv")
    _save_csv(compare_df, path, "Comparison")
    return path


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def generate_optimizer_plots(
    candidates_df: pd.DataFrame,
    obj_values: np.ndarray,
    proposals_df: pd.DataFrame,
    sea_pred: np.ndarray,
    cfe_pred: np.ndarray,
    pareto_df: pd.DataFrame | None,
    model_name: str,
    mode: str,
    save: bool = True,
    output_dir: str | None = None,
) -> None:
    """
    Generate 3 diagnostic plots (4 in pareto mode).

    Plot 1 — Objective Landscape: marginal obj vs each input, color=CFE, red stars=proposals.
    Plot 2 — SEA vs CFE Trade-off: pool (gray), HF data (black), Pareto front (blue),
              proposals (red stars with index labels).
    Plot 3 — Input Scatter Matrix: proposals (red) vs HF data (gray).
    Plot 4 — Pareto Front Curve (pareto mode only): connected SEA-CFE trade-off line,
              each point labeled with (CC, VC).
    """
    out_dir = _resolve_output_dir(output_dir)
    os.makedirs(out_dir, exist_ok=True)
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = f"{model_name}_{mode}"

    hf_df: pd.DataFrame | None = None
    if os.path.isfile(HF_CSV):
        hf_df = pd.read_csv(HF_CSV)

    # ── Plot 1: Objective Landscape ──────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Objective Landscape — {label}", fontsize=13)

    obj_max  = float(np.max(np.abs(obj_values))) if len(obj_values) > 0 else 1.0
    if obj_max == 0.0:
        obj_max = 1.0
    obj_norm = obj_values / obj_max
    cfe_vmin = float(np.percentile(cfe_pred, 5))
    cfe_vmax = float(np.percentile(cfe_pred, 95))

    for ax, col in zip(axes.flatten(), FEATURE_COLS):
        x_vals = candidates_df[col].values
        sc = ax.scatter(x_vals, obj_norm, c=cfe_pred, cmap="plasma",
                        s=8, alpha=0.5, vmin=cfe_vmin, vmax=cfe_vmax)
        plt.colorbar(sc, ax=ax, label="CFE pred")
        ax.set_xlabel(col)
        ax.set_ylabel("Normalized objective")
        ax.set_title(col)

        prop_x  = proposals_df[col].values
        cand_x  = candidates_df[col].values
        for px in prop_x:
            ni = int(np.argmin(np.abs(cand_x - px)))
            ax.scatter([px], [obj_norm[ni]], marker="*", color="red", s=120, zorder=5)

    plt.tight_layout()
    p1 = os.path.join(out_dir, f"obj_landscape_{label}_{ts}.png")
    if save:
        fig.savefig(p1, dpi=120)
        print(f"[optimizer] Plot 1 saved -> {p1}")
    plt.close(fig)

    # ── Plot 2: SEA vs CFE Trade-off Scatter ────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    fig2.suptitle(f"SEA vs CFE Trade-off — {label}", fontsize=12)

    ax2.scatter(sea_pred, cfe_pred, c="lightgray", s=10, alpha=0.5,
                label="Sobol pool", zorder=1)

    if hf_df is not None and "SEA" in hf_df.columns and "CFE" in hf_df.columns:
        ax2.scatter(hf_df["SEA"].values, hf_df["CFE"].values,
                    c="black", s=20, alpha=0.6, zorder=2, label="HF data")

    nd_mask    = nondominated_sort(sea_pred, cfe_pred)
    pf_sea     = sea_pred[nd_mask]
    pf_cfe     = cfe_pred[nd_mask]
    sort_order = np.argsort(pf_sea)
    ax2.plot(pf_sea[sort_order], pf_cfe[sort_order],
             "b-", lw=1.5, alpha=0.7, label="Pool Pareto front", zorder=3)

    prop_sea = proposals_df["SEA_pred"].values
    prop_cfe = proposals_df["CFE_pred"].values
    ax2.scatter(prop_sea, prop_cfe, marker="*", c="red", s=150,
                zorder=5, label="Proposals")
    for i, (s, c) in enumerate(zip(prop_sea, prop_cfe)):
        ax2.annotate(str(i + 1), (s, c), textcoords="offset points",
                     xytext=(4, 4), fontsize=7)

    ax2.set_xlabel("SEA (N·mm/mm³)")
    ax2.set_ylabel("CFE")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    p2 = os.path.join(out_dir, f"sea_cfe_scatter_{label}_{ts}.png")
    if save:
        fig2.savefig(p2, dpi=120)
        print(f"[optimizer] Plot 2 saved -> {p2}")
    plt.close(fig2)

    # ── Plot 3: Input Scatter Matrix ─────────────────────────────────────
    n_cols = len(FEATURE_COLS)
    fig3, axes3 = plt.subplots(n_cols, n_cols, figsize=(10, 10))
    fig3.suptitle(
        f"Input Scatter Matrix — Proposals (red) vs HF data (gray)  [{label}]",
        fontsize=11,
    )
    for i, col_i in enumerate(FEATURE_COLS):
        for j, col_j in enumerate(FEATURE_COLS):
            ax = axes3[i, j]
            if i == j:
                ax.hist(proposals_df[col_i].values, bins=10, color="red", alpha=0.7)
                if hf_df is not None and col_i in hf_df.columns:
                    ax.hist(hf_df[col_i].values, bins=10, color="gray", alpha=0.4)
            else:
                if hf_df is not None and col_i in hf_df.columns and col_j in hf_df.columns:
                    ax.scatter(hf_df[col_j].values, hf_df[col_i].values,
                               c="gray", s=8, alpha=0.4)
                ax.scatter(proposals_df[col_j].values, proposals_df[col_i].values,
                           c="red", s=20, alpha=0.8)
            ax.set_ylabel(col_i if j == 0 else "")
            ax.set_xlabel(col_j if i == n_cols - 1 else "")

    plt.tight_layout()
    p3 = os.path.join(out_dir, f"scatter_matrix_opt_{label}_{ts}.png")
    if save:
        fig3.savefig(p3, dpi=120)
        print(f"[optimizer] Plot 3 saved -> {p3}")
    plt.close(fig3)

    # ── Plot 4: Pareto Front Curve (pareto mode only) ────────────────────
    if pareto_df is not None and len(pareto_df) > 0:
        fig4, ax4 = plt.subplots(figsize=(9, 6))
        fig4.suptitle(f"Pareto Front — {model_name}", fontsize=12)

        if hf_df is not None and "SEA" in hf_df.columns and "CFE" in hf_df.columns:
            ax4.scatter(hf_df["SEA"].values, hf_df["CFE"].values,
                        c="lightgray", s=15, alpha=0.5, label="HF data", zorder=1)

        pf_sea = pareto_df["SEA_pred"].values
        pf_cfe = pareto_df["CFE_pred"].values
        ax4.plot(pf_sea, pf_cfe, "b-o", lw=1.5, ms=6, label="Pareto front", zorder=3)

        for _, row in pareto_df.iterrows():
            ax4.annotate(
                f"({int(row['CC'])},{int(row['VC'])})",
                (row["SEA_pred"], row["CFE_pred"]),
                textcoords="offset points", xytext=(4, 4), fontsize=7,
            )

        ax4.set_xlabel("SEA (N·mm/mm³)")
        ax4.set_ylabel("CFE")
        ax4.legend(fontsize=8)
        plt.tight_layout()
        p4 = os.path.join(out_dir, f"pareto_front_{model_name}_{ts}.png")
        if save:
            fig4.savefig(p4, dpi=120)
            print(f"[optimizer] Plot 4 saved -> {p4}")
        plt.close(fig4)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(
    proposals_df: pd.DataFrame,
    model_name: str,
    mode: str,
    has_compare: bool,
) -> None:
    if model_name == "gp":
        has_std = "SEA_std" in proposals_df.columns
        hdr = (f"{'#':>3}  {'R':>6}  {'A':>6}  {'CC':>4}  {'VC':>4}  "
               f"{'obj':>9}  {'sea_mu':>9}  {'sea_std':>9}  "
               f"{'cfe_mu':>8}  {'cfe_std':>8}")
        if has_compare and "mlp_SEA" in proposals_df.columns:
            hdr += f"  {'mlp_sea':>9}  {'mlp_cfe':>8}  {'agree':>5}"
        print(f"\n[optimizer] -- GP Proposals (mode={mode}, n={len(proposals_df)}) --")
        print(hdr)
        print("-" * len(hdr))
        for idx, row in proposals_df.reset_index(drop=True).iterrows():
            line = (
                f"{idx + 1:>3}  {row['R']:6.3f}  {row['A']:6.1f}  "
                f"{int(row['CC']):4d}  {int(row['VC']):4d}  "
                f"{row.get('obj_value', float('nan')):9.5f}  "
                f"{row.get('SEA_pred', float('nan')):9.4f}  "
                f"{row.get('SEA_std',  float('nan')):9.4f}  "
                f"{row.get('CFE_pred', float('nan')):8.4f}  "
                f"{row.get('CFE_std',  float('nan')):8.4f}"
            )
            if has_compare and "mlp_SEA" in proposals_df.columns:
                line += (
                    f"  {row.get('mlp_SEA', float('nan')):9.4f}  "
                    f"{row.get('mlp_CFE', float('nan')):8.4f}  "
                    f"{'Y' if row.get('agree', False) else 'N':>5}"
                )
            print(line)
    else:
        hdr = (f"{'#':>3}  {'R':>6}  {'A':>6}  {'CC':>4}  {'VC':>4}  "
               f"{'obj':>9}  {'sea':>9}  {'cfe':>8}")
        if has_compare and "gp_SEA" in proposals_df.columns:
            hdr += f"  {'gp_sea_mu':>10}  {'gp_sea_std':>10}  {'gp_cfe_mu':>10}  {'agree':>5}"
        print(f"\n[optimizer] -- MLP Proposals (method={mode}, n={len(proposals_df)}) --")
        print(hdr)
        print("-" * len(hdr))
        for idx, row in proposals_df.reset_index(drop=True).iterrows():
            line = (
                f"{idx + 1:>3}  {row['R']:6.3f}  {row['A']:6.1f}  "
                f"{int(row['CC']):4d}  {int(row['VC']):4d}  "
                f"{row.get('obj_value', float('nan')):9.5f}  "
                f"{row.get('SEA_pred', float('nan')):9.4f}  "
                f"{row.get('CFE_pred', float('nan')):8.4f}"
            )
            if has_compare and "gp_SEA" in proposals_df.columns:
                line += (
                    f"  {row.get('gp_SEA',     float('nan')):10.4f}  "
                    f"{row.get('gp_SEA_std', float('nan')):10.4f}  "
                    f"{row.get('gp_CFE',     float('nan')):10.4f}  "
                    f"{'Y' if row.get('agree', False) else 'N':>5}"
                )
            print(line)
    print()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_optimizer(
    model: str = "gp",
    mode: str = "exploitation",
    method: str = "de",
    n_proposals: int = 10,
    alpha: float = 0.5,
    beta: float = 2.0,
    formula: str = "weighted_sum",
    k_penalty: float | None = None,
    target: str = "obj",
    pool_size: int = 4096,
    top_k_multiplier: int = 8,
    pop_size: int = 20,
    max_iter: int = 1000,
    n_restarts: int = 50,
    n_weights: int = 50,
    dup_threshold: float = 0.05,
    seed: int = 42,
    compare: bool = False,
    save: bool = False,
    dry_run: bool = False,
    OD: float = 40.0,
    L: float = 50.0,
    G: float = 3.0,
    models_dir: str | None = None,
    output_dir: str | None = None,
) -> pd.DataFrame:
    """Orchestrates the full optimization pipeline for a single model."""
    if formula not in OBJ_REGISTRY:
        raise ValueError(f"Unknown formula '{formula}'. Choose from: {list(OBJ_REGISTRY)}")
    if target not in TARGET_REGISTRY:
        raise ValueError(f"Unknown target '{target}'. Choose from: {list(TARGET_REGISTRY)}")

    # Translate --target shortcut into alpha / formula overrides.
    if target == "sea":
        if formula == "penalty":
            k_penalty = 0.0
            print("[optimizer] target=sea + penalty: setting k_penalty=0 (obj = SEA).")
        else:
            alpha = 1.0
            print("[optimizer] target=sea: setting alpha=1.0 (obj = SEA_norm).")
    elif target == "cfe":
        alpha   = 0.0
        formula = "weighted_sum"
        print("[optimizer] target=cfe: setting alpha=0.0, formula=weighted_sum (obj = CFE).")

    is_pareto = (model == "gp" and mode == "pareto")

    print(f"\n[optimizer] -- Optimization Run --")
    print(f"  model={model}  "
          + (f"mode={mode}" if model == "gp" else f"method={method}")
          + f"  target={target}  n={n_proposals}  alpha={alpha}  formula={formula}"
          + (f"  dry_run=True" if dry_run else ""))
    print()

    # ── GP path ──────────────────────────────────────────────────────────
    if model == "gp":
        if mode not in GP_MODE_REGISTRY:
            raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(GP_MODE_REGISTRY)}")

        proposals_df = optimize_gp(
            n_proposals=n_proposals,
            mode=mode,
            alpha=alpha,
            beta=beta,
            formula=formula,
            k_penalty=k_penalty,
            pool_size=pool_size,
            top_k_multiplier=top_k_multiplier,
            dup_threshold=dup_threshold,
            n_weights=n_weights,
            OD=OD, L=L, G=G,
            models_dir=models_dir,
        )

        cands    = proposals_df.attrs.get("candidates_df", proposals_df)
        sea_pool = proposals_df.attrs.get("pool_sea", proposals_df["SEA_pred"].values)
        cfe_pool = proposals_df.attrs.get("pool_cfe", proposals_df["CFE_pred"].values)
        obj_pool = proposals_df.attrs.get("pool_obj",
                                          proposals_df.get("obj_value",
                                                           proposals_df["SEA_pred"]).values)

        pareto_out = proposals_df if is_pareto else None

        if compare and not is_pareto:
            proposals_df = cross_evaluate(
                proposals_df, primary_model="gp", models_dir=models_dir
            )

        _print_summary(proposals_df, "gp", mode, compare and not is_pareto)

        if not dry_run:
            if is_pareto:
                if save:
                    save_pareto(proposals_df, model_name="gp", output_dir=output_dir)
            else:
                if save:
                    save_proposals(proposals_df, "gp", mode, n_proposals, output_dir)
                    if compare and "mlp_SEA" in proposals_df.columns:
                        save_comparison(proposals_df, "gp", mode, output_dir)

            generate_optimizer_plots(
                candidates_df=cands,
                obj_values=obj_pool,
                proposals_df=proposals_df,
                sea_pred=sea_pool,
                cfe_pred=cfe_pool,
                pareto_df=pareto_out,
                model_name="gp",
                mode=mode,
                save=save,
                output_dir=output_dir,
            )
        else:
            print("[optimizer] Dry-run mode: no files written.")

        return proposals_df

    # ── MLP path ─────────────────────────────────────────────────────────
    if method not in MLP_METHOD_REGISTRY:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(MLP_METHOD_REGISTRY)}")

    md = models_dir or MODELS_DIR
    common = dict(
        n_proposals=n_proposals,
        alpha=alpha,
        formula=formula,
        k_penalty=k_penalty,
        OD=OD, L=L, G=G,
        models_dir=models_dir,
    )

    if method == "de":
        proposals_df = optimize_mlp_de(
            pop_size=pop_size, max_iter=max_iter, seed=seed, **common
        )
    elif method == "lbfgsb":
        proposals_df = optimize_mlp_lbfgsb(n_restarts=n_restarts, **common)
    else:
        proposals_df = optimize_mlp_sobol(
            pool_size=pool_size,
            top_k_multiplier=top_k_multiplier,
            dup_threshold=dup_threshold,
            **common,
        )

    # For de/lbfgsb: build a pool for plotting (sobol already has it in attrs)
    cands    = proposals_df.attrs.get("candidates_df", None)
    sea_pool = proposals_df.attrs.get("pool_sea", None)
    cfe_pool = proposals_df.attrs.get("pool_cfe", None)
    obj_pool = proposals_df.attrs.get("pool_obj", None)

    if cands is None:
        cands = build_candidate_pool(pool_size=min(pool_size, 1024), OD=OD, L=L, G=G)
        X_plot = cands[FEATURE_COLS].values.astype(float)
        sea_pool, cfe_pool = predict_mlp(X_plot, phase="hf", models_dir=md)
        obj_pool, _ = compute_scalarized_obj(
            sea_pool, cfe_pool, alpha=alpha, formula=formula, k_penalty=k_penalty
        )

    if compare:
        proposals_df = cross_evaluate(
            proposals_df, primary_model="mlp", models_dir=models_dir
        )

    _print_summary(proposals_df, "mlp", method, compare)

    if not dry_run:
        if save:
            save_proposals(proposals_df, "mlp", method, n_proposals, output_dir)
            if compare and "gp_SEA" in proposals_df.columns:
                save_comparison(proposals_df, "mlp", method, output_dir)

        generate_optimizer_plots(
            candidates_df=cands,
            obj_values=obj_pool,
            proposals_df=proposals_df,
            sea_pred=sea_pool,
            cfe_pred=cfe_pool,
            pareto_df=None,
            model_name="mlp",
            mode=method,
            save=save,
            output_dir=output_dir,
        )
    else:
        print("[optimizer] Dry-run mode: no files written.")

    return proposals_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _list_modes() -> None:
    print("\nOptimization targets (--target <target>):")
    for name, desc in TARGET_REGISTRY.items():
        print(f"  {name:<6}  {desc}")
    print("\nGP modes (--model gp --mode <mode>):")
    for name, desc in GP_MODE_REGISTRY.items():
        print(f"  {name:<14}  {desc}")
    print("\nMLP methods (--model mlp --method <method>):")
    for name, desc in MLP_METHOD_REGISTRY.items():
        print(f"  {name:<10}  {desc}")
    print("\nObjective formulas (--formula <formula>):")
    for name, desc in OBJ_REGISTRY.items():
        print(f"  {name:<14}  {desc}")
    print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Surrogate-Based Optimization — find tube designs that maximize SEA "
            "while bringing CFE toward 1, using GP or MLP surrogates."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/optimizer.py --model gp --mode exploitation --n 10 --dry-run\n"
            "  python src/optimizer.py --model gp --mode ei           --n 10 --dry-run\n"
            "  python src/optimizer.py --model gp --mode ucb --beta 3.0 --n 10 --dry-run\n"
            "  python src/optimizer.py --model gp --mode pareto --n-weights 50 --dry-run\n"
            "  python src/optimizer.py --model mlp --method de      --n 10 --dry-run\n"
            "  python src/optimizer.py --model mlp --method lbfgsb  --n 10 --dry-run\n"
            "  python src/optimizer.py --model mlp --method sobol   --n 10 --dry-run\n"
            "  python src/optimizer.py --model gp --mode exploitation --n 10 --compare --save\n"
        ),
    )
    parser.add_argument("--model",  type=str, default="gp",
                        choices=["gp", "mlp"],
                        help="Surrogate model. Default: gp.")
    parser.add_argument("--mode",   type=str, default="exploitation",
                        choices=list(GP_MODE_REGISTRY),
                        help="GP acquisition mode. Default: exploitation.")
    parser.add_argument("--method", type=str, default="de",
                        choices=list(MLP_METHOD_REGISTRY),
                        help="MLP optimization method. Default: de.")
    parser.add_argument("--n",      type=int, default=10, dest="n_proposals",
                        help="Number of proposals. Default: 10.")
    parser.add_argument("--target", type=str, default="obj",
                        choices=list(TARGET_REGISTRY),
                        help=(
                            "Output to optimize. "
                            "'obj' = combined (respects --alpha/--formula), "
                            "'sea' = SEA only, "
                            "'cfe' = CFE only. "
                            "Default: obj."
                        ))
    parser.add_argument("--alpha",  type=float, default=0.5,
                        help="Scalarization weight alpha (alpha*SEA_norm + (1-alpha)*CFE). "
                             "Ignored when --target is 'sea' or 'cfe'. Default: 0.5.")
    parser.add_argument("--beta",   type=float, default=2.0,
                        help="UCB exploration parameter beta. Default: 2.0.")
    parser.add_argument("--formula", type=str, default="weighted_sum",
                        choices=list(OBJ_REGISTRY),
                        help="Objective formula. Default: weighted_sum.")
    parser.add_argument("--k-penalty", type=float, default=None, dest="k_penalty",
                        help="Penalty coefficient k (formula=penalty only). "
                             "Default: mean(SEA) over pool.")
    parser.add_argument("--pool-size", type=int, default=4096, dest="pool_size",
                        help="Sobol candidate pool size. Default: 4096.")
    parser.add_argument("--pop-size",  type=int, default=20,   dest="pop_size",
                        help="DE population multiplier (actual = pop_size × 4). Default: 20.")
    parser.add_argument("--max-iter",  type=int, default=1000, dest="max_iter",
                        help="DE maximum iterations per run. Default: 1000.")
    parser.add_argument("--n-restarts", type=int, default=50, dest="n_restarts",
                        help="L-BFGS-B random restarts. Default: 50.")
    parser.add_argument("--n-weights",  type=int, default=50, dest="n_weights",
                        help="Chebyshev weight sweep count (pareto mode). Default: 50.")
    parser.add_argument("--dup-threshold", type=float, default=0.05, dest="dup_threshold",
                        help="Near-duplicate removal threshold (normalized). Default: 0.05.")
    parser.add_argument("--seed",    type=int, default=42,
                        help="Random seed. Default: 42.")
    parser.add_argument("--compare", action="store_true",
                        help="Cross-evaluate proposals with the other model. "
                             "With --save, writes a separate comparison CSV.")
    parser.add_argument("--save",    action="store_true",
                        help="Save proposals CSV and plots to "
                             "data_folder/output/optimization/.")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="Print summary only; write no files.")
    parser.add_argument("--list-modes", action="store_true", dest="list_modes",
                        help="Print all available modes/methods/formulas and exit.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.list_modes:
        _list_modes()
        return

    run_optimizer(
        model=args.model,
        mode=args.mode,
        method=args.method,
        n_proposals=args.n_proposals,
        alpha=args.alpha,
        beta=args.beta,
        formula=args.formula,
        k_penalty=args.k_penalty,
        target=args.target,
        pool_size=args.pool_size,
        pop_size=args.pop_size,
        max_iter=args.max_iter,
        n_restarts=args.n_restarts,
        n_weights=args.n_weights,
        dup_threshold=args.dup_threshold,
        seed=args.seed,
        compare=args.compare,
        save=args.save,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
