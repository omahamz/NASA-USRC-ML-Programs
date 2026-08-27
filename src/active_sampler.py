"""
Active Learning Experimental Design Module.

Proposes the next k high-fidelity simulation points by querying the GP's
posterior uncertainty or acquisition functions grounded in Bayesian
experimental design literature.

Usage
-----
    # USED FOR FINETUNING #
    # Single-purpose batch: reduce CFE model uncertainty only
    python src/active_sampler.py --k 64 --acq uncertainty_cfe --save
    # Dual-purpose batch: reduce CFE uncertainty AND target high-CFE designs
    python src/active_sampler.py --k 128 --acq combined_cfe --w-sea 0.5 --w-cfe 0.5 --save

    # List available acquisition functions
    python src/active_sampler.py --list-acq

    # Default: uncertainty_cfe, k=64, dry-run (no files written)
    python src/active_sampler.py --k 64 --acq uncertainty_cfe --dry-run

    # Full run with save
    python src/active_sampler.py --k 64 --acq uncertainty_cfe --save

    # Weighted composite EI, CFE-biased
    python src/active_sampler.py --k 32 --acq ei_weighted --w-sea 0.2 --w-cfe 0.8 --save

    # UCB with custom beta
    python src/active_sampler.py --k 64 --acq ucb --beta 3.0 --save

    # Very aggressive duplicate removal (testing)
    python src/active_sampler.py --k 32 --acq uncertainty_cfe --dup-threshold 0.99 --dry-run

References
----------
Jones, D.R., Schonlau, M., & Welch, W.J. (1998). Efficient global optimization.
    J. Global Optimization, 13(4), 455-492.
Sacks, J., Welch, W.J., Mitchell, T.J., & Wynn, H.P. (1989). Design and analysis
    of computer experiments. Statistical Science, 4(4), 409-435.
Frazier, P.I. (2018). A tutorial on Bayesian optimization. arXiv:1807.02811.
Svenson, J.D., & Santner, T.J. (2016). Multiobjective optimization of expensive
    black-box functions. J. Global Optimization, 64(2), 297-310.
Srinivas, N., Krause, A., Kakade, S., & Seeger, M. (2012). Information-theoretic
    regret bounds for GP optimization. IEEE Trans. Information Theory, 58(5).
Settles, B. (2012). Active Learning. Morgan & Claypool.
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import norm

from sample import generate_sobol, stratified_subsample
from ml_models.predict import predict_gp
from ml_models.data_loader import LF_CSV, HF_CSV, MODELS_DIR, FEATURE_COLS


# ---------------------------------------------------------------------------
# Parameter domain bounds (used for near-duplicate normalization)
# ---------------------------------------------------------------------------

_PARAM_DOMAINS: dict[str, tuple[float, float]] = {
    "R":  (2.0,  8.8),
    "A":  (30.0, 90.0),
    "CC": (4.0,  22.0),
    "VC": (4.0,  10.0),
}

# ---------------------------------------------------------------------------
# Acquisition function registry
# ---------------------------------------------------------------------------

ACQ_REGISTRY: dict[str, str] = {
    "uncertainty_cfe": (
        "CFE-only uncertainty — maximizes sigma_CFE "
        "(default; recommended for current CFE R²=0.2-0.4)"
    ),
    "uncertainty": (
        "Weighted uncertainty — w_sea*sigma_SEA + w_cfe*sigma_CFE (both outputs)"
    ),
    "ei_sea": (
        "Expected Improvement on SEA only (Jones et al. 1998)"
    ),
    "ei_cfe": (
        "Expected Improvement on CFE only (Jones et al. 1998)"
    ),
    "ei_weighted": (
        "Weighted Composite EI: w_sea*EI_SEA + w_cfe*EI_CFE, "
        "both normalized (Svenson & Santner 2016)"
    ),
    "ucb": (
        "Composite UCB: w_sea*UCB_SEA + w_cfe*UCB_CFE, "
        "with exploration param beta (Srinivas et al. 2012)"
    ),
    "combined_cfe": (
        "Dual-purpose CFE blend: w_sea*sigma_CFE_norm + w_cfe*EI_CFE_norm. "
        "Use for a single large batch that must both reduce CFE uncertainty "
        "(model improvement) and target high-CFE regions (optimization). "
        "Here --w-sea = uncertainty weight, --w-cfe = EI weight. "
        "Recommended: --w-sea 0.5 --w-cfe 0.5 for equal split."
    ),
}


# ---------------------------------------------------------------------------
# Acquisition functions — pure functions, no I/O
# ---------------------------------------------------------------------------

def _ei_single(mu: np.ndarray, sigma: np.ndarray, y_star: float) -> np.ndarray:
    """EI for a single maximization output. sigma==0 -> EI=0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sigma > 0, (mu - y_star) / sigma, 0.0)
        ei = np.where(
            sigma > 0,
            (mu - y_star) * norm.cdf(z) + sigma * norm.pdf(z),
            0.0,
        )
    return np.maximum(ei, 0.0)


def compute_ei_sea(
    sea_mean: np.ndarray,
    sea_std: np.ndarray,
    y_star_sea: float | None = None,
) -> np.ndarray:
    """EI for SEA (maximize). y_star defaults to max(sea_mean) over candidates."""
    y_star = float(np.max(sea_mean)) if y_star_sea is None else y_star_sea
    return _ei_single(sea_mean, sea_std, y_star)


def compute_ei_cfe(
    cfe_mean: np.ndarray,
    cfe_std: np.ndarray,
    y_star_cfe: float | None = None,
) -> np.ndarray:
    """EI for CFE (maximize; closer to 1 is better). y_star defaults to max(cfe_mean)."""
    y_star = float(np.max(cfe_mean)) if y_star_cfe is None else y_star_cfe
    return _ei_single(cfe_mean, cfe_std, y_star)


def compute_wcei(
    ei_sea: np.ndarray,
    ei_cfe: np.ndarray,
    w_sea: float = 0.3,
    w_cfe: float = 0.7,
) -> np.ndarray:
    """
    Weighted Composite EI, each term normalized by its maximum.

    Edge cases:
      - If max(EI_SEA) == 0, that term collapses to 0 (weight transfers to CFE).
      - If both are 0, returns a zero array (caller must handle the fallback).

    Reference: Svenson & Santner (2016).
    """
    max_sea = float(np.max(ei_sea))
    max_cfe = float(np.max(ei_cfe))
    sea_term = (ei_sea / max_sea) * w_sea if max_sea > 0 else np.zeros_like(ei_sea)
    cfe_term = (ei_cfe / max_cfe) * w_cfe if max_cfe > 0 else np.zeros_like(ei_cfe)
    return sea_term + cfe_term


def compute_uncertainty_sampling(
    sea_std: np.ndarray,
    cfe_std: np.ndarray,
    w_sea: float = 0.3,
    w_cfe: float = 0.7,
) -> np.ndarray:
    """
    Weighted uncertainty: w_sea * sigma_SEA_norm + w_cfe * sigma_CFE_norm.

    Each sigma is normalized by its maximum over the candidate pool.
    Falls back gracefully if a maximum is zero.

    Reference: Sacks et al. (1989); Settles (2012).
    """
    max_sea = float(np.max(sea_std))
    max_cfe = float(np.max(cfe_std))
    sea_term = (sea_std / max_sea) * w_sea if max_sea > 0 else np.zeros_like(sea_std)
    cfe_term = (cfe_std / max_cfe) * w_cfe if max_cfe > 0 else np.zeros_like(cfe_std)
    return sea_term + cfe_term


def compute_uncertainty_cfe(cfe_std: np.ndarray) -> np.ndarray:
    """CFE-only uncertainty — returns cfe_std directly (unnormalized; rank-stable)."""
    return cfe_std.copy()


def compute_combined_cfe(
    cfe_mean: np.ndarray,
    cfe_std: np.ndarray,
    w_uncertainty: float = 0.5,
    w_ei: float = 0.5,
) -> np.ndarray:
    """
    Dual-purpose CFE acquisition: normalized blend of uncertainty sampling and EI.

    ACQ(x) = w_uncertainty * [sigma_CFE(x) / max(sigma_CFE)]
           + w_ei          * [EI_CFE(x)    / max(EI_CFE)]

    Why normalize before blending:
      sigma_CFE and EI_CFE live on different scales. Dividing each by its
      pool maximum maps both to [0, 1] so the weights directly control the
      fraction of proposals driven by each criterion — no accidental dominance
      from unit differences (Marler & Arora, 2004).

    What each term contributes:
      Uncertainty term -- finds regions where the GP does not know CFE well.
        Adding data there reduces posterior variance everywhere (Sacks et al. 1989).
      EI term         -- finds regions where CFE is plausibly higher than the
        current best, balancing exploitation with exploration (Jones et al. 1998).

    When to use:
      A single large batch that must serve both purposes: improving the CFE
      surrogate (uncertainty) and identifying high-CFE designs (EI). With
      w_uncertainty = w_ei = 0.5, roughly half the proposals come from each
      criterion after the diversity filter.

    Edge cases:
      max(sigma_CFE) == 0  -> uncertainty term collapses to 0 (all weight on EI).
      max(EI_CFE)    == 0  -> EI term collapses to 0 (all weight on uncertainty).
      Both zero            -> returns zero array; caller falls back to uncertainty_cfe.
    """
    # Uncertainty term: normalize sigma_CFE to [0, 1]
    max_std = float(np.max(cfe_std))
    us_term = (cfe_std / max_std) * w_uncertainty if max_std > 0 else np.zeros_like(cfe_std)

    # EI term: compute EI_CFE, then normalize to [0, 1]
    ei_vals = compute_ei_cfe(cfe_mean, cfe_std)
    max_ei = float(np.max(ei_vals))
    ei_term = (ei_vals / max_ei) * w_ei if max_ei > 0 else np.zeros_like(ei_vals)

    return us_term + ei_term


def compute_ucb(
    sea_mean: np.ndarray,
    sea_std: np.ndarray,
    cfe_mean: np.ndarray,
    cfe_std: np.ndarray,
    beta: float = 2.0,
    w_sea: float = 0.3,
    w_cfe: float = 0.7,
) -> np.ndarray:
    """
    Composite normalized UCB.

    UCB_i(x) = mu_i(x) + beta * sigma_i(x)
    UCBC(x) = w_sea * UCB_SEA_norm + w_cfe * UCB_CFE_norm

    Reference: Srinivas et al. (2012).
    """
    ucb_sea = sea_mean + beta * sea_std
    ucb_cfe = cfe_mean + beta * cfe_std

    max_sea = float(np.max(ucb_sea))
    max_cfe = float(np.max(ucb_cfe))
    sea_term = (ucb_sea / max_sea) * w_sea if max_sea > 0 else np.zeros_like(ucb_sea)
    cfe_term = (ucb_cfe / max_cfe) * w_cfe if max_cfe > 0 else np.zeros_like(ucb_cfe)
    return sea_term + cfe_term


# ---------------------------------------------------------------------------
# Pipeline functions
# ---------------------------------------------------------------------------

def build_candidate_pool(
    pool_size: int = 4096,
    OD: float = 40.0,
    L: float = 50.0,
    G: float = 3.0,
) -> pd.DataFrame:
    """
    Generate a constraint-satisfying Sobol candidate pool.

    Does NOT pass instance= to avoid polluting the project's persistent Sobol state.

    G is forwarded via a minimal domains dict — the only path generate_sobol
    exposes for passing G to apply_constraints (which uses domains["G"] if
    domains is provided, else 0.0). Without this, C2 and C3 would be checked
    with G=0 instead of the project's G=3.0, allowing infeasible proposals at
    higher R values.
    """
    print(f"[active_sampler] Building candidate pool: {pool_size} Sobol points "
          f"(OD={OD}, L={L}, G={G})...")
    # Minimal domains dict — full original bounds, correct G.
    # Passes G into generate_sobol's apply_constraints call without
    # tightening the sampling space (no propagate_domains side-effects).
    _domains = {
        "R":  (2.0,  8.8),
        "A":  (30.0, 90.0),
        "CC": (4,    22),
        "VC": (4,    10),
        "OD": OD,
        "L":  L,
        "G":  G,
    }
    df = generate_sobol(k=pool_size, OD=OD, L=L, domains=_domains)
    print(f"[active_sampler] Pool built: {len(df)} valid candidates.")
    return df


def query_gp(
    candidates_df: pd.DataFrame,
    models_dir: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Query the GP for all candidates, returning posterior means and stds.

    Exits with a clear message if the GP has not been trained yet.
    """
    md = models_dir or MODELS_DIR
    X = candidates_df[FEATURE_COLS].values.astype(float)
    try:
        sea_mean, sea_std, cfe_mean, cfe_std = predict_gp(X, models_dir=md)
    except FileNotFoundError:
        print(
            "[active_sampler] ERROR: GP not trained — run:\n"
            "    python -m src.ml_models.train_gp"
        )
        sys.exit(1)
    print(f"[active_sampler] GP queried: {len(X)} candidates.")
    return sea_mean, sea_std, cfe_mean, cfe_std


def select_top_k_raw(
    candidates_df: pd.DataFrame,
    acq_values: np.ndarray,
    k_raw: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return the top k_raw candidates by acquisition value, sorted descending."""
    k_raw = min(k_raw, len(candidates_df))
    idx = np.argsort(acq_values)[::-1][:k_raw]
    return candidates_df.iloc[idx].reset_index(drop=True), acq_values[idx]


def apply_diversity_filter(top_df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Apply greedy farthest-point (maxmin) diversity filter to select k proposals."""
    if len(top_df) <= k:
        return top_df.reset_index(drop=True)
    return stratified_subsample(top_df, n=k, strategy="maxmin")


def check_near_duplicates(
    proposed_df: pd.DataFrame,
    threshold: float = 0.05,
    lf_csv: str | None = None,
    hf_csv: str | None = None,
) -> pd.DataFrame:
    """
    Remove proposed points that are too close to existing LF or HF data.

    Distances are computed in normalized [0,1]^4 space using domain ranges.
    A point is removed if its minimum distance to any existing point is < threshold.
    threshold=0.05 corresponds to ~5% of the normalized space diagonal.
    """
    lf_path = lf_csv or LF_CSV
    hf_path = hf_csv or HF_CSV

    existing_frames = []
    for path, label in [(lf_path, "LF"), (hf_path, "HF")]:
        if os.path.isfile(path):
            df = pd.read_csv(path)
            cols = [c for c in FEATURE_COLS if c in df.columns]
            existing_frames.append(df[cols])
        else:
            print(f"[active_sampler] Warning: {label} CSV not found at {path}; "
                  "skipping near-duplicate check against it.")

    if not existing_frames:
        return proposed_df

    existing = pd.concat(existing_frames, ignore_index=True)[FEATURE_COLS]

    # Normalize both sets to [0, 1] using full domain ranges
    ranges = np.array([_PARAM_DOMAINS[c][1] - _PARAM_DOMAINS[c][0] for c in FEATURE_COLS])
    mins   = np.array([_PARAM_DOMAINS[c][0]                         for c in FEATURE_COLS])

    prop_norm = (proposed_df[FEATURE_COLS].values - mins) / ranges
    exist_norm = (existing[FEATURE_COLS].values   - mins) / ranges

    keep = []
    removed = 0
    for i, pv in enumerate(prop_norm):
        dists = np.linalg.norm(exist_norm - pv, axis=1)
        if dists.min() < threshold:
            removed += 1
            print(f"[active_sampler] Removed near-duplicate: row {i} "
                  f"(min dist={dists.min():.4f} < threshold={threshold:.4f})")
        else:
            keep.append(i)

    if removed:
        print(f"[active_sampler] Near-duplicate check: removed {removed} point(s).")

    return proposed_df.iloc[keep].reset_index(drop=True)


def save_proposed_points(
    proposed_df: pd.DataFrame,
    acq_name: str,
    output_dir: str | None = None,
) -> str:
    """
    Save proposed points to CSV in data_folder/output/active_sampling/.

    Columns: [R, A, CC, VC] only — directly appendable to HF CSV after simulation.
    Returns the path of the saved file.
    """
    if output_dir is None:
        _src = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(_src, "..", "data_folder", "output", "active_sampling")
    os.makedirs(output_dir, exist_ok=True)

    k = len(proposed_df)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"proposed_{acq_name}_k{k}_{ts}.csv"
    path = os.path.join(output_dir, filename)

    proposed_df[FEATURE_COLS].to_csv(path, index=False)
    print(f"[active_sampler] Proposed points saved -> {path}")
    return path


def generate_plots(
    candidates_df: pd.DataFrame,
    acq_values: np.ndarray,
    proposed_df: pd.DataFrame,
    sea_mean: np.ndarray,
    sea_std: np.ndarray,
    cfe_mean: np.ndarray,
    cfe_std: np.ndarray,
    acq_name: str,
    save: bool = True,
    output_dir: str | None = None,
) -> None:
    """
    Generate three diagnostic plots.

    Plot 1: Acquisition Landscape — marginal acq value vs each input variable.
    Plot 2: Scatter Matrix — spatial coverage of proposals vs existing HF data.
    Plot 3: GP Uncertainty Comparison — mean ± 2σ for SEA and CFE.
    """
    if output_dir is None:
        _src = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(_src, "..", "data_folder", "output", "active_sampling")
    os.makedirs(output_dir, exist_ok=True)

    k = len(proposed_df)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Normalize acquisition values for color and y-axis
    acq_max = float(np.max(acq_values)) if np.max(acq_values) > 0 else 1.0
    acq_norm = acq_values / acq_max

    # ------------------------------------------------------------------ Plot 1
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Acquisition Landscape — {acq_name} (k={k})", fontsize=13)
    axes = axes.flatten()

    for ax, col in zip(axes, FEATURE_COLS):
        x_vals = candidates_df[col].values
        sc = ax.scatter(x_vals, acq_norm, c=cfe_std, cmap="plasma",
                        s=8, alpha=0.6, vmin=0)
        plt.colorbar(sc, ax=ax, label="σ_CFE")
        ax.set_xlabel(col)
        ax.set_ylabel("Normalized acq. value")
        ax.set_title(col)

        # Overlay proposed points as red stars
        prop_x = proposed_df[col].values
        # Map proposed point x-values to acq_norm by nearest candidate index
        cand_x = candidates_df[col].values
        for px in prop_x:
            dists = np.abs(cand_x - px)
            ni = int(np.argmin(dists))
            ax.scatter([px], [acq_norm[ni]], marker="*", color="red", s=120, zorder=5)

    plt.tight_layout()
    p1 = os.path.join(output_dir, f"acq_landscape_{acq_name}_k{k}_{ts}.png")
    if save:
        fig.savefig(p1, dpi=120)
        print(f"[active_sampler] Plot 1 saved -> {p1}")
    plt.close(fig)

    # ------------------------------------------------------------------ Plot 2
    hf_df = None
    if os.path.isfile(HF_CSV):
        hf_df = pd.read_csv(HF_CSV)

    n_cols = len(FEATURE_COLS)
    fig2, axes2 = plt.subplots(n_cols, n_cols, figsize=(10, 10))
    fig2.suptitle(f"Scatter Matrix — Proposals (red) vs HF data (gray)  k={k}", fontsize=12)

    for i, col_i in enumerate(FEATURE_COLS):
        for j, col_j in enumerate(FEATURE_COLS):
            ax = axes2[i, j]
            if i == j:
                ax.hist(proposed_df[col_i].values, bins=10, color="red", alpha=0.7)
                if hf_df is not None and col_i in hf_df.columns:
                    ax.hist(hf_df[col_i].values, bins=10, color="gray", alpha=0.4)
                ax.set_ylabel(col_i if j == 0 else "")
                ax.set_xlabel(col_j if i == n_cols - 1 else "")
            else:
                if hf_df is not None and col_i in hf_df.columns and col_j in hf_df.columns:
                    ax.scatter(hf_df[col_j].values, hf_df[col_i].values,
                               c="gray", s=8, alpha=0.4, label="HF")
                ax.scatter(proposed_df[col_j].values, proposed_df[col_i].values,
                           c="red", s=20, alpha=0.8, label="Proposed")
                ax.set_ylabel(col_i if j == 0 else "")
                ax.set_xlabel(col_j if i == n_cols - 1 else "")

    plt.tight_layout()
    p2 = os.path.join(output_dir, f"scatter_matrix_{acq_name}_k{k}_{ts}.png")
    if save:
        fig2.savefig(p2, dpi=120)
        print(f"[active_sampler] Plot 2 saved -> {p2}")
    plt.close(fig2)

    # ------------------------------------------------------------------ Plot 3
    # Sort candidates by index for a clean x-axis
    n_cand = len(candidates_df)
    x_idx = np.arange(n_cand)

    # Find proposed point indices in the sorted candidate array (by acq value, descending)
    # Use top-k_raw sort order already applied; just mark first k proposed among all cands.
    # For visual clarity, use acq_values as the ordering axis.
    sort_order = np.argsort(acq_values)[::-1]
    proposed_set = set(range(k))  # first k in sort order correspond to proposals (approx)

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    fig3.suptitle(f"GP Uncertainty — {acq_name}  (red lines = proposed)", fontsize=12)

    for ax, mu, sigma, label, color in [
        (axes3[0], sea_mean[sort_order], sea_std[sort_order], "SEA (N·mm/mm³)", "steelblue"),
        (axes3[1], cfe_mean[sort_order], cfe_std[sort_order], "CFE", "darkorange"),
    ]:
        x = np.arange(len(mu))
        ax.fill_between(x, mu - 2 * sigma, mu + 2 * sigma,
                        alpha=0.25, color=color, label="±2σ")
        ax.plot(x, mu, color=color, lw=1.0, label="GP mean")
        for pi in range(min(k, len(x))):
            ax.axvline(pi, color="red", lw=0.6, alpha=0.5)
        ax.set_xlabel("Candidate rank (by acq. value)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)

    plt.tight_layout()
    p3 = os.path.join(output_dir, f"uncertainty_cmp_{acq_name}_k{k}_{ts}.png")
    if save:
        fig3.savefig(p3, dpi=120)
        print(f"[active_sampler] Plot 3 saved -> {p3}")
    plt.close(fig3)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_active_sampler(
    k: int,
    acq: str = "uncertainty_cfe",
    w_sea: float = 0.3,
    w_cfe: float = 0.7,
    beta: float = 2.0,
    pool_size: int = 4096,
    save: bool = False,
    dry_run: bool = False,
    dup_threshold: float = 0.05,
    top_k_multiplier: int = 8,
    models_dir: str | None = None,
    output_dir: str | None = None,
) -> pd.DataFrame:
    """
    Full active learning pipeline: candidate pool → GP → acquisition → diversity → propose.

    Steps
    -----
    1.  Validate k (round up to next power of 2 with warning if needed).
    2.  Build candidate pool (~pool_size Sobol points, constraint-satisfying).
    3.  Query GP → (sea_mean, sea_std, cfe_mean, cfe_std).
    4.  Compute acquisition values via selected function.
    5.  Select top k * top_k_multiplier candidates by acquisition value.
    6.  Apply greedy maxmin diversity filter → k final proposals.
    7.  Remove near-duplicates of existing LF/HF data.
    8.  Print summary table to stdout.
    9.  Save proposed points CSV (unless dry_run=True).
    10. Save diagnostic plots (unless dry_run=True).

    Returns
    -------
    pd.DataFrame with columns [R, A, CC, VC] — the k proposed simulation points.
    """
    if acq not in ACQ_REGISTRY:
        raise ValueError(
            f"Unknown acquisition '{acq}'. "
            f"Available: {list(ACQ_REGISTRY.keys())}"
        )

    if pool_size < k:
        raise ValueError(
            f"pool_size ({pool_size}) must be >= k ({k}). "
            "Increase --pool-size."
        )

    # --- Step 1: validate k ---
    def _is_power_of_2(n: int) -> bool:
        return n > 0 and (n & (n - 1)) == 0

    def _next_power_of_2(n: int) -> int:
        p = 1
        while p < n:
            p <<= 1
        return p

    if not _is_power_of_2(k):
        k_orig = k
        k = _next_power_of_2(k)
        print(f"[active_sampler] Warning: k={k_orig} is not a power of 2. "
              f"Rounding up to k={k}.")

    if abs(w_sea + w_cfe - 1.0) > 1e-6:
        print(f"[active_sampler] Warning: w_sea + w_cfe = {w_sea + w_cfe:.4f} ≠ 1.0. "
              "Proceeding (normalization inside each function handles it).")

    print(f"\n[active_sampler] -- Active Sampling Run --")
    print(f"  k={k}  acq={acq}  w_sea={w_sea}  w_cfe={w_cfe}  "
          f"beta={beta}  pool_size={pool_size}  dry_run={dry_run}\n")

    # --- Step 2: build candidate pool ---
    candidates_df = build_candidate_pool(pool_size=pool_size)

    # --- Step 3: query GP ---
    sea_mean, sea_std, cfe_mean, cfe_std = query_gp(candidates_df, models_dir=models_dir)

    # --- Step 4: compute acquisition values ---
    acq_values: np.ndarray

    if acq == "uncertainty_cfe":
        acq_values = compute_uncertainty_cfe(cfe_std)

    elif acq == "uncertainty":
        max_sea = float(np.max(sea_std))
        max_cfe = float(np.max(cfe_std))
        if max_sea == 0 and max_cfe == 0:
            print("[active_sampler] Warning: all sigma values are zero. "
                  "Falling back to uncertainty_cfe.")
            acq_values = compute_uncertainty_cfe(cfe_std)
        else:
            acq_values = compute_uncertainty_sampling(sea_std, cfe_std, w_sea, w_cfe)

    elif acq == "ei_sea":
        acq_values = compute_ei_sea(sea_mean, sea_std)

    elif acq == "ei_cfe":
        acq_values = compute_ei_cfe(cfe_mean, cfe_std)

    elif acq == "ei_weighted":
        ei_sea = compute_ei_sea(sea_mean, sea_std)
        ei_cfe = compute_ei_cfe(cfe_mean, cfe_std)
        acq_values = compute_wcei(ei_sea, ei_cfe, w_sea, w_cfe)
        if np.max(acq_values) == 0:
            print("[active_sampler] Warning: all EI values are zero (GP may be very certain). "
                  "Falling back to uncertainty_cfe.")
            acq_values = compute_uncertainty_cfe(cfe_std)

    elif acq == "ucb":
        max_ucb = float(np.max(
            (sea_mean + beta * sea_std) + (cfe_mean + beta * cfe_std)
        ))
        if max_ucb == 0:
            print("[active_sampler] Warning: all UCB values are zero. "
                  "Falling back to uncertainty_cfe.")
            acq_values = compute_uncertainty_cfe(cfe_std)
        else:
            acq_values = compute_ucb(sea_mean, sea_std, cfe_mean, cfe_std,
                                     beta=beta, w_sea=w_sea, w_cfe=w_cfe)

    elif acq == "combined_cfe":
        # w_sea repurposed as uncertainty weight; w_cfe repurposed as EI weight.
        acq_values = compute_combined_cfe(
            cfe_mean=cfe_mean,
            cfe_std=cfe_std,
            w_uncertainty=w_sea,
            w_ei=w_cfe,
        )
        if np.max(acq_values) == 0:
            print("[active_sampler] Warning: combined_cfe returned all zeros "
                  "(GP may be fully certain about CFE). Falling back to uncertainty_cfe.")
            acq_values = compute_uncertainty_cfe(cfe_std)

    else:
        # Should never reach here due to earlier validation
        raise ValueError(f"Unhandled acquisition function: {acq}")

    # --- Step 5: select top k_raw ---
    k_raw = min(k * top_k_multiplier, len(candidates_df))
    top_df, top_acq = select_top_k_raw(candidates_df, acq_values, k_raw)
    print(f"[active_sampler] Selected top {len(top_df)} candidates by acq. value.")

    # --- Step 6: diversity filter ---
    proposed_df = apply_diversity_filter(top_df, k)
    # Attach acquisition values for reporting — approximate by nearest in top_df
    acq_col = top_acq[:len(proposed_df)]
    print(f"[active_sampler] Diversity filter: {len(proposed_df)} diverse proposals.")

    # --- Step 7: near-duplicate check ---
    proposed_df = check_near_duplicates(proposed_df, threshold=dup_threshold)
    if len(proposed_df) == 0:
        raise ValueError(
            "All proposed points were removed as near-duplicates. "
            "Lower --dup-threshold or increase --pool-size."
        )

    # Recompute GP predictions for the final proposed set (for the summary table)
    X_prop = proposed_df[FEATURE_COLS].values.astype(float)
    try:
        sea_mu_p, sea_std_p, cfe_mu_p, cfe_std_p = predict_gp(
            X_prop, models_dir=models_dir or MODELS_DIR
        )
    except FileNotFoundError:
        sea_mu_p = sea_std_p = cfe_mu_p = cfe_std_p = np.full(len(proposed_df), float("nan"))

    # --- Step 8: print summary table ---
    print(f"\n[active_sampler] -- Proposed Points (k={len(proposed_df)}, acq={acq}) --")
    header = f"{'R':>6}  {'A':>6}  {'CC':>4}  {'VC':>4}  "
    header += f"{'sea_mu':>9}  {'sea_std':>9}  {'cfe_mu':>9}  {'cfe_std':>9}"
    print(header)
    print("-" * len(header))
    for i in range(len(proposed_df)):
        row = proposed_df.iloc[i]
        print(
            f"{row['R']:6.2f}  {row['A']:6.1f}  {int(row['CC']):4d}  {int(row['VC']):4d}  "
            f"{sea_mu_p[i]:9.4f}  {sea_std_p[i]:9.4f}  "
            f"{cfe_mu_p[i]:9.4f}  {cfe_std_p[i]:9.4f}"
        )
    print()

    # --- Steps 9 & 10: save and plot ---
    if not dry_run:
        if save:
            save_proposed_points(proposed_df, acq_name=acq, output_dir=output_dir)

        # Build full acq_values array aligned with candidates_df for plotting
        generate_plots(
            candidates_df=candidates_df,
            acq_values=acq_values,
            proposed_df=proposed_df,
            sea_mean=sea_mean,
            sea_std=sea_std,
            cfe_mean=cfe_mean,
            cfe_std=cfe_std,
            acq_name=acq,
            save=save,
            output_dir=output_dir,
        )
    else:
        print("[active_sampler] Dry-run mode: no files written.")

    return proposed_df[FEATURE_COLS]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _list_acq() -> None:
    print("\nAvailable acquisition functions:")
    for name, desc in ACQ_REGISTRY.items():
        print(f"  {name:<20}  {desc}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Active Learning Experimental Design — propose the next k "
            "high-fidelity simulation points using GP uncertainty."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/active_sampler.py --list-acq\n"
            "  python src/active_sampler.py --k 64  --acq uncertainty_cfe --dry-run\n"
            "  python src/active_sampler.py --k 64  --acq uncertainty_cfe --save\n"
            "  python src/active_sampler.py --k 128 --acq combined_cfe "
            "--w-sea 0.5 --w-cfe 0.5 --save\n"
            "  python src/active_sampler.py --k 32  --acq ei_weighted "
            "--w-sea 0.2 --w-cfe 0.8 --save\n"
        ),
    )
    parser.add_argument(
        "--k", type=int, default=64,
        help="Number of new simulation points to propose. Rounded up to next power of 2. "
             "Default: 64.",
    )
    parser.add_argument(
        "--acq", type=str, default="uncertainty_cfe",
        choices=list(ACQ_REGISTRY.keys()),
        help="Acquisition function. Default: uncertainty_cfe.",
    )
    parser.add_argument(
        "--w-sea", type=float, default=0.3, dest="w_sea",
        help=(
            "Weight on SEA term for composite acquisition functions "
            "(uncertainty, ei_weighted, ucb). "
            "For combined_cfe: weight on the uncertainty-sampling term. Default: 0.3."
        ),
    )
    parser.add_argument(
        "--w-cfe", type=float, default=0.7, dest="w_cfe",
        help=(
            "Weight on CFE term for composite acquisition functions "
            "(uncertainty, ei_weighted, ucb). "
            "For combined_cfe: weight on the EI term. Default: 0.7."
        ),
    )
    parser.add_argument(
        "--beta", type=float, default=2.0,
        help="Exploration parameter for UCB. beta=2 ~= 95%% confidence. Default: 2.0.",
    )
    parser.add_argument(
        "--pool-size", type=int, default=4096, dest="pool_size",
        help="Number of Sobol candidates to generate. Must be >= k. Default: 4096.",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save proposed points CSV and diagnostic plots to "
             "data_folder/output/active_sampling/.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Print summary table only; do not write any files.",
    )
    parser.add_argument(
        "--dup-threshold", type=float, default=0.05, dest="dup_threshold",
        help="Normalized distance threshold for near-duplicate removal. Default: 0.05.",
    )
    parser.add_argument(
        "--top-k-multiplier", type=int, default=8, dest="top_k_multiplier",
        help="Multiplier for the pre-diversity-filter top-k pool (k * multiplier). "
             "Default: 8.",
    )
    parser.add_argument(
        "--list-acq", action="store_true", dest="list_acq",
        help="Print all available acquisition functions and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_acq:
        _list_acq()
        return

    run_active_sampler(
        k=args.k,
        acq=args.acq,
        w_sea=args.w_sea,
        w_cfe=args.w_cfe,
        beta=args.beta,
        pool_size=args.pool_size,
        save=args.save,
        dry_run=args.dry_run,
        dup_threshold=args.dup_threshold,
        top_k_multiplier=args.top_k_multiplier,
    )


if __name__ == "__main__":
    main()
