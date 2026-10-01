"""
GP training pipeline: LF fitting (Phase 1) and HF correction fitting (Phase 2).

Usage
-----
From the project root:

    python -m src.ml_models.train_gp                   # run both phases
    python -m src.ml_models.train_gp --phase lf        # LF GPs only
    python -m src.ml_models.train_gp --phase correction # correction GPs only
    python -m src.ml_models.train_gp --restarts 10     # more optimizer restarts
    python -m src.ml_models.train_gp --seed 7
    python -m src.ml_models.train_gp --models-dir models_6in           # keep other artifacts untouched
    python -m src.ml_models.train_gp --lf-csv a_PD.csv b_PD.csv --hf-csv c_PD.csv    # several configs

Inputs are [R, A, CC, VC, T, N] (one ARD length scale each).  CSVs without N / T columns
(legacy campaign) are read as N=6, T=0.  A models dir that already holds a model with a
different input width is never overwritten unless --force is given.

Saved artifacts
---------------
<models-dir>/   (default: models/)
  gp/
    gp_sea_lf.pkl          Low-fidelity GP for SEA
    gp_cfe_lf.pkl          Low-fidelity GP for CFE
    gp_sea_delta.pkl       Correction GP for SEA
    gp_cfe_delta.pkl       Correction GP for CFE
    gp_metadata.json       Kernel hyperparameters, fit status, timestamps
  scalers.pkl              Shared with MLP (same LF-train StandardScaler)
  plots/
    Parity_GP_LF_only.png  LF-only parity plot on HF test set
    Parity_GP_TL.png       Transfer-learning parity plot on HF test set
    GP_uncertainty_SEA.png Posterior mean ± 2σ vs actuals for SEA
    GP_uncertainty_CFE.png Posterior mean ± 2σ vs actuals for CFE
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from .data_loader import FEATURE_COLS, MODELS_DIR, guard_model_dir, load_data, save_scalers
from .evaluate import (
    compute_metrics,
    compute_metrics_by_config,
    gp_uncertainty_plot,
    parity_plot,
    print_metrics_by_config,
    print_metrics_table,
    save_metrics_json,
    set_plots_dir,
)
from .gp_model import MultiFidelityGP

GP_DIR = os.path.join(MODELS_DIR, "gp")


def _by_config(splits, Y_true: np.ndarray, Y_pred: np.ndarray) -> dict:
    """Per-(N, T) HF-test metrics; empty unless the test set holds more than one configuration."""
    labels = getattr(splits, "config_hf_test", None)
    if labels is None or len(np.unique(labels)) < 2:
        return {}
    return compute_metrics_by_config(Y_true, Y_pred, labels)


# ---------------------------------------------------------------------------
# Phase wrappers
# ---------------------------------------------------------------------------

def run_lf_fit(
    splits,
    gp: MultiFidelityGP | None = None,
    gp_dir: str = GP_DIR,
) -> MultiFidelityGP:
    """
    Fit the two LF GPs (SEA and CFE) on the shell training data.

    The full LF dataset is used (train + val splits) — unlike the MLP, GPs do not
    require a separate validation set during fitting; regularization is handled
    implicitly through the marginal likelihood optimization.

    Note: GP fitting on n points requires solving an O(n³) linear system.
    Expect 30 seconds–3 minutes for n≈750; the cost grows quickly when several
    (N, T) configurations are pooled into the LF set.
    """
    print("\n" + "=" * 60)
    print("PHASE 1 - Low-Fidelity GP Fitting")
    print("=" * 60)

    if gp is None:
        gp = MultiFidelityGP(
            n_restarts=splits._n_restarts if hasattr(splits, "_n_restarts") else 5,
            n_features=splits.X_lf_train.shape[1],
        )

    # Use the combined LF train + val data for GP fitting (GPs self-regularize)
    X_lf = np.vstack([splits.X_lf_train, splits.X_lf_val])
    Y_lf = np.vstack([splits.Y_lf_train, splits.Y_lf_val])
    print(f"  Fitting on {len(X_lf)} LF (shell) samples ...")
    gp.fit_lf(X_lf, Y_lf)

    # Evaluate LF-only on HF test set (baseline)
    sea_mu, cfe_mu = gp.predict(splits.X_hf_test, return_std=False)
    Y_pred_std = np.column_stack([sea_mu, cfe_mu])
    Y_pred = splits.y_scaler.inverse_transform(Y_pred_std)
    Y_true = splits.y_scaler.inverse_transform(splits.Y_hf_test)
    m = compute_metrics(Y_true, Y_pred)
    print(f"\n  [LF GP] HF test:  SEA R2={m['SEA']['R2']:.4f}  CFE R2={m['CFE']['R2']:.4f}")
    print_metrics_by_config({"GP (LF only)": _by_config(splits, Y_true, Y_pred)})

    gp.save(gp_dir)
    return gp


def run_correction_fit(
    splits,
    gp: MultiFidelityGP | None = None,
    gp_dir: str = GP_DIR,
) -> MultiFidelityGP:
    """
    Fit the correction GPs on the HF (solid) fine-tune split.

    For each output, the correction GP models:
        delta(x) = Y_HF(x) - GP_LF.predict(x)

    This fidelity residual captures the systematic bias introduced by using
    shell elements instead of solid elements in the FEA.  The Kennedy-O'Hagan
    framework (2000) guarantees that the combined prediction is at least as
    accurate as either fidelity alone.
    """
    print("\n" + "=" * 60)
    print("PHASE 2 - High-Fidelity Correction Fitting")
    print("=" * 60)

    if gp is None:
        gp = MultiFidelityGP.load(gp_dir)

    print(f"  Computing fidelity deltas for {len(splits.X_hf_finetune)} HF (solid) points ...")
    gp.fit_hf_correction(splits.X_hf_finetune, splits.Y_hf_finetune)

    # Evaluate full multi-fidelity model on HF test set
    sea_mu, sea_std, cfe_mu, cfe_std = gp.predict(
        splits.X_hf_test, return_std=True, use_correction=True
    )
    Y_pred_std = np.column_stack([sea_mu, cfe_mu])
    Y_pred     = splits.y_scaler.inverse_transform(Y_pred_std)
    Y_true     = splits.y_scaler.inverse_transform(splits.Y_hf_test)
    m = compute_metrics(Y_true, Y_pred)
    print(f"\n  [GP TL] HF test:  SEA R2={m['SEA']['R2']:.4f}  CFE R2={m['CFE']['R2']:.4f}")
    print_metrics_by_config({"GP (TL)": _by_config(splits, Y_true, Y_pred)})

    # Uncertainty plots (convert std to original units)
    sea_std_orig = sea_std * splits.y_scaler.scale_[0]
    cfe_std_orig = cfe_std * splits.y_scaler.scale_[1]
    gp_uncertainty_plot(
        Y_true[:, 0], Y_true[:, 0], Y_pred[:, 0], sea_std_orig,
        "SEA", x_label="Actual SEA", title="GP_uncertainty_SEA",
    )
    gp_uncertainty_plot(
        Y_true[:, 1], Y_true[:, 1], Y_pred[:, 1], cfe_std_orig,
        "CFE", x_label="Actual CFE", title="GP_uncertainty_CFE",
    )

    gp.save(gp_dir)
    return gp


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_phases(splits, gp: MultiFidelityGP, models_dir: str = MODELS_DIR) -> dict:
    """
    Evaluate LF-only and full multi-fidelity GP on HF test set.
    Saves parity plots and a metrics JSON.
    """
    Y_true = splits.y_scaler.inverse_transform(splits.Y_hf_test)
    all_metrics: dict[str, dict] = {}
    by_config: dict[str, dict] = {}

    for label, use_corr in [("GP (LF only)", False), ("GP (TL)", True)]:
        sea_mu, cfe_mu = gp.predict(splits.X_hf_test, return_std=False, use_correction=use_corr)
        Y_pred_std = np.column_stack([sea_mu, cfe_mu])
        Y_pred = splits.y_scaler.inverse_transform(Y_pred_std)
        all_metrics[label] = compute_metrics(Y_true, Y_pred)
        by_config[label] = _by_config(splits, Y_true, Y_pred)
        slug = label.replace(" ", "_").replace("(", "").replace(")", "")
        parity_plot(Y_true, Y_pred, f"Parity_{slug}")

    print_metrics_table(all_metrics)
    save_metrics_json(all_metrics, os.path.join(models_dir, "gp_comparison_metrics.json"))
    if any(by_config.values()):
        print_metrics_by_config(by_config)
        save_metrics_json(by_config, os.path.join(models_dir, "gp_comparison_by_config.json"))
    return all_metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    guard_model_dir(args.models_dir, len(FEATURE_COLS), force=args.force)
    set_plots_dir(os.path.join(args.models_dir, "plots"))

    splits = load_data(seed=args.seed, lf_csv=args.lf_csv, hf_csv=args.hf_csv)
    splits._n_restarts = args.restarts
    save_scalers(splits, models_dir=args.models_dir)

    gp = MultiFidelityGP(n_restarts=args.restarts, n_features=splits.X_lf_train.shape[1])
    lf_done = False

    if args.phase in ("lf", "both"):
        gp = run_lf_fit(splits, gp=gp, gp_dir=os.path.join(args.models_dir, "gp"))
        lf_done = True

    if args.phase in ("correction", "both"):
        if not lf_done:
            gp = MultiFidelityGP.load(os.path.join(args.models_dir, "gp"))
        gp = run_correction_fit(splits, gp=gp, gp_dir=os.path.join(args.models_dir, "gp"))

    compare_phases(splits, gp, models_dir=args.models_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the multi-fidelity GP surrogate.")
    parser.add_argument(
        "--phase", default="both", choices=["lf", "correction", "both"],
        help="Which phase to run (default: both)"
    )
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--restarts",   type=int, default=5,
                        help="Number of kernel hyperparameter optimization restarts")
    parser.add_argument("--models-dir", default=MODELS_DIR)
    parser.add_argument("--lf-csv", nargs="+", default=None, metavar="CSV",
                        help="LF processed-data CSV(s) (default: the legacy shell file). "
                             "Missing N/T columns are read as N=6, T=0.")
    parser.add_argument("--hf-csv", nargs="+", default=None, metavar="CSV",
                        help="HF processed-data CSV(s) (default: the legacy solid file).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite a --models-dir that holds a model with a different input width.")
    args = parser.parse_args()
    main(args)
