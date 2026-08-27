"""
alpha_sweep.py
==============
Analyse how optimal design-variable ranges shift as alpha (SEA/CFE weight) varies.

For each alpha in [alpha_min, alpha_max] at a fixed step size:
  - Run GP exploitation  (n proposals)
  - Run MLP DE           (n proposals)
  Compute mean ± std of each input variable (R, A, CC, VC) and each predicted
  output (SEA_pred, CFE_pred) across the returned proposals.

Two plots are saved:
  alpha_sweep_inputs_<ts>.png   — 2×2 grid, one subplot per input variable
  alpha_sweep_outputs_<ts>.png  — 1×2 grid, SEA_pred and CFE_pred

Results are cached to alpha_sweep_cache.json so re-runs are instant.  Pass
--no-cache to force a full re-run.

Usage
-----
    python src/alpha_sweep.py                            # default sweep
    python src/alpha_sweep.py --step 0.1                # coarser grid
    python src/alpha_sweep.py --models gp               # GP only
    python src/alpha_sweep.py --n 5 --step 0.2          # quick test
    python src/alpha_sweep.py --no-cache                # force re-run
    python src/alpha_sweep.py --list                    # print what would run
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
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
matplotlib.use("Agg")          # consistent with optimizer.py; plots saved to file
import matplotlib.pyplot as plt

from optimizer import run_optimizer
from ml_models.data_loader import FEATURE_COLS   # ["R", "A", "CC", "VC"]


# ---------------------------------------------------------------------------
# Constants / style
# ---------------------------------------------------------------------------

ALPHA_MIN   = 0.0
ALPHA_MAX   = 1.0
ALPHA_STEP  = 0.05
N_PROPOSALS = 10

INPUT_VARS  = FEATURE_COLS                 # ["R", "A", "CC", "VC"]
OUTPUT_VARS = ["SEA_pred", "CFE_pred"]

_OUT_DIR = os.path.normpath(
    os.path.join(_HERE, "..", "data_folder", "output", "alpha_sweep")
)

_MODEL_STYLES: dict[str, dict] = {
    "gp":  {"color": "steelblue",  "label": "GP (exploitation)"},
    "mlp": {"color": "darkorange", "label": "MLP (DE)"},
}

# Human-readable y-axis labels
_VAR_LABELS: dict[str, str] = {
    "R":        "R  (corner radius, mm)",
    "A":        "A  (angle, deg)",
    "CC":       "CC  (cell count)",
    "VC":       "VC  (volume coeff)",
    "SEA_pred": "SEA pred  (N.mm/mm3)",
    "CFE_pred": "CFE pred",
}


# ---------------------------------------------------------------------------
# Silent optimizer runner
# ---------------------------------------------------------------------------

def _run_silent(model: str, alpha: float, n: int) -> pd.DataFrame | None:
    """
    Run the optimizer, suppressing all stdout and warnings.
    Returns the proposals DataFrame, or None if the run failed.
    """
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = run_optimizer(
                model       = model,
                mode        = "exploitation",   # GP path
                method      = "de",             # MLP path
                n_proposals = n,
                alpha       = alpha,
                dry_run     = True,             # skip file writes inside optimizer
            )
        return df
    except Exception as exc:
        print(f"    [WARN] {model} alpha={alpha:.2f} failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def sweep_alpha(
    alphas: list[float],
    models: list[str],
    n:      int,
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """
    For every (model, alpha) pair, run the optimizer and record per-proposal
    values for all tracked variables.

    Returns
    -------
    results[model][alpha_key][variable] = [v1, v2, ..., vn]

    alpha_key is the alpha value formatted as a 4-decimal string, e.g. "0.3500".
    """
    results: dict = {m: {} for m in models}
    total = len(alphas) * len(models)
    done  = 0

    for model in models:
        style = _MODEL_STYLES[model]
        print(f"\n  [{style['label']}]  {len(alphas)} alpha values, {n} proposals each")

        for alpha in alphas:
            done += 1
            key = f"{alpha:.4f}"
            print(f"    [{done:>3}/{total}]  alpha={alpha:.2f} ...", end=" ", flush=True)

            df = _run_silent(model, alpha, n)

            if df is None or df.empty:
                print("SKIPPED")
                continue

            track = [c for c in INPUT_VARS + OUTPUT_VARS if c in df.columns]
            results[model][key] = {c: df[c].tolist() for c in track}
            print(f"OK  ({len(df)} proposals)")

    return results


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def _cache_path(out_dir: str) -> str:
    return os.path.join(out_dir, "alpha_sweep_cache.json")


def save_cache(results: dict, alphas: list[float], n: int, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    payload = {"params": {"alphas": alphas, "n": n}, "results": results}
    with open(_cache_path(out_dir), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[alpha_sweep] Cache saved  → {_cache_path(out_dir)}")


def load_cache(
    out_dir: str,
    alphas:  list[float],
    n:       int,
    models:  list[str],
) -> dict | None:
    """
    Load cache if it exists AND matches the current run parameters.
    Returns the results dict, or None if cache is missing / stale.
    """
    path = _cache_path(out_dir)
    if not os.path.isfile(path):
        return None

    with open(path) as f:
        payload = json.load(f)

    cached_alphas  = payload["params"]["alphas"]
    cached_n       = payload["params"]["n"]
    cached_results = payload["results"]

    if cached_alphas != alphas or cached_n != n:
        print("[alpha_sweep] Cache exists but parameters differ — re-running.")
        return None
    if not all(m in cached_results for m in models):
        print("[alpha_sweep] Cache missing some requested models — re-running.")
        return None

    print(f"[alpha_sweep] Cache hit  ← {path}")
    return cached_results


# ---------------------------------------------------------------------------
# Plot utilities
# ---------------------------------------------------------------------------

def _extract_series(
    results: dict,
    model:   str,
    alphas:  list[float],
    var:     str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract (alpha_values, means, stds) for one (model, variable) pair.
    Entries with no data become NaN (plotted as gaps).
    """
    means, stds = [], []
    for alpha in alphas:
        key  = f"{alpha:.4f}"
        vals = results[model].get(key, {}).get(var, [])
        if vals:
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
        else:
            means.append(float("nan"))
            stds.append(float("nan"))
    return np.array(alphas), np.array(means), np.array(stds)


def _draw_panel(
    ax:      plt.Axes,
    results: dict,
    alphas:  list[float],
    models:  list[str],
    var:     str,
) -> None:
    """Draw mean lines + ±1σ shaded bands for all models on a single Axes."""
    for model in models:
        if model not in results or not results[model]:
            continue
        style = _MODEL_STYLES[model]
        x, means, stds = _extract_series(results, model, alphas, var)
        valid = ~np.isnan(means)
        if not valid.any():
            continue

        ax.plot(
            x[valid], means[valid],
            color=style["color"], linewidth=2.0,
            marker="o", markersize=4,
            label=style["label"],
        )
        ax.fill_between(
            x[valid],
            (means - stds)[valid],
            (means + stds)[valid],
            color=style["color"], alpha=0.15,
            label="_nolegend_",
        )

    # Vertical reference lines at the pure-CFE and pure-SEA extremes
    ax.axvline(0.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

    ax.set_ylabel(_VAR_LABELS.get(var, var), fontsize=9)
    ax.set_xlim(ALPHA_MIN - 0.03, ALPHA_MAX + 0.03)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8, loc="best")


def _shared_xlabel(axes_row: list[plt.Axes]) -> None:
    """Set the x-label and ticks on a row of Axes."""
    for ax in axes_row:
        ax.set_xlabel(
            "alpha   (0 = CFE only  <-->  SEA only = 1)",
            fontsize=9,
        )
        ax.set_xticks(np.arange(0.0, 1.05, 0.1))


def plot_sweep(
    results: dict,
    alphas:  list[float],
    out_dir: str = _OUT_DIR,
) -> list[str]:
    """
    Build and save both figures.  Returns a list of saved file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    models_present = [m for m in ["gp", "mlp"] if m in results and results[m]]
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved: list[str] = []

    suptitle_base = (
        "alpha * SEA_norm + (1-alpha) * CFE"
        "   |   shaded band = +/- 1 std across proposals"
    )

    # ── Figure 1: Input variables (2×2) ──────────────────────────────────
    fig1, axes1 = plt.subplots(2, 2, figsize=(13, 8))
    fig1.suptitle(f"Optimal Input Variables vs. Alpha\n{suptitle_base}", fontsize=12)

    for ax, var in zip(axes1.flatten(), INPUT_VARS):
        _draw_panel(ax, results, alphas, models_present, var)
        ax.set_title(var, fontsize=10)

    _shared_xlabel(list(axes1[1]))
    plt.tight_layout()

    p1 = os.path.join(out_dir, f"alpha_sweep_inputs_{ts}.png")
    fig1.savefig(p1, dpi=130, bbox_inches="tight")
    saved.append(p1)
    plt.close(fig1)

    # ── Figure 2: Predicted outputs (1×2) ────────────────────────────────
    # Only draw if at least one model returned prediction columns
    has_outputs = any(
        any(v in results[m].get(f"{a:.4f}", {}) for a in alphas for v in OUTPUT_VARS)
        for m in models_present
    )
    if has_outputs:
        fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4.5))
        fig2.suptitle(f"Predicted Outputs vs. Alpha\n{suptitle_base}", fontsize=12)

        for ax, var in zip(axes2, OUTPUT_VARS):
            _draw_panel(ax, results, alphas, models_present, var)
            ax.set_title(_VAR_LABELS.get(var, var), fontsize=10)

        _shared_xlabel(list(axes2))
        plt.tight_layout()

        p2 = os.path.join(out_dir, f"alpha_sweep_outputs_{ts}.png")
        fig2.savefig(p2, dpi=130, bbox_inches="tight")
        saved.append(p2)
        plt.close(fig2)

    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_alphas(alpha_min: float, alpha_max: float, step: float) -> list[float]:
    n      = round((alpha_max - alpha_min) / step)
    alphas = [round(alpha_min + i * step, 8) for i in range(n + 1)]
    return [a for a in alphas if alpha_min - 1e-9 <= a <= alpha_max + 1e-9]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep alpha (SEA/CFE scalarization weight) from 0 to 1 and plot how "
            "the optimal input variable ranges change for GP and MLP optimizers."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/alpha_sweep.py\n"
            "  python src/alpha_sweep.py --step 0.1\n"
            "  python src/alpha_sweep.py --models gp\n"
            "  python src/alpha_sweep.py --n 5 --step 0.2\n"
            "  python src/alpha_sweep.py --no-cache\n"
            "  python src/alpha_sweep.py --list\n"
        ),
    )
    parser.add_argument("--n", type=int, default=N_PROPOSALS,
                        help=f"Proposals per optimizer run. Default: {N_PROPOSALS}.")
    parser.add_argument("--step", type=float, default=ALPHA_STEP,
                        help=f"Alpha increment. Default: {ALPHA_STEP}.")
    parser.add_argument("--alpha-min", type=float, default=ALPHA_MIN, dest="alpha_min",
                        help=f"Start of alpha range. Default: {ALPHA_MIN}.")
    parser.add_argument("--alpha-max", type=float, default=ALPHA_MAX, dest="alpha_max",
                        help=f"End of alpha range. Default: {ALPHA_MAX}.")
    parser.add_argument("--models", type=str, nargs="+", default=["gp", "mlp"],
                        choices=["gp", "mlp"],
                        help="Models to sweep. Default: gp mlp.")
    parser.add_argument("--no-cache", action="store_true", dest="no_cache",
                        help="Ignore existing cache and re-run all optimizations.")
    parser.add_argument("--list", action="store_true",
                        help="Print the alpha grid and exit without running.")
    return parser.parse_args()


def main() -> None:
    args   = _parse_args()
    alphas = _build_alphas(args.alpha_min, args.alpha_max, args.step)

    print(f"\n[alpha_sweep] Grid: {len(alphas)} values  "
          f"[{alphas[0]:.2f} to {alphas[-1]:.2f}, step={args.step}]")
    print(f"[alpha_sweep] Models: {args.models}   n={args.n} proposals each")
    print(f"[alpha_sweep] Total optimizer runs: {len(alphas) * len(args.models)}")

    if args.list:
        print(f"[alpha_sweep] Alpha values: {[round(a, 4) for a in alphas]}")
        return

    # ── Cache lookup ──────────────────────────────────────────────────────
    results: dict | None = None
    if not args.no_cache:
        results = load_cache(_OUT_DIR, alphas, args.n, args.models)

    # ── Run sweep if needed ───────────────────────────────────────────────
    if results is None:
        print("\n[alpha_sweep] Starting sweep  (optimizer stdout suppressed)...")
        results = sweep_alpha(alphas=alphas, models=args.models, n=args.n)
        save_cache(results, alphas, args.n, _OUT_DIR)

    # ── Plot ──────────────────────────────────────────────────────────────
    print("\n[alpha_sweep] Generating plots...")
    saved = plot_sweep(results, alphas, out_dir=_OUT_DIR)
    for path in saved:
        print(f"[alpha_sweep] Saved → {path}")
    print("[alpha_sweep] Done.")


if __name__ == "__main__":
    main()
