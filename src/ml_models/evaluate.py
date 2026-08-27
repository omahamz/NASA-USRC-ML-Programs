"""
Evaluation metrics and visualization for the multi-fidelity surrogate models.

All metric functions operate on predictions in original (non-standardized) units.
Callers are responsible for inverse-transforming before calling here.
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET_NAMES = ["SEA", "CFE"]
_PLOTS_DIR   = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "plots",
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, dict[str, float]]:
    """
    Compute R², RMSE, and MAE for each output column.

    Parameters
    ----------
    y_true : (n, 2) array of true values  [SEA, CFE]
    y_pred : (n, 2) array of predictions  [SEA, CFE]

    Returns
    -------
    dict  { 'SEA': {'R2': ..., 'RMSE': ..., 'MAE': ...},
            'CFE': {'R2': ..., 'RMSE': ..., 'MAE': ...} }
    """
    out = {}
    for i, name in enumerate(TARGET_NAMES):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        out[name] = {
            "R2":   round(float(r2_score(yt, yp)),                            4),
            "RMSE": round(float(np.sqrt(mean_squared_error(yt, yp))),         4),
            "MAE":  round(float(mean_absolute_error(yt, yp)),                 4),
        }
    return out


def print_metrics_table(metrics: dict[str, dict[str, dict[str, float]]]) -> None:
    """
    Print a formatted comparison table for multiple model variants.

    Parameters
    ----------
    metrics : { 'Model Name': {'SEA': {'R2': ..., ...}, 'CFE': {...}}, ... }
    """
    col_w = 14
    header = f"{'Model':<24}" + "".join(
        f"{'SEA-R2':>{col_w}}{'SEA-RMSE':>{col_w}}{'CFE-R2':>{col_w}}{'CFE-RMSE':>{col_w}}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for name, m in metrics.items():
        row = f"{name:<24}"
        for out in TARGET_NAMES:
            row += f"{m[out]['R2']:>{col_w}.4f}{m[out]['RMSE']:>{col_w}.4f}"
        print(row)
    print("=" * len(header) + "\n")


def save_metrics_json(
    metrics: dict,
    path: str,
) -> None:
    """Write a metrics dict to a JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluate] Metrics saved -> {path}")


# ---------------------------------------------------------------------------
# Parity plot
# ---------------------------------------------------------------------------

def parity_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save: bool = True,
) -> plt.Figure:
    """
    Two-panel parity plot (predicted vs actual) for SEA and CFE.

    The perfect-prediction diagonal (y=x) is shown in red.  Points are
    coloured by absolute residual so under-/over-predictions are easy to spot.

    Parameters
    ----------
    y_true : (n, 2)  [SEA, CFE] in original units
    y_pred : (n, 2)  [SEA, CFE] in original units
    title  : figure title (also used as the saved filename)
    save   : write PNG to models/plots/
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Compute all metrics once upfront
    all_m = compute_metrics(y_true, y_pred)

    for ax, i, name in zip(axes, range(2), TARGET_NAMES):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        resid = np.abs(yp - yt)

        lo = min(yt.min(), yp.min()) * 0.95
        hi = max(yt.max(), yp.max()) * 1.05
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect", zorder=1)

        sc = ax.scatter(yt, yp, c=resid, cmap="viridis", alpha=0.7, s=50, zorder=2)
        plt.colorbar(sc, ax=ax, label="|residual|")

        m = all_m[name]
        ax.set_xlabel(f"Actual {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(f"{name}  |  R2={m['R2']:.3f}  RMSE={m['RMSE']:.3f}")
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    if save:
        _save_fig(fig, title)
    return fig


# ---------------------------------------------------------------------------
# Learning curve
# ---------------------------------------------------------------------------

def learning_curve_plot(
    history: dict,
    title: str,
    save: bool = True,
) -> plt.Figure:
    """
    Plot training and validation MSE over epochs.

    Parameters
    ----------
    history : dict with keys 'epochs', 'train_total', 'val_total',
              'train_sea', 'train_cfe' (all lists of equal length)
    title   : figure title
    save    : write PNG to models/plots/
    """
    epochs = history["epochs"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Left: total loss
    ax = axes[0]
    ax.plot(epochs, history["train_total"], label="Train (total)", linewidth=1.5)
    if history.get("val_total"):
        ax.plot(epochs, history["val_total"], label="Val (total)", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (standardized)")
    ax.set_title("Total Loss")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_yscale("log")

    # Right: per-output loss
    ax = axes[1]
    ax.plot(epochs, history["train_sea"], label="Train SEA", linewidth=1.5)
    ax.plot(epochs, history["train_cfe"], label="Train CFE", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (standardized)")
    ax.set_title("Per-Output Training Loss")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_yscale("log")

    plt.tight_layout()
    if save:
        _save_fig(fig, title)
    return fig


# ---------------------------------------------------------------------------
# Comparison bar chart
# ---------------------------------------------------------------------------

def comparison_bar_chart(
    metrics: dict[str, dict[str, dict[str, float]]],
    title: str = "Model Comparison — R² on HF Test Set",
    save: bool = True,
) -> plt.Figure:
    """
    Grouped bar chart comparing R² across multiple model variants.

    Parameters
    ----------
    metrics : { 'Model Name': {'SEA': {'R2': ...}, 'CFE': {'R2': ...}}, ... }
    """
    model_names = list(metrics.keys())
    sea_r2 = [metrics[m]["SEA"]["R2"] for m in model_names]
    cfe_r2 = [metrics[m]["CFE"]["R2"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(model_names) * 1.8), 5))
    bars1 = ax.bar(x - width / 2, sea_r2, width, label="SEA", color="steelblue",  alpha=0.85)
    bars2 = ax.bar(x + width / 2, cfe_r2, width, label="CFE", color="darkorange", alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("R² (higher is better)")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(min(0, min(sea_r2 + cfe_r2) - 0.1), 1.05)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    for bar in (*bars1, *bars2):
        h = bar.get_height()
        ax.annotate(
            f"{h:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=8,
        )

    plt.tight_layout()
    if save:
        _save_fig(fig, title)
    return fig


# ---------------------------------------------------------------------------
# GP uncertainty plot
# ---------------------------------------------------------------------------

def gp_uncertainty_plot(
    x_axis: np.ndarray,
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    output_name: str,
    x_label: str = "Sample index",
    title: str | None = None,
    save: bool = True,
) -> plt.Figure:
    """
    Plot GP posterior mean ± 2σ against true values, sorted by x_axis.

    Parameters
    ----------
    x_axis      : (n,) array to sort and use as x-axis (e.g., actual SEA values)
    y_true      : (n,) true values
    y_mean      : (n,) GP posterior mean
    y_std       : (n,) GP posterior standard deviation (in original units)
    output_name : 'SEA' or 'CFE'
    """
    order = np.argsort(x_axis)
    xs    = x_axis[order]
    yt    = y_true[order]
    ym    = y_mean[order]
    ys    = y_std[order]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(xs, ym - 2 * ys, ym + 2 * ys, alpha=0.25, color="steelblue", label="±2σ")
    ax.plot(xs, ym, color="steelblue", linewidth=1.5, label="GP mean")
    ax.scatter(xs, yt, color="darkred", s=25, zorder=3, alpha=0.7, label="Actual")
    ax.set_xlabel(x_label)
    ax.set_ylabel(output_name)
    ax.set_title(title or f"GP Uncertainty — {output_name}", fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save:
        _save_fig(fig, title or f"GP_uncertainty_{output_name}")
    return fig


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, title: str) -> None:
    os.makedirs(_PLOTS_DIR, exist_ok=True)
    safe = title.replace(" ", "_").replace("/", "-").replace("|", "").replace("²", "2")
    path = os.path.join(_PLOTS_DIR, f"{safe}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[evaluate] Plot saved -> {path}")
