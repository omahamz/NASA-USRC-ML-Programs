"""
Unified prediction interface for both surrogate models.

Both functions accept raw (un-standardized) inputs in original physical units,
apply the saved scalers internally, and return predictions in original units.

CC and VC are intentionally NOT rounded — the caller can observe how the model
responds to non-integer values, which is informative for understanding the model's
interpolation behavior between the discrete integer design points.

Examples
--------
>>> import numpy as np
>>> from src.ml_models.predict import predict_mlp, predict_gp

>>> X = np.array([[3.5, 60.0, 12.0, 6.0],   # R, A, CC, VC
...               [5.0, 45.0,  8.0, 5.0]])

>>> sea, cfe = predict_mlp(X, phase="hf")
>>> print(sea, cfe)

>>> sea_mu, sea_std, cfe_mu, cfe_std = predict_gp(X)
>>> print(f"SEA = {sea_mu[0]:.3f} ± {sea_std[0]:.3f}")
>>> print(f"CFE = {cfe_mu[0]:.3f} ± {cfe_std[0]:.3f}")
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import torch

from .data_loader import FEATURE_COLS, MODELS_DIR, load_scalers
from .gp_model import MultiFidelityGP
from .mlp_model import SurrogateNet

_GP_DIR = os.path.join(MODELS_DIR, "gp")

# Domain bounds (from sample.py)
_BOUNDS = {
    "R":  (2.0,  8.8),
    "A":  (30.0, 90.0),
    "CC": (4.0,  22.0),
    "VC": (4.0,  10.0),
}


def _validate_X(X: np.ndarray) -> None:
    """Warn (do not error) if any input is outside its known domain."""
    for j, col in enumerate(FEATURE_COLS):
        lo, hi = _BOUNDS[col]
        out = np.any((X[:, j] < lo) | (X[:, j] > hi))
        if out:
            warnings.warn(
                f"Column '{col}' has values outside the training domain [{lo}, {hi}]. "
                "Predictions may be unreliable (extrapolation).",
                UserWarning,
                stacklevel=3,
            )


def predict_mlp(
    X: np.ndarray | list,
    phase: str = "hf",
    models_dir: str = MODELS_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict SEA and CFE using the saved MLP.

    Parameters
    ----------
    X        : array-like of shape (n, 4), columns = [R, A, CC, VC] in original units
    phase    : 'hf' → fine-tuned model (recommended)
               'lf' → LF pre-trained model (baseline comparison)
    models_dir : directory containing saved model files

    Returns
    -------
    sea : (n,) array of SEA predictions in original units  (N·mm / mm³)
    cfe : (n,) array of CFE predictions in original units  [0, 1]
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[np.newaxis, :]
    if X.shape[1] != 4:
        raise ValueError(f"X must have 4 columns [R, A, CC, VC], got {X.shape[1]}.")
    _validate_X(X)

    filename = "mlp_finetuned_hf.pt" if phase == "hf" else "mlp_pretrained_lf.pt"
    model = SurrogateNet.load(os.path.join(models_dir, filename))
    model.eval()

    x_sc, y_sc = load_scalers(models_dir)
    X_std = x_sc.transform(X)

    with torch.no_grad():
        Y_std = model(torch.FloatTensor(X_std)).numpy()

    Y = y_sc.inverse_transform(Y_std)
    return Y[:, 0], Y[:, 1]   # sea, cfe


def predict_gp(
    X: np.ndarray | list,
    models_dir: str = MODELS_DIR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict SEA and CFE using the saved multi-fidelity GP.

    Returns posterior mean and standard deviation for each output, enabling
    uncertainty-aware decision-making during optimization or design exploration.

    Parameters
    ----------
    X : array-like of shape (n, 4), columns = [R, A, CC, VC] in original units

    Returns
    -------
    sea_mean : (n,) GP posterior mean for SEA
    sea_std  : (n,) GP posterior std  for SEA  (in original units)
    cfe_mean : (n,) GP posterior mean for CFE
    cfe_std  : (n,) GP posterior std  for CFE  (in original units)

    Notes on uncertainty
    --------------------
    The standard deviation is composed of both the LF GP uncertainty and the
    correction GP uncertainty:  std_HF = sqrt(std_LF² + std_delta²).
    It reflects how confident the model is at a given input location — larger
    std means the input is far from training data (sparse region of the Sobol
    sample).  Use ±2σ for a ~95 % credible interval.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[np.newaxis, :]
    if X.shape[1] != 4:
        raise ValueError(f"X must have 4 columns [R, A, CC, VC], got {X.shape[1]}.")
    _validate_X(X)

    x_sc, y_sc = load_scalers(models_dir)
    gp = MultiFidelityGP.load(os.path.join(models_dir, "gp"))

    X_std = x_sc.transform(X)
    sea_mu_s, sea_std_s, cfe_mu_s, cfe_std_s = gp.predict(X_std, return_std=True)

    # Inverse-transform the means via the scaler
    Y_mu = y_sc.inverse_transform(np.column_stack([sea_mu_s, cfe_mu_s]))

    # Scale the standard deviations by the output scaler's scale_ attribute.
    # StandardScaler: y_orig = y_std * scale_ + mean_  →  σ_orig = σ_std * scale_
    sea_std = sea_std_s * y_sc.scale_[0]
    cfe_std = cfe_std_s * y_sc.scale_[1]

    return Y_mu[:, 0], sea_std, Y_mu[:, 1], cfe_std


def predict_objective(
    X: np.ndarray | list,
    k: float | None = None,
    model: str = "gp",
    phase: str = "hf",
    models_dir: str = MODELS_DIR,
) -> np.ndarray:
    """
    Compute the scalarized objective Obj = SEA - k*(1-CFE) from model predictions.

    Parameters
    ----------
    X     : (n, 4) raw inputs
    k     : weighting coefficient.  None → uses mean(SEA_pred) of the batch,
            matching the convention in analysis.py.
    model : 'gp' or 'mlp'
    phase : 'hf' or 'lf' (relevant for MLP only)

    Returns
    -------
    (n,) array of objective values in original units
    """
    if model == "gp":
        sea, _, cfe, _ = predict_gp(X, models_dir)
    else:
        sea, cfe = predict_mlp(X, phase=phase, models_dir=models_dir)

    if k is None:
        k = float(sea.mean())
    return sea - k * (1.0 - cfe)
