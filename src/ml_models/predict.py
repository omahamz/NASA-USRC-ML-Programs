"""
Unified prediction interface for both surrogate models.

Both functions accept raw (un-standardized) inputs in original physical units,
apply the saved scalers internally, and return predictions in original units.

Input columns are [R, A, CC, VC, T, N]:
    T : 0 = untwisted, 1 = twist-angle part
    N : 6 = hexagonal cells, 0 = ellipsoidal holes
T and N are categorical flags: any other value raises ValueError.  The legacy 4-column
form [R, A, CC, VC] is still accepted and means (T=0, N=6) - with a warning when the
loaded model has 6 inputs.  A model trained on the legacy 4-input data only supports that
configuration.

CC and VC are intentionally NOT rounded — the caller can observe how the model
responds to non-integer values, which is informative for understanding the model's
interpolation behavior between the discrete integer design points.

Examples
--------
>>> import numpy as np
>>> from src.ml_models.predict import predict_mlp, predict_gp

>>> X = np.array([[3.5, 60.0, 12.0, 6.0, 0, 6],   # R, A, CC, VC, T, N
...               [5.0, 45.0,  8.0, 5.0, 0, 0]])

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

from .data_loader import (
    FEATURE_COLS,
    LEGACY_FEATURE_COLS,
    LEGACY_CONFIG_DEFAULTS,
    MODELS_DIR,
    VALID_N,
    VALID_T,
    load_scalers,
)
from .gp_model import MultiFidelityGP
from .mlp_model import SurrogateNet

_GP_DIR = os.path.join(MODELS_DIR, "gp")

# Domain bounds of the continuous / integer inputs (from sample.py / constraints.py;
# a test keeps the copies in sync).  T and N are categorical and validated strictly.
_BOUNDS = {
    "R":  (2.0,  8.8),
    "A":  (30.0, 90.0),
    "CC": (4.0,  22.0),
    "VC": (4.0,  10.0),
}


def _validate_X(X: np.ndarray) -> None:
    """Warn (do not error) if any continuous/integer input is outside its known domain."""
    for j, col in enumerate(FEATURE_COLS[: X.shape[1]]):
        if col not in _BOUNDS:
            continue
        lo, hi = _BOUNDS[col]
        out = np.any((X[:, j] < lo) | (X[:, j] > hi))
        if out:
            warnings.warn(
                f"Column '{col}' has values outside the training domain [{lo}, {hi}]. "
                "Predictions may be unreliable (extrapolation).",
                UserWarning,
                stacklevel=4,
            )


def _validate_config(T: np.ndarray, N: np.ndarray) -> None:
    """T and N are categorical flags - reject anything but {0, 1} / {0, 6}."""
    for name, vals, valid in (("T", T, VALID_T), ("N", N, VALID_N)):
        if not np.all(np.isin(vals, valid)):
            bad = sorted({float(v) for v in vals[~np.isin(vals, valid)]})[:5]
            raise ValueError(f"Column '{name}' must contain only {valid}; found {bad}.")


def _prepare_X(X, x_scaler) -> np.ndarray:
    """
    Validate X and return raw inputs with exactly the width the fitted scaler expects.

    * 6 columns [R, A, CC, VC, T, N]        -> used as-is (6-input model)
    * 4 columns [R, A, CC, VC]              -> padded with (T=0, N=6) for a 6-input model
    * 6 columns on a legacy 4-input model   -> allowed only for (T=0, N=6); T/N dropped
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[np.newaxis, :]

    n_in    = X.shape[1]
    n_model = int(getattr(x_scaler, "n_features_in_", len(FEATURE_COLS)))
    if n_in not in (len(LEGACY_FEATURE_COLS), len(FEATURE_COLS)):
        raise ValueError(
            f"X must have {len(FEATURE_COLS)} columns {FEATURE_COLS} "
            f"(or the legacy {len(LEGACY_FEATURE_COLS)} columns {LEGACY_FEATURE_COLS}), got {n_in}."
        )

    if n_in == len(FEATURE_COLS):
        _validate_config(X[:, 4], X[:, 5])
        if n_model == len(LEGACY_FEATURE_COLS):
            legacy = np.all(X[:, 4] == LEGACY_CONFIG_DEFAULTS["T"]) and np.all(X[:, 5] == LEGACY_CONFIG_DEFAULTS["N"])
            if not legacy:
                raise ValueError(
                    "This model was trained on the legacy 4-input data (R, A, CC, VC) and only "
                    "supports (T=0, N=6). Retrain with the 6-input pipeline to predict other configurations."
                )
            X = X[:, : len(LEGACY_FEATURE_COLS)]
    elif n_model == len(FEATURE_COLS):
        warnings.warn(
            "X has 4 columns [R, A, CC, VC]; assuming the legacy configuration (T=0, N=6). "
            "Pass 6 columns [R, A, CC, VC, T, N] to be explicit.",
            UserWarning,
            stacklevel=3,
        )
        pad = np.tile([LEGACY_CONFIG_DEFAULTS["T"], LEGACY_CONFIG_DEFAULTS["N"]], (len(X), 1)).astype(float)
        X = np.hstack([X, pad])

    _validate_X(X)
    return X


def predict_mlp(
    X: np.ndarray | list,
    phase: str = "hf",
    models_dir: str = MODELS_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict SEA and CFE using the saved MLP.

    Parameters
    ----------
    X        : array-like of shape (n, 6), columns = [R, A, CC, VC, T, N] in original units
               (the legacy (n, 4) form [R, A, CC, VC] means T=0, N=6)
    phase    : 'hf' → fine-tuned model (recommended)
               'lf' → LF pre-trained model (baseline comparison)
    models_dir : directory containing saved model files

    Returns
    -------
    sea : (n,) array of SEA predictions in original units  (N·mm / mm³)
    cfe : (n,) array of CFE predictions in original units  [0, 1]
    """
    x_sc, y_sc = load_scalers(models_dir)
    X_std = x_sc.transform(_prepare_X(X, x_sc))

    filename = "mlp_finetuned_hf.pt" if phase == "hf" else "mlp_pretrained_lf.pt"
    model = SurrogateNet.load(os.path.join(models_dir, filename))
    model.eval()

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
    X : array-like of shape (n, 6), columns = [R, A, CC, VC, T, N] in original units
        (the legacy (n, 4) form [R, A, CC, VC] means T=0, N=6)

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
    x_sc, y_sc = load_scalers(models_dir)
    X_std = x_sc.transform(_prepare_X(X, x_sc))

    gp = MultiFidelityGP.load(os.path.join(models_dir, "gp"))
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
    X     : (n, 6) raw inputs [R, A, CC, VC, T, N]  (legacy (n, 4) accepted)
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
