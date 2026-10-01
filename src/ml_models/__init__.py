"""
Multi-fidelity surrogate models for SEA and CFE prediction.

Public API
----------
predict_mlp(X, phase)     -> (sea, cfe)  in original units
predict_gp(X)             -> (sea_mean, sea_std, cfe_mean, cfe_std)  in original units

X has columns [R, A, CC, VC, T, N] (T: 0/1 twist flag, N: 6 = hexagonal cells / 0 = ellipsoidal
holes).  The legacy 4-column form [R, A, CC, VC] means T=0, N=6.

Training entry points (run with python -m):
    python -m src.ml_models.train_mlp
    python -m src.ml_models.train_gp
"""

from .predict import predict_mlp, predict_gp
from .mlp_model import SurrogateNet, LAYER_SIZES
from .gp_model import MultiFidelityGP
from .data_loader import (
    load_data, load_scalers, save_scalers,
    FEATURE_COLS, LEGACY_FEATURE_COLS, CONFIG_COLS, TARGET_COLS,
)
