"""
Data loading, splitting, and standardization for the multi-fidelity surrogate models.

Design notes
------------
All StandardScaler objects are fit exclusively on the LF training split. Applying
the same scalers to HF data and the test set prevents any statistical information
from those sets from leaking into the feature normalization — a subtle but important
source of optimistic bias identified in Cawley & Talbot (2010).

The T column (a binary flag present in the raw CSVs) is all-zero for both fidelity
datasets and carries no predictive information; it is excluded from model inputs.

References
----------
Cawley, G.C., & Talbot, N.L.C. (2010). On over-fitting in model selection and
subsequent selection bias in performance evaluation. JMLR, 11, 2079-2107.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file so scripts run from any working dir)
# ---------------------------------------------------------------------------

_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data_folder", "1_param")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")

LF_CSV = os.path.join(DATA_DIR, "FDData_SobS_OD40L50G3_Shell_Fixed_PD.csv")
HF_CSV = os.path.join(DATA_DIR, "FDData_SobS_OD40L50G3_Solid_PD.csv")

FEATURE_COLS  = ["R", "A", "CC", "VC"]
TARGET_COLS   = ["SEA", "CFE"]
SCALERS_FILE  = "scalers.pkl"


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class DataSplits:
    """
    All standardized arrays and fitted scalers produced by load_data().

    Shapes (n = number of samples, standardized with LF-train scalers):
        X_lf_train     : (n_lf_tr, 4)   LF training inputs
        Y_lf_train     : (n_lf_tr, 2)   LF training targets [SEA, CFE]
        X_lf_val       : (n_lf_val, 4)  LF validation inputs
        Y_lf_val       : (n_lf_val, 2)  LF validation targets
        X_hf_finetune  : (n_hf_ft, 4)   HF fine-tune inputs
        Y_hf_finetune  : (n_hf_ft, 2)   HF fine-tune targets
        X_hf_test      : (n_hf_tst, 4)  HELD-OUT test inputs (never used in training)
        Y_hf_test      : (n_hf_tst, 2)  HELD-OUT test targets
    """

    X_lf_train: np.ndarray
    Y_lf_train: np.ndarray
    X_lf_val: np.ndarray
    Y_lf_val: np.ndarray

    X_hf_finetune: np.ndarray
    Y_hf_finetune: np.ndarray
    X_hf_test: np.ndarray
    Y_hf_test: np.ndarray

    x_scaler: StandardScaler
    y_scaler: StandardScaler

    n_lf: int
    n_hf: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(
    seed: int = 42,
    lf_val_frac: float = 0.20,
    hf_test_frac: float = 0.20,
) -> DataSplits:
    """
    Load both fidelity CSVs, create train/val/test splits, and standardize.

    Split protocol
    --------------
    LF  (938 pts): 80 % train (750) / 20 % validation (188)
        Used for MLP Phase-1 pre-training and GP LF fitting.

    HF  (123 pts): 80 % fine-tune (98) / 20 % test (25)
        Fine-tune set is used in MLP Phase-2 and GP correction fitting.
        The 25-point test set is held out for the final comparison only.

    Parameters
    ----------
    seed         : random seed (default 42 for reproducibility)
    lf_val_frac  : fraction of LF reserved for validation
    hf_test_frac : fraction of HF reserved as the final test set

    Returns
    -------
    DataSplits with all standardized arrays and fitted scalers.
    """
    for path in (LF_CSV, HF_CSV):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Data file not found: {path}")

    lf_df = pd.read_csv(LF_CSV).dropna(subset=FEATURE_COLS + TARGET_COLS)
    hf_df = pd.read_csv(HF_CSV).dropna(subset=FEATURE_COLS + TARGET_COLS)

    X_lf = lf_df[FEATURE_COLS].values.astype(float)
    Y_lf = lf_df[TARGET_COLS].values.astype(float)
    X_hf = hf_df[FEATURE_COLS].values.astype(float)
    Y_hf = hf_df[TARGET_COLS].values.astype(float)

    X_lf_tr, X_lf_val, Y_lf_tr, Y_lf_val = train_test_split(
        X_lf, Y_lf, test_size=lf_val_frac, random_state=seed
    )
    X_hf_ft, X_hf_tst, Y_hf_ft, Y_hf_tst = train_test_split(
        X_hf, Y_hf, test_size=hf_test_frac, random_state=seed
    )

    # Fit on LF train only — apply (transform) everywhere else
    x_sc = StandardScaler().fit(X_lf_tr)
    y_sc = StandardScaler().fit(Y_lf_tr)

    splits = DataSplits(
        X_lf_train=x_sc.transform(X_lf_tr),      Y_lf_train=y_sc.transform(Y_lf_tr),
        X_lf_val=x_sc.transform(X_lf_val),        Y_lf_val=y_sc.transform(Y_lf_val),
        X_hf_finetune=x_sc.transform(X_hf_ft),    Y_hf_finetune=y_sc.transform(Y_hf_ft),
        X_hf_test=x_sc.transform(X_hf_tst),       Y_hf_test=y_sc.transform(Y_hf_tst),
        x_scaler=x_sc,
        y_scaler=y_sc,
        n_lf=len(X_lf),
        n_hf=len(X_hf),
    )

    print(
        f"[data_loader] LF: {len(X_lf_tr)} train / {len(X_lf_val)} val  |  "
        f"HF: {len(X_hf_ft)} fine-tune / {len(X_hf_tst)} test  |  seed={seed}"
    )
    return splits


def save_scalers(splits: DataSplits, models_dir: str = MODELS_DIR) -> str:
    """Persist the two scalers to <models_dir>/scalers.pkl.  Returns the path."""
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, SCALERS_FILE)
    joblib.dump({"x_scaler": splits.x_scaler, "y_scaler": splits.y_scaler}, path)
    print(f"[data_loader] Scalers saved -> {path}")
    return path


def load_scalers(models_dir: str = MODELS_DIR) -> tuple[StandardScaler, StandardScaler]:
    """Load and return (x_scaler, y_scaler) from <models_dir>/scalers.pkl."""
    path = os.path.join(models_dir, SCALERS_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Scalers not found at {path}.  Run train_mlp or train_gp first."
        )
    d = joblib.load(path)
    return d["x_scaler"], d["y_scaler"]
