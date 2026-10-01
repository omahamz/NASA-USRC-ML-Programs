"""
Data loading, splitting, and standardization for the multi-fidelity surrogate models.

Inputs are  [R, A, CC, VC, T, N]:
    R, A, CC, VC : geometry parameters
    T            : 0 = untwisted part, 1 = twist-angle part        (binary)
    N            : 6 = hexagonal cells, 0 = ellipsoidal holes       (binary, values 0 / 6)

Legacy processed-data CSVs have no N column (the whole campaign was hexagonal) and a
constant T = 0.  They still load: a missing N is filled with 6 and a missing T with 0, so
the 6-input pipeline can be trained on them as a sanity check (T and N are then constant
columns; StandardScaler maps a constant column to 0).

Design notes
------------
All StandardScaler objects are fit exclusively on the LF training split. Applying
the same scalers to HF data and the test set prevents any statistical information
from those sets from leaking into the feature normalization — a subtle but important
source of optimistic bias identified in Cawley & Talbot (2010).

Splits are stratified by (N, T) configuration whenever every configuration has at least
two rows, so a small HF test set cannot end up without one of the part families.

References
----------
Cawley, G.C., & Talbot, N.L.C. (2010). On over-fitting in model selection and
subsequent selection bias in performance evaluation. JMLR, 11, 2079-2107.
"""

from __future__ import annotations

import math
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

LEGACY_FEATURE_COLS = ["R", "A", "CC", "VC"]
CONFIG_COLS         = ["T", "N"]
FEATURE_COLS        = LEGACY_FEATURE_COLS + CONFIG_COLS
TARGET_COLS         = ["SEA", "CFE"]
SCALERS_FILE        = "scalers.pkl"

# Allowed configuration values.  Deliberately duplicated from src/constraints.py:
# ml_models must import without `src` on sys.path (python -m src.ml_models.train_mlp).
# tests/test_ml_data.py asserts the two copies agree.
VALID_N = (0, 6)
VALID_T = (0, 1)
LEGACY_CONFIG_DEFAULTS = {"N": 6, "T": 0}


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class DataSplits:
    """
    All standardized arrays and fitted scalers produced by load_data().

    Shapes (n = number of samples, F = len(FEATURE_COLS) = 6, standardized with
    LF-train scalers):
        X_lf_train     : (n_lf_tr, F)   LF training inputs
        Y_lf_train     : (n_lf_tr, 2)   LF training targets [SEA, CFE]
        X_lf_val       : (n_lf_val, F)  LF validation inputs
        Y_lf_val       : (n_lf_val, 2)  LF validation targets
        X_hf_finetune  : (n_hf_ft, F)   HF fine-tune inputs
        Y_hf_finetune  : (n_hf_ft, 2)   HF fine-tune targets
        X_hf_test      : (n_hf_tst, F)  HELD-OUT test inputs (never used in training)
        Y_hf_test      : (n_hf_tst, 2)  HELD-OUT test targets

    config_* hold one "N6_T0"-style label per row of the matching split (used for the
    per-configuration metrics).
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

    config_lf_train: np.ndarray | None = None
    config_lf_val: np.ndarray | None = None
    config_hf_finetune: np.ndarray | None = None
    config_hf_test: np.ndarray | None = None
    feature_cols: list | None = None


# ---------------------------------------------------------------------------
# Reading processed-data CSVs
# ---------------------------------------------------------------------------

def config_labels(df: pd.DataFrame) -> np.ndarray:
    """One label per row, e.g. 'N6_T0', from the integer N and T columns."""
    return ("N" + df["N"].astype(int).astype(str) + "_T" + df["T"].astype(int).astype(str)).to_numpy()


def _coerce_config_column(series: pd.Series, name: str, valid: tuple) -> pd.Series:
    """Numeric N / T values.  A trailing '!' (modified-mesh marker in PD files, e.g. '0!') is dropped."""
    cleaned = series.astype(str).str.strip().str.rstrip("!")
    vals = pd.to_numeric(cleaned, errors="coerce")
    bad = vals.isna() | ~vals.isin(valid)
    if bad.any():
        shown = sorted({str(v) for v in series[bad].tolist()})[:5]
        raise ValueError(f"Column '{name}' must contain only {valid}; found {shown}.")
    return vals.astype(int)


def read_pd_csv(paths) -> pd.DataFrame:
    """
    Read one or several processed-data CSVs into one frame with columns FEATURE_COLS + TARGET_COLS.

    * Rows with NaN in R, A, CC, VC, SEA or CFE are dropped.
    * A missing N column defaults to 6 and a missing T column to 0 (legacy campaign).
    * N must be in {0, 6} and T in {0, 1}; anything else raises ValueError.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]

    frames = []
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Data file not found: {path}")
        df = pd.read_csv(path)
        missing = [c for c in LEGACY_FEATURE_COLS + TARGET_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{path}: missing required column(s) {missing}")

        df = df.dropna(subset=LEGACY_FEATURE_COLS + TARGET_COLS).copy()
        for col, valid in (("N", VALID_N), ("T", VALID_T)):
            if col in df.columns:
                df[col] = _coerce_config_column(df[col], col, valid)
            else:
                df[col] = LEGACY_CONFIG_DEFAULTS[col]
                print(f"[data_loader] {os.path.basename(str(path))}: no '{col}' column - "
                      f"assuming {col}={LEGACY_CONFIG_DEFAULTS[col]} (legacy family).")
        frames.append(df[FEATURE_COLS + TARGET_COLS])

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def _stratify_labels(labels: np.ndarray, test_frac: float, what: str):
    """Labels to stratify on, or None (with a warning) when stratification is impossible."""
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        return None
    n_test = math.ceil(test_frac * len(labels))
    if counts.min() < 2 or n_test < len(classes):
        print(f"[data_loader] WARNING: cannot stratify the {what} split by configuration "
              f"(counts: {dict(zip(classes.tolist(), counts.tolist()))}); using a plain random split.")
        return None
    return labels


def _split(X, Y, labels, test_frac, seed, what):
    return train_test_split(
        X, Y, labels,
        test_size=test_frac,
        random_state=seed,
        stratify=_stratify_labels(labels, test_frac, what),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(
    seed: int = 42,
    lf_val_frac: float = 0.20,
    hf_test_frac: float = 0.20,
    lf_csv=None,
    hf_csv=None,
) -> DataSplits:
    """
    Load both fidelity CSVs, create train/val/test splits, and standardize.

    Split protocol
    --------------
    LF  (e.g. 938 pts): 80 % train / 20 % validation
        Used for MLP Phase-1 pre-training and GP LF fitting.

    HF  (e.g. 123 pts): 80 % fine-tune / 20 % test
        Fine-tune set is used in MLP Phase-2 and GP correction fitting.
        The test set is held out for the final comparison only.

    Both splits are stratified by (N, T) configuration when possible.

    Parameters
    ----------
    seed         : random seed (default 42 for reproducibility)
    lf_val_frac  : fraction of LF reserved for validation
    hf_test_frac : fraction of HF reserved as the final test set
    lf_csv       : path or list of paths of LF processed-data CSVs (default: LF_CSV)
    hf_csv       : path or list of paths of HF processed-data CSVs (default: HF_CSV)

    Returns
    -------
    DataSplits with all standardized arrays and fitted scalers.
    """
    lf_df = read_pd_csv(LF_CSV if lf_csv is None else lf_csv)
    hf_df = read_pd_csv(HF_CSV if hf_csv is None else hf_csv)

    X_lf = lf_df[FEATURE_COLS].values.astype(float)
    Y_lf = lf_df[TARGET_COLS].values.astype(float)
    X_hf = hf_df[FEATURE_COLS].values.astype(float)
    Y_hf = hf_df[TARGET_COLS].values.astype(float)
    C_lf = config_labels(lf_df)
    C_hf = config_labels(hf_df)

    X_lf_tr, X_lf_val, Y_lf_tr, Y_lf_val, C_lf_tr, C_lf_val = _split(
        X_lf, Y_lf, C_lf, lf_val_frac, seed, "LF"
    )
    X_hf_ft, X_hf_tst, Y_hf_ft, Y_hf_tst, C_hf_ft, C_hf_tst = _split(
        X_hf, Y_hf, C_hf, hf_test_frac, seed, "HF"
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
        config_lf_train=C_lf_tr,
        config_lf_val=C_lf_val,
        config_hf_finetune=C_hf_ft,
        config_hf_test=C_hf_tst,
        feature_cols=list(FEATURE_COLS),
    )

    print(
        f"[data_loader] LF: {len(X_lf_tr)} train / {len(X_lf_val)} val  |  "
        f"HF: {len(X_hf_ft)} fine-tune / {len(X_hf_tst)} test  |  seed={seed}"
    )
    for name, labels in (("LF", C_lf), ("HF", C_hf)):
        uniq, cnt = np.unique(labels, return_counts=True)
        print(f"[data_loader] {name} configurations: " + ", ".join(f"{u}={c}" for u, c in zip(uniq, cnt)))
    return splits


# ---------------------------------------------------------------------------
# Scalers and model-directory bookkeeping
# ---------------------------------------------------------------------------

def save_scalers(splits: DataSplits, models_dir: str = MODELS_DIR) -> str:
    """Persist the two scalers (and the feature order) to <models_dir>/scalers.pkl.  Returns the path."""
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, SCALERS_FILE)
    joblib.dump(
        {
            "x_scaler": splits.x_scaler,
            "y_scaler": splits.y_scaler,
            "feature_cols": list(splits.feature_cols or FEATURE_COLS),
        },
        path,
    )
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


def load_feature_cols(models_dir: str = MODELS_DIR) -> list[str]:
    """Feature order the models in `models_dir` were trained with (legacy 4 columns if not recorded)."""
    path = os.path.join(models_dir, SCALERS_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Scalers not found at {path}.  Run train_mlp or train_gp first.")
    d = joblib.load(path)
    return list(d.get("feature_cols", LEGACY_FEATURE_COLS))


def existing_model_n_features(models_dir: str = MODELS_DIR) -> int | None:
    """Input width of the model artifacts already in `models_dir`, or None if there are none."""
    path = os.path.join(models_dir, SCALERS_FILE)
    if not os.path.isfile(path):
        return None
    try:
        return int(joblib.load(path)["x_scaler"].n_features_in_)
    except Exception:                       # unreadable / foreign file: do not guess
        return None


def guard_model_dir(models_dir: str, n_features: int, force: bool = False) -> None:
    """
    Refuse to overwrite trained artifacts whose input width differs from the new run
    (e.g. a legacy 4-input model in `models/`) unless `force` is set.
    """
    existing = existing_model_n_features(models_dir)
    if existing is not None and existing != n_features and not force:
        raise SystemExit(
            f"[data_loader] {models_dir} already holds a {existing}-input model, but this run trains a "
            f"{n_features}-input model.\n"
            f"  Use a different --models-dir to keep the existing artifacts, or pass --force to overwrite them."
        )
