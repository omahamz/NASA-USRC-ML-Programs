"""
MLP training pipeline: Phase 1 (LF pre-training) and Phase 2 (HF fine-tuning).

Usage
-----
From the project root:

    python -m src.ml_models.train_mlp                  # run both phases
    python -m src.ml_models.train_mlp --phase 1        # Phase 1 only
    python -m src.ml_models.train_mlp --phase 2        # Phase 2 only (Phase 1 must exist)
    python -m src.ml_models.train_mlp --seed 0         # different random seed
    python -m src.ml_models.train_mlp --epochs-p1 800  # override Phase-1 epoch cap

Saved artifacts
---------------
models/
  mlp_pretrained_lf.pt   + .json   Phase-1 weights and metadata
  mlp_finetuned_hf.pt    + .json   Phase-2 weights and metadata
  scalers.pkl                       StandardScaler objects (shared with GP)
  plots/
    LC_Phase1_MLP.png               Phase-1 learning curves
    LC_Phase2_MLP.png               Phase-2 learning curves
    Parity_MLP_LF.png               LF-only parity plot on HF test set
    Parity_MLP_TL.png               Transfer-learning parity plot on HF test set
"""

from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .data_loader import (
    DataSplits,
    MODELS_DIR,
    TARGET_COLS,
    load_data,
    save_scalers,
)
from .evaluate import (
    compute_metrics,
    learning_curve_plot,
    parity_plot,
    print_metrics_table,
    save_metrics_json,
)
from .mlp_model import LAYER_SIZES, N_FREEZE_FOR_FINETUNE, SurrogateNet

# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------

# Phase 1 — LF pre-training
P1_EPOCHS     = 600
P1_LR         = 1e-3
P1_WD         = 1e-4     # L2 weight decay
P1_BATCH      = 64
P1_PATIENCE   = 50       # early-stopping patience (epochs without val improvement)

# Phase 2 — HF fine-tuning
P2_EPOCHS     = 300
P2_LR         = 1e-4     # lower LR: preserve pre-trained representation
P2_WD         = 1e-3     # stronger WD: regularise on the smaller HF dataset
P2_BATCH      = None     # None → full-batch (all HF fine-tune samples per step)
P2_PATIENCE   = 30
P2_FREEZE     = N_FREEZE_FOR_FINETUNE   # number of hidden layers to freeze


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train_loop(
    model: SurrogateNet,
    optimizer: torch.optim.Optimizer,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    epochs: int,
    patience: int,
    batch_size: int | None,
    phase_label: str,
) -> dict:
    """
    Full training loop with mini-batch SGD (or full-batch), early stopping,
    and per-output loss monitoring.

    Returns a history dict with per-epoch losses for plotting.
    """
    device = next(model.parameters()).device
    loss_fn = nn.MSELoss()

    X_tr_t = torch.FloatTensor(X_train).to(device)
    Y_tr_t = torch.FloatTensor(Y_train).to(device)
    X_vl_t = torch.FloatTensor(X_val).to(device)
    Y_vl_t = torch.FloatTensor(Y_val).to(device)

    bs = batch_size if batch_size is not None else len(X_train)
    loader = DataLoader(
        TensorDataset(X_tr_t, Y_tr_t),
        batch_size=bs,
        shuffle=True,
    )

    best_val   = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    wait       = 0

    history = {
        "epochs": [], "train_total": [], "val_total": [],
        "train_sea": [], "train_cfe": [],
    }

    for epoch in range(epochs):
        model.train()
        tr_sea_acc = tr_cfe_acc = 0.0
        n_seen = 0

        for X_b, Y_b in loader:
            optimizer.zero_grad()
            pred = model(X_b)
            loss_sea = loss_fn(pred[:, 0], Y_b[:, 0])
            loss_cfe = loss_fn(pred[:, 1], Y_b[:, 1])
            loss = loss_sea + loss_cfe
            loss.backward()
            optimizer.step()
            tr_sea_acc += loss_sea.item() * len(X_b)
            tr_cfe_acc += loss_cfe.item() * len(X_b)
            n_seen += len(X_b)

        tr_sea = tr_sea_acc / n_seen
        tr_cfe = tr_cfe_acc / n_seen
        tr_tot = tr_sea + tr_cfe

        model.eval()
        with torch.no_grad():
            vl_pred = model(X_vl_t)
            vl_tot  = loss_fn(vl_pred, Y_vl_t).item()

        # Log every 25 epochs and at the final epoch
        if (epoch + 1) % 25 == 0 or epoch == epochs - 1:
            history["epochs"].append(epoch + 1)
            history["train_total"].append(tr_tot)
            history["val_total"].append(vl_tot)
            history["train_sea"].append(tr_sea)
            history["train_cfe"].append(tr_cfe)
            print(
                f"  [{phase_label}] ep {epoch+1:4d}  "
                f"train={tr_tot:.5f} (SEA={tr_sea:.5f} CFE={tr_cfe:.5f})  "
                f"val={vl_tot:.5f}"
            )

        if vl_tot < best_val:
            best_val   = vl_tot
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            print(
                f"  [{phase_label}] Early stopping at epoch {epoch+1}. "
                f"Best val loss {best_val:.5f} at epoch {best_epoch}."
            )
            break

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["best_val"]   = best_val
    return history


# ---------------------------------------------------------------------------
# Phase wrappers
# ---------------------------------------------------------------------------

def run_phase1(
    splits: DataSplits,
    models_dir: str = MODELS_DIR,
    epochs: int = P1_EPOCHS,
    lr: float = P1_LR,
    wd: float = P1_WD,
    batch_size: int = P1_BATCH,
    patience: int = P1_PATIENCE,
    layer_sizes: list[int] = LAYER_SIZES,
) -> SurrogateNet:
    """
    Pre-train the MLP on the 750-point LF (shell) training split.

    The model learns the mapping (R, A, CC, VC) → (SEA, CFE) from shell
    simulation data.  All layers are trainable; Adam optimizer with moderate
    L2 regularization.

    Returns the best-validation-loss model (weights restored after early stopping).
    """
    print("\n" + "=" * 60)
    print("PHASE 1 - Low-Fidelity Pre-Training")
    print("=" * 60)

    model = SurrogateNet(layer_sizes=layer_sizes)
    print(
        f"  Architecture : {layer_sizes}\n"
        f"  Total params : {model.total_params():,}\n"
        f"  LR={lr}  WD={wd}  batch={batch_size}  patience={patience}"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    history = _train_loop(
        model, optimizer,
        splits.X_lf_train, splits.Y_lf_train,
        splits.X_lf_val,   splits.Y_lf_val,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        phase_label="Phase1",
    )

    # Evaluate LF model on the HF test set (baseline — no transfer)
    model.eval()
    with torch.no_grad():
        Y_pred_std = model(torch.FloatTensor(splits.X_hf_test)).numpy()
    Y_pred = splits.y_scaler.inverse_transform(Y_pred_std)
    Y_true = splits.y_scaler.inverse_transform(splits.Y_hf_test)
    m = compute_metrics(Y_true, Y_pred)
    print(f"\n  [Phase1] HF test:  SEA R2={m['SEA']['R2']:.4f}  CFE R2={m['CFE']['R2']:.4f}")

    # Save model
    save_path = os.path.join(models_dir, "mlp_pretrained_lf.pt")
    model.save(
        save_path,
        metadata={
            "phase": 1,
            "n_lf_train": len(splits.X_lf_train),
            "epochs_run": history["best_epoch"],
            "best_val_loss": history["best_val"],
            "hf_test_R2_SEA": m["SEA"]["R2"],
            "hf_test_R2_CFE": m["CFE"]["R2"],
            "hyperparams": {"lr": lr, "wd": wd, "batch_size": batch_size},
        },
    )

    # Save learning curve
    learning_curve_plot(history, "LC_Phase1_MLP")

    return model


def run_phase2(
    splits: DataSplits,
    source_model: SurrogateNet | None = None,
    models_dir: str = MODELS_DIR,
    epochs: int = P2_EPOCHS,
    lr: float = P2_LR,
    wd: float = P2_WD,
    batch_size: int | None = P2_BATCH,
    patience: int = P2_PATIENCE,
    n_freeze: int = P2_FREEZE,
) -> SurrogateNet:
    """
    Fine-tune the pre-trained MLP on the 98-point HF (solid) fine-tune split.

    The first n_freeze hidden layers are frozen (requires_grad=False).  Only the
    last hidden layer and the output layer are updated — they adapt the pre-trained
    geometric representation to the higher-fidelity solid simulation physics.

    source_model : if None, loads mlp_pretrained_lf.pt from models_dir.
    """
    print("\n" + "=" * 60)
    print("PHASE 2 - High-Fidelity Fine-Tuning")
    print("=" * 60)

    if source_model is None:
        pt_path = os.path.join(models_dir, "mlp_pretrained_lf.pt")
        source_model = SurrogateNet.load(pt_path)

    model = copy.deepcopy(source_model)
    model.freeze_until(n_freeze)

    print(
        f"  Frozen layers : {model.frozen_layers()}  "
        f"({model.total_params() - model.trainable_params():,} params frozen)\n"
        f"  Trainable     : {model.trainable_params():,} params\n"
        f"  LR={lr}  WD={wd}  batch={'full' if batch_size is None else batch_size}  patience={patience}"
    )

    # Only pass trainable parameters to the optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=wd,
    )

    # Use a small portion of the HF fine-tune set as an internal val set for
    # early stopping during Phase 2 (80 % finetune train / 20 % finetune val)
    n_ft = len(splits.X_hf_finetune)
    n_ft_val = max(1, int(n_ft * 0.20))
    X_ft_tr  = splits.X_hf_finetune[n_ft_val:]
    Y_ft_tr  = splits.Y_hf_finetune[n_ft_val:]
    X_ft_val = splits.X_hf_finetune[:n_ft_val]
    Y_ft_val = splits.Y_hf_finetune[:n_ft_val]

    history = _train_loop(
        model, optimizer,
        X_ft_tr, Y_ft_tr,
        X_ft_val, Y_ft_val,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        phase_label="Phase2",
    )

    # Evaluate fine-tuned model on the HF test set
    model.eval()
    with torch.no_grad():
        Y_pred_std = model(torch.FloatTensor(splits.X_hf_test)).numpy()
    Y_pred = splits.y_scaler.inverse_transform(Y_pred_std)
    Y_true = splits.y_scaler.inverse_transform(splits.Y_hf_test)
    m = compute_metrics(Y_true, Y_pred)
    print(f"\n  [Phase2] HF test:  SEA R2={m['SEA']['R2']:.4f}  CFE R2={m['CFE']['R2']:.4f}")

    # Save
    save_path = os.path.join(models_dir, "mlp_finetuned_hf.pt")
    model.save(
        save_path,
        metadata={
            "phase": 2,
            "n_hf_finetune": n_ft,
            "frozen_layers": model.frozen_layers(),
            "trainable_params": model.trainable_params(),
            "epochs_run": history["best_epoch"],
            "best_val_loss": history["best_val"],
            "hf_test_R2_SEA": m["SEA"]["R2"],
            "hf_test_R2_CFE": m["CFE"]["R2"],
            "hyperparams": {"lr": lr, "wd": wd, "n_freeze": n_freeze},
        },
    )

    # Save learning curve
    learning_curve_plot(history, "LC_Phase2_MLP")

    return model


# ---------------------------------------------------------------------------
# Comparison helper (both phases already run)
# ---------------------------------------------------------------------------

def compare_phases(
    splits: DataSplits,
    model_lf: SurrogateNet,
    model_hf: SurrogateNet,
) -> dict:
    """
    Evaluate and compare LF-only vs Transfer-Learning MLP on the HF test set.
    Saves parity plots and a metrics JSON.
    """
    Y_true = splits.y_scaler.inverse_transform(splits.Y_hf_test)

    all_metrics: dict[str, dict] = {}
    for label, model in [("MLP (LF only)", model_lf), ("MLP (TL)", model_hf)]:
        model.eval()
        with torch.no_grad():
            Y_pred_std = model(torch.FloatTensor(splits.X_hf_test)).numpy()
        Y_pred = splits.y_scaler.inverse_transform(Y_pred_std)
        all_metrics[label] = compute_metrics(Y_true, Y_pred)
        parity_plot(Y_true, Y_pred, f"Parity_{label.replace(' ', '_').replace('(', '').replace(')', '')}")

    print_metrics_table(all_metrics)

    models_dir = os.path.dirname(
        os.path.join(MODELS_DIR, "mlp_pretrained_lf.pt")
    )
    save_metrics_json(all_metrics, os.path.join(models_dir, "mlp_comparison_metrics.json"))
    return all_metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    splits = load_data(seed=args.seed)
    save_scalers(splits, models_dir=args.models_dir)

    model_lf = model_hf = None

    if args.phase in (1, "both"):
        model_lf = run_phase1(
            splits,
            models_dir=args.models_dir,
            epochs=args.epochs_p1,
        )

    if args.phase in (2, "both"):
        model_hf = run_phase2(
            splits,
            source_model=model_lf,
            models_dir=args.models_dir,
            epochs=args.epochs_p2,
        )

    if model_lf is not None and model_hf is not None:
        compare_phases(splits, model_lf, model_hf)
    elif model_hf is not None:
        # Phase 2 only: load LF model for comparison
        pt_path = os.path.join(args.models_dir, "mlp_pretrained_lf.pt")
        if os.path.isfile(pt_path):
            model_lf = SurrogateNet.load(pt_path)
            compare_phases(splits, model_lf, model_hf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the MLP surrogate model.")
    parser.add_argument(
        "--phase", default="both",
        help="Which phase to run: 1, 2, or both (default: both)"
    )
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--epochs-p1", type=int,   default=P1_EPOCHS)
    parser.add_argument("--epochs-p2", type=int,   default=P2_EPOCHS)
    parser.add_argument("--models-dir", default=MODELS_DIR)
    args = parser.parse_args()

    # Coerce --phase to int if numeric
    try:
        args.phase = int(args.phase)
    except ValueError:
        pass

    main(args)
