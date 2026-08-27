"""
PyTorch MLP surrogate model for joint prediction of SEA and CFE.

Architecture
------------
Input  (4)  →  Hidden-1 (64) + tanh  →  Hidden-2 (32) + tanh
            →  Hidden-3 (16) + tanh  →  Output (2, linear)

Total parameters: ~2,962  (see docs/MLP_DESIGN.md for full justification)

Design decisions
----------------
Activation  : tanh — smooth, continuously differentiable, well-suited for
              regression on physical simulation data that varies smoothly with
              geometry parameters.  (Glorot et al., 2010; LeCun et al., 1998)

Initialization: Xavier uniform (Glorot & Bengio, 2010) — derived for tanh;
              sets initial weight magnitudes so the variance of activations is
              preserved across layers, preventing early vanishing/exploding gradients.

Transfer learning: Trained in two phases.
  Phase 1 — Pre-train on 938 LF (shell) samples.
  Phase 2 — Freeze the first N_FREEZE hidden layers and fine-tune the remaining
             layers on 98 HF (solid) samples at a lower learning rate.
             (Yosinski et al., 2014; Pan & Yang, 2010)

References
----------
Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep
feedforward neural networks. AISTATS.

LeCun, Y., Bottou, L., Orr, G., & Müller, K.R. (1998). Efficient backprop.
Neural Networks: Tricks of the Trade. Springer.

Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How transferable are
features in deep neural networks? NeurIPS 27.

Pan, S.J., & Yang, Q. (2010). A survey on transfer learning. IEEE TKDE, 22(10).
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import torch
import torch.nn as nn

# Default architecture and transfer-learning freeze depth
LAYER_SIZES         = [4, 64, 32, 16, 2]
N_FREEZE_FOR_FINETUNE = 2   # freeze first 2 hidden layers in Phase 2


class SurrogateNet(nn.Module):
    """
    Fully connected MLP with configurable depth, tanh activations, and
    Xavier uniform weight initialization.

    Parameters
    ----------
    layer_sizes : list[int]
        Neuron counts per layer (input → hidden... → output).
        Default [4, 64, 32, 16, 2] gives 3 hidden layers for this problem.
    """

    def __init__(self, layer_sizes: list[int] = LAYER_SIZES):
        super().__init__()
        self.layer_sizes = layer_sizes

        # Hidden layers stored in a ModuleList for clean index-based freezing
        self.hidden_layers = nn.ModuleList(
            nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 2)
        )
        self.output_layer = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialization — optimal variance preservation for tanh."""
        for layer in self.hidden_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = torch.tanh(layer(x))
        return self.output_layer(x)   # linear output (no activation)

    # ------------------------------------------------------------------
    # Transfer learning helpers
    # ------------------------------------------------------------------

    def freeze_until(self, n_hidden: int) -> None:
        """
        Freeze the first n_hidden hidden layers.

        Frozen layers have requires_grad=False for all their parameters.
        They receive no gradient updates during backpropagation, preserving
        the features learned during Phase-1 pre-training.

        Calling freeze_until(2) on the default [4,64,32,16,2] architecture
        freezes Hidden-1 and Hidden-2 (4→64 and 64→32) and leaves Hidden-3
        and the output layer fully trainable.  This exposes 578 trainable
        parameters for the 98-sample HF fine-tune split — a ratio ~6× larger
        than the output-only case, while still protecting most pretrained
        knowledge.  (Yosinski et al., 2014)
        """
        self.unfreeze_all()
        for i in range(min(n_hidden, len(self.hidden_layers))):
            for p in self.hidden_layers[i].parameters():
                p.requires_grad = False

    def unfreeze_all(self) -> None:
        """Enable gradients on all parameters."""
        for p in self.parameters():
            p.requires_grad = True

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def frozen_layers(self) -> list[int]:
        """Return indices of hidden layers that are currently frozen."""
        return [
            i for i, layer in enumerate(self.hidden_layers)
            if not next(layer.parameters()).requires_grad
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str, metadata: dict | None = None) -> None:
        """
        Save model weights (state_dict) to path.

        A companion JSON file (same stem, .json extension) stores the
        architecture and any caller-supplied metadata (training config, metrics,
        etc.) for fully self-documenting model artifacts.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.state_dict(), path)

        meta = {
            "layer_sizes":   self.layer_sizes,
            "total_params":  self.total_params(),
            "saved_at":      datetime.now().isoformat(timespec="seconds"),
        }
        if metadata:
            meta.update(metadata)

        json_path = os.path.splitext(path)[0] + ".json"
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[mlp_model] Saved -> {path}  ({self.total_params():,} total params)")

    @classmethod
    def load(cls, path: str) -> "SurrogateNet":
        """
        Load a saved SurrogateNet from path.

        Architecture is inferred from the companion .json metadata file, so
        you can load models with non-default architectures without supplying
        layer_sizes explicitly.
        """
        json_path = os.path.splitext(path)[0] + ".json"
        if not os.path.isfile(json_path):
            raise FileNotFoundError(
                f"Metadata JSON not found: {json_path}\n"
                "Ensure the model was saved with SurrogateNet.save()."
            )
        with open(json_path) as f:
            meta = json.load(f)

        model = cls(layer_sizes=meta["layer_sizes"])
        model.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )
        model.eval()
        print(f"[mlp_model] Loaded <- {path}  (saved {meta.get('saved_at', 'unknown')})")
        return model
