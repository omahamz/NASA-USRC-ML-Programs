# NASA USRC | Multi-Fidelity ML Surrogate Models

Machine learning surrogate models for crashworthiness prediction of thin-walled
energy-absorbing structures, developed as part of NASA's University Student
Research Challenge (USRC).

Two surrogate models, a **Multi-Layer Perceptron (MLP)** and a **Multi-Output
Gaussian Process (GP)**, are trained to predict:

- **SEA**, Specific Energy Absorption (higher is better)
- **CFE**, Crushing Force Efficiency (closer to 1 is better)

from four structural design parameters:

| Input | Type  | Description        |
|-------|-------|--------------------|
| `R`   | float | Corner radius (mm) |
| `A`   | float | Angle (degrees)    |
| `CC`  | int   | Cell count         |
| `VC`  | int   | Volume coefficient |

Both models use a **multi-fidelity transfer-learning** strategy: they are
pre-trained on a large set of low-fidelity (shell) simulation results, then
adapted to a small set of high-fidelity (solid) simulations via a learned
correction. The goal is to compare the two approaches as simulation surrogates
and to drive design-space exploration and optimization.

> **Note on data:** The simulation datasets used to train these models are
> confidential and are **not** included in this repository (see
> [.gitignore](.gitignore)). Only source code and input-space sample designs
> (Sobol sequences) are tracked. Trained model artifacts are likewise excluded,
> since serialized GP models embed their training data.

## Repository Structure

```
├── main.py                 # Force–displacement post-processing (AUC, CFE, peak force)
├── compare.py              # Comparison utilities
├── ML_PLAN.md              # Full modeling plan: data, architecture, training phases
├── docs/
│   ├── USAGE.md            # How to train, evaluate, and predict
│   ├── MLP_DESIGN.md       # MLP architecture & transfer-learning design
│   └── GP_DESIGN.md        # GP kernel, correction model & acquisition design
├── src/
│   ├── sample.py           # Constraint-aware Sobol sampling of the design space
│   ├── check_constraints.py# Geometric constraint verification
│   ├── active_sampler.py   # Active learning / adaptive sampling
│   ├── optimizer.py        # Design optimization over the surrogates
│   ├── analysis.py         # Data analysis utilities
│   ├── visualizer.py       # Plotting tools
│   ├── src_data/           # Sobol sample designs (inputs only, tracked)
│   └── ml_models/
│       ├── data_loader.py  # Dataset loading, scaling, LF/HF splits
│       ├── mlp_model.py    # MLP architecture
│       ├── gp_model.py     # Multi-output GP + correction model
│       ├── train_mlp.py    # Two-phase MLP training pipeline
│       ├── train_gp.py     # Two-phase GP training pipeline
│       ├── evaluate.py     # Metrics & comparison plots
│       └── predict.py      # Inference API
└── Automation/             # Simulation pipeline automation experiments
```

## Getting Started

### Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Training

```bash
# Train the MLP (Phase 1: LF pre-train, Phase 2: HF fine-tune)
python -m src.ml_models.train_mlp

# Train the GP (LF fit + learned delta correction)
python -m src.ml_models.train_gp
```

Both pipelines support running a single phase (e.g. `--phase 1` / `--phase lf`);
see [docs/USAGE.md](docs/USAGE.md) for all options, expected training times, and
what to watch during training.

### Prediction

After training, use the inference API in `src/ml_models/predict.py` to evaluate
either surrogate at new design points. Trained artifacts (weights, scalers,
metrics, and plots) are written to `models/` locally.

## Design Space Sampling

`src/sample.py` generates Sobol sample designs subject to the geometric
manufacturability constraints of the structure (which eliminate roughly 65% of
the raw input space). Generated designs and their sampler state live in
`src/src_data/`, these contain input coordinates only, no simulation results.

## Documentation

- [ML_PLAN.md](ML_PLAN.md), end-to-end modeling plan and rationale
- [docs/MLP_DESIGN.md](docs/MLP_DESIGN.md), MLP architecture and training design
- [docs/GP_DESIGN.md](docs/GP_DESIGN.md), GP design, kernels, and acquisition
- [docs/USAGE.md](docs/USAGE.md), practical usage guide
