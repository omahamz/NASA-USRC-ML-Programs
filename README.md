# NASA USRC | Multi-Fidelity ML Surrogate Models

Machine learning surrogate models for crashworthiness prediction of thin-walled
energy-absorbing structures, developed as part of NASA's University Student
Research Challenge (USRC).

Two surrogate models, a **Multi-Layer Perceptron (MLP)** and a **Multi-Output
Gaussian Process (GP)**, are trained to predict:

- **SEA**, Specific Energy Absorption (higher is better)
- **CFE**, Crushing Force Efficiency (closer to 1 is better)

from six design inputs:

| Input | Type  | Description                                              |
|-------|-------|----------------------------------------------------------|
| `R`   | float | Corner radius (mm)                                       |
| `A`   | float | Angle (degrees)                                          |
| `CC`  | int   | Cell count                                               |
| `VC`  | int   | Volume coefficient                                       |
| `T`   | 0 / 1 | Twist: 0 = untwisted part, 1 = twist-angle part          |
| `N`   | 0 / 6 | Cell shape: 6 = hexagonal cells, 0 = ellipsoidal holes   |

`T` and `N` select the part file (`N6AShell`, `N6TAShell`, `N0AShell`, `N0TAShell`, and the
`...Solid` counterparts). Data from before the `(N, T)` extension has no `N` column and a
constant `T = 0`; it is read as `N = 6, T = 0`.

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
├── pytest.ini              # Test configuration (SolidWorks tests are opt-in)
├── tests/                  # pytest suite: constraints, STP pre-flight, sampler, ML pipeline
├── src/
│   ├── constraints.py      # Geometric constraints per (N, T) configuration (single source of truth)
│   ├── sample.py           # Constraint-aware Sobol / LHS sampling, one stream per configuration
│   ├── check_constraints.py# Geometric constraint verification of a sample CSV
│   ├── stp_preflight.py    # Checks points and splits them into per-configuration SwGen batches
│   ├── validate_constraints.py # Compares constraints with real SolidWorks rebuilds (SwGen results)
│   ├── active_sampler.py   # Active learning / adaptive sampling (legacy 4-input, N=6 T=0)
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

`src/sample.py` generates Sobol (`--sobol K`) or Latin hypercube (`--k K`) sample designs subject to
the geometric manufacturability constraints of each part family. `T` and `N` are not extra
sampling dimensions: every `(N, T)` configuration gets its own 4-D stream and its own constraints
(`src/constraints.py`), and `K` means *K points per configuration*.

| Configuration | Constraints | Box acceptance |
|---|---|---|
| `(N=6, T=0)` | `C1`–`C3` | ~37 % |
| `(N=0, T=0)` | `E1`–`E3` | ~6 % |
| `(N=6, T=1)`, `(N=0, T=1)` | placeholders, not derived yet: never sampled, rejected by the pre-flight | – |

```bash
python src/sample.py --sobol 256 --OD 40 --L 50 --G 3 --seed 42 --propagate          # (6,0) and (0,0)
python src/sample.py --sobol 256 --OD 40 --L 50 --G 3 --configs 0:0 --sobol-instance Sample_N0   # one family, resumable
python src/check_constraints.py Sample_N0.csv --OD 40 --L 50 --G 3                    # verify a sample
python src/stp_preflight.py Sample_N0.csv --OD 40 --L 50 --G 3 --out batches/run1    # split for SwGen
```

Generated designs and their sampler state live in `src/src_data/`, these contain input
coordinates only, no simulation results. State files written before the `(N, T)` extension are
read as `(6, 0)` and migrated on the next resume.

## Tests

```bash
pip install -r requirements.txt
python -m pytest                    # constraints, STP pre-flight, sampler, ML pipeline (~30 s)
python -m pytest -m solidworks      # opt-in: drives SwGen/SolidWorks (needs SWGEN_PARTS_DIR, see the test file)
```

## Documentation

- [ML_PLAN.md](ML_PLAN.md), end-to-end modeling plan and rationale
- [docs/MLP_DESIGN.md](docs/MLP_DESIGN.md), MLP architecture and training design
- [docs/GP_DESIGN.md](docs/GP_DESIGN.md), GP design, kernels, and acquisition
- [docs/USAGE.md](docs/USAGE.md), practical usage guide
