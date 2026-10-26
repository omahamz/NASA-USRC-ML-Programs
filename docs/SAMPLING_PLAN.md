# Sampling Plan — v2 training data for the SEA / CFE surrogates

Plan for regenerating all training data for the surrogate `f(N, T, R, A, CC, VC) = (SEA, CFE)` after the
change to the finite element method. **All earlier LF/HF data (the 936 shell / 247 solid (6,0) files) is
obsolete. Do not mix it with v2 data.**

## Decisions at a glance

| Item | Decision |
|---|---|
| Design | Scrambled Sobol, one independent 4-D stream (R, A, CC, VC) per (N, T) configuration |
| Configurations | (6,0), (6,1), (0,0). (0,1) gets added to the same instance once its constraints are derived |
| Constants | OD = 40, L = 50, G = 3 |
| Domain | The full `PARAM_BOUNDS` box, filtered by each configuration's constraints, **without `--propagate`** |
| LF (shell) | `--sobol 1024`: 4096 raw draws per configuration, every feasible point kept (~1300–1500 per configuration) |
| HF (solid) | The **first** LF points of each stream: the feasible points of the first 1024 draws (~320–380 per configuration) |
| HF test set | A separate instance with a different seed: 256 draws per configuration (~80–90 per configuration), solid only, never trained on |
| Seeds | LF/HF instance `20261024`, test instance `20261025` |
| Instance files | `src/src_data/Sample_OD40L50G3_v2.{csv,sobol_state.json}` and `src/src_data/Sample_OD40L50G3_v2_test.{csv,sobol_state.json}` |

### Expected counts (dry run with the seeds above)

| Configuration | Part (shell / solid) | LF points (4096 draws) | HF points (first 1024 draws) | HF stage 1 (first 512 draws) | Test points (256 draws) |
|---|---|---|---|---|---|
| (6,0) `N6_T0` | `N6AShell` / `N6ASolid` | 1522 | 382 | 191 | 91 |
| (6,1) `N6_T1` | `N6TAShell` / `N6TASolid` | 1395 | 345 | 170 | 86 |
| (0,0) `N0_T0` | `N0AShell` / `N0ASolid` | 1294 | 322 | 157 | 78 |
| **Total** | | **4211 shell runs** | **1049 solid runs** | 518 | **255 solid runs** |

**Run 1 (2026-10-26)** generated only (6,0) and (0,0), adding `--configs 6:0,0:0` to the two `sample.py` commands
below. (6,1) was left out until its constraints are final. It joins the same instance later with
`--configs 6:1` and gets exactly the stream shown above. The driver is `runs/v2/run_swgen.py` (local, git-ignored).
N0 solid STEP files turned out to be 5–8.5 MB each, not the ~1.8 MB of N6 solids.

These numbers are deterministic for the given seeds and the current constraints. Re-running the commands below
reproduces them exactly. If they differ, the constraints or bounds have changed (see [Rules](#rules-that-keep-the-sample-clean)).

LF:HF ≈ 4:1. That leans further toward HF than the usual ~10:1, on purpose: the shell model predicted HF CFE poorly
(R² ≈ 0.04 with LF only), so CFE accuracy depends on solid runs.

---

## 1. Why these numbers

### Design space

- Per configuration the surrogate sees 4 inputs: R and A are continuous (rounded to 0.01; SolidWorks stores
  angles with 2 decimals), and CC (4–22) and VC (4–10) are integers. That gives 19 × 7 = 133 (CC, VC) combinations, and **all 133 are
  feasible in every configuration**.
- N and T are not sampling dimensions. Each (N, T) pair is a separate part family with its own constraints, so
  it gets its own Sobol stream. This is the "sliced design" approach for mixed qualitative and quantitative inputs.
- **A matters in every configuration.** At T=0 the horizontal layers alternate twist direction, so the geometry
  changes with A. As A approaches 90°, T=0 and T=1 geometries with otherwise equal variables converge. A model trained
  on all configurations together can share information near A = 90° (see [Modelling notes](#5-modelling-notes)).
- Feasible share of the box: (6,0) ~37 %, (6,1) ~34 %, (0,0) ~32 %.

### Sample size

| Guideline | What it gives here | Source |
|---|---|---|
| ~10 runs per input dimension for an *initial* GP experiment | 40 per configuration (60 pooled). This is a floor, not enough for a strong surrogate | Loeppky, Sacks & Welch (2009) |
| If 10·d is inaccurate, the response is rough or noisy and extra runs help less than expected | Applies to CFE (noisy peak force) | Harari et al. (2018) |
| Multi-fidelity works best with LF ≈ 10× HF, spending 10–80 % of the budget on LF | 4:1 here, chosen on purpose for CFE | WCSMO 2015 framework comparison; Southampton multi-fidelity Kriging guidelines |
| Coverage of the rarest (CC, VC) combination (~0.2 % of the feasible volume) | ~3 LF points in the rarest combination at ~1400 per configuration. Models interpolate across CC/VC, so this is acceptable | dry run of the constraints |
| MLP samples-per-parameter | ~4200 LF points / 3090 parameters ≈ 1.4 (the old data gave 0.24) | `ML_PLAN.md` |
| GP O(n³) cost | ~4200 LF points pooled: ~140 MB covariance matrix, minutes per optimizer restart. Above ~5–8k, switch to one GP per configuration or a sparse GP | — |

**There is no rule that gives an exact count.** Measure it with a learning curve (section 4) and extend the
instance if error is still falling at 4096 draws.

### Why Sobol, and why the stream always ends on a power of 2

- A Sobol sequence keeps its balance (low-discrepancy) properties only at lengths 2^m. Thinning out points or
  skipping part of the stream breaks that.
- `generate_sobol` therefore always extends each raw stream to a power-of-2 length and keeps **every** feasible point.
  `K` (`--sobol K`) is a *minimum*. The sample is always exactly *all feasible points of the first 2^m draws*.
- The first 2^j draws of a 2^m stream are themselves a balanced Sobol sample. That gives two free properties:
  - **HF = a balanced leading subset of LF.** Nesting comes automatically: every HF point has an LF point at the same
    location, which the GP correction step (Kennedy–O'Hagan) and the MLP fine-tuning both need.
  - **Any run interrupted at a prefix boundary is still a balanced design.** Learning curves come for free.
- `Sample_OD40L50G3_v2.sobol_state.json` stores `prefixes = {"2^j": rows}` for every configuration. The first
  `rows` rows of that configuration in the CSV are exactly the feasible points of the first 2^j draws.
- Integers are mapped by inverse CDF. 19 and 7 levels don't line up with Sobol's power-of-2 structure, so there is a slight
  marginal imbalance between CC/VC levels. That is acceptable.

---

## 2. Instructions

All commands run from the repository root. Python uses the project venv (`.venv`).

### 2.1 Generate the designs

```bash
python src/sample.py --sobol 1024 --OD 40 --L 50 --G 3 --seed 20261024 --sobol-instance src/src_data/Sample_OD40L50G3_v2
```

```bash
python src/sample.py --sobol 64 --OD 40 --L 50 --G 3 --seed 20261025 --sobol-instance src/src_data/Sample_OD40L50G3_v2_test
```

Expected output: `[N6_T0] 1522 new valid points ... stream 0 -> 4096 draws`, `[N6_T1] 1395`, `[N0_T0] 1294`, and for
the test set 91 / 86 / 78. The test points have no row in common with the training points.

Check both:

```bash
python src/check_constraints.py src/src_data/Sample_OD40L50G3_v2.csv --OD 40 --L 50 --G 3
```

```bash
python src/check_constraints.py src/src_data/Sample_OD40L50G3_v2_test.csv --OD 40 --L 50 --G 3
```

Both should print `All rows satisfy their constraints.` The instance CSV and state JSON are sample designs (inputs
only), which the repository tracks. Commit them so the streams can be resumed on any machine.

### 2.2 Split into SwGen batches

The pre-flight writes one `R,A,CC,VC` CSV per configuration (`N6_T0_points.csv`, ...) **in the same row order as
the instance CSV**, so row *i* of a batch is the *i*-th Sobol point of that configuration.

```bash
python src/stp_preflight.py src/src_data/Sample_OD40L50G3_v2.csv --OD 40 --L 50 --G 3 --fidelity Shell --out runs/v2/shell
```

```bash
python src/stp_preflight.py src/src_data/Sample_OD40L50G3_v2.csv --OD 40 --L 50 --G 3 --fidelity Solid --out runs/v2/solid
```

```bash
python src/stp_preflight.py src/src_data/Sample_OD40L50G3_v2_test.csv --OD 40 --L 50 --G 3 --fidelity Solid --out runs/v2/test_solid
```

`rejected.csv` should be absent or empty. `runs/` is git-ignored.

### 2.3 Build the STEP files

SwGen: `Automation/SwGen/bin/Release/net48/SwGen.exe`. Close SolidWorks before a batch, run one part per
call, and use the part folders of the current FE model (as of writing: `...\SolidWorksStuff\3mmGap4mmThickNoFillet\{Shell,Solid}`
for N6 parts and `...\SolidWorksStuff\3mmGap4mmThickN0\{Shell,Solid}` for N0 parts). A 120-point batch takes ~8–10 min.
`--skip-existing` resumes an interrupted batch.

**LF (shell): every row, in order.**

```bash
Automation/SwGen/bin/Release/net48/SwGen.exe generate --part "<NoFillet>/Shell/N6AShell.SLDPRT" --csv runs/v2/shell/N6_T0_points.csv --out runs/v2/shell/step_N6_T0 --skip-existing
```

Repeat with `N6TAShell` + `N6_T1_points.csv` and `N0AShell` (N0 folder) + `N0_T0_points.csv`.

**HF (solid): only the leading rows**, using `--start` / `--top`. Stage 1 = the first 512 draws; stage 2 extends to the
first 1024 draws:

| Configuration | Solid part | Stage 1: `--start 1 --top` | Stage 2: `--start` | Stage 2: `--top` | HF total |
|---|---|---|---|---|---|
| `N6_T0` | `N6ASolid` | 191 | 192 | 191 | 382 |
| `N6_T1` | `N6TASolid` | 170 | 171 | 175 | 345 |
| `N0_T0` | `N0ASolid` | 157 | 158 | 165 | 322 |

```bash
Automation/SwGen/bin/Release/net48/SwGen.exe generate --part "<NoFillet>/Solid/N6ASolid.SLDPRT" --csv runs/v2/solid/N6_T0_points.csv --out runs/v2/solid/step_N6_T0 --start 1 --top 191
```

Always read the row counts from `prefixes` in the state file rather than from this table (they are the same for the seeds above).
For example, `prefixes["1024"]` = HF rows, and `prefixes["512"]` = stage-1 rows.

**HF test set (solid): every row** of `runs/v2/test_solid/*_points.csv`. A shell run of the test points is optional
(useful to measure the LF→HF gap on unseen points, but not required for training).

### 2.4 Simulate and record

Run the new FE method on every STEP file, in batch order where possible, so that partial results always cover a prefix.
Process the results into the usual processed-data CSVs (`R, A, CC, VC, T, N, ..., SEA, CFE`) under `data_folder/`
(git-ignored, confidential). Keep LF, HF and test in separate files.

**Failed rebuilds or simulations:** record them and leave them out. **Do not backfill** with later points of the
stream. A few holes barely affect balance, but substitute points break the prefix property. If many points in one region fail,
the constraints there are wrong: fix the constraints and start a new instance.

---

## 3. Run order and staging

1. Shell, all three configurations, first 1024 draws (`--top prefixes["1024"]`). This is a balanced LF set to start with.
2. Solid HF stage 1 (first 512 draws, ~520 runs) plus the full test set (~255 runs).
3. The rest of the shell runs (to 2048, then 4096 draws), with `--start` continuing from the last row.
4. Solid HF stage 2 (to 1024 draws).

After each stage, train and evaluate on the HF test set. Steps 1 and 2 already give a usable multi-fidelity model.

---

## 4. Deciding whether more data is needed (learning curve)

The prefixes give balanced training subsets for free. For each configuration take the first `prefixes["1024"]`,
`["2048"]` and `["4096"]` LF rows (and the first `["256"]`, `["512"]`, `["1024"]` HF rows) and train on each. Then plot
HF-test RMSE / R² against the number of points, separately for SEA and CFE.

- Error still falling at 4096 LF draws: extend the instance to 8192 (section 6).
- LF curve flat but HF-test error still falling as HF grows: add HF (extend to the first 2048 draws) instead of LF.
- CFE flat at a poor value: likely the noise floor of the CFE measure (peak force is sensitive to mesh and contact),
  not a data shortage. Look at the CFE definition and filtering before buying more runs.

---

## 5. Modelling notes

- **Train one model across configurations.** The MLP takes N and T as inputs, and pooling lets T=0 and T=1
  share information, especially near A = 90°, where their geometries converge.
- **GP and T.** With T as a plain ARD input, the GP correlation between a T=0 and a T=1 point is the same at every A, so it
  cannot express "identical at 90°, different at 30°". If the GP underperforms on the twisted parts, use a feature such as
  `T·cos(A)` (0 for both families at A = 90°) instead of T, or fit one GP per configuration.
- **Sanity check.** At A ≈ 90°, predictions for (6,0) and (6,1) with equal R, CC and VC should nearly agree. Their constraint
  sets differ ((6,0) has C2, (6,1) does not), so their sampled regions don't fully overlap there.
- **Constraints are conservative.** SolidWorks rebuilds showed many buildable points outside them. The surrogate is only
  trusted inside the constraint-defined region, so the optimizer must stay inside it.
- **Boundaries are not enriched.** High-SEA designs may sit on active constraints (large R; C1/C2). Points are spread evenly
  inside the feasible region with no extra density at the edges. If the optimum lands on a boundary, add a targeted batch
  there (a separate file, not part of the Sobol instance).

---

## 6. Extending the sample later

Re-run the **same command** (same instance, OD/L/G, no `--propagate`). The stored seed is reused automatically, and each
stream continues exactly where it stopped and doubles to the next power of 2:

```bash
python src/sample.py --sobol 1024 --OD 40 --L 50 --G 3 --sobol-instance src/src_data/Sample_OD40L50G3_v2
```

- From 4096 draws, any `--sobol K` ≤ 4096 goes to 8192 draws (another ~1300–1500 points per configuration). Doubling
  is the smallest clean extension.
- New rows are appended to the CSV, and `prefixes` gains the new boundary. Re-run the pre-flight and give SwGen
  `--start <previous row count + 1>`.
- **Adding (0,1)** once its constraints exist: run the same command with `--configs 0:1`. It starts its own stream at 0 in the
  same instance. Use `--sobol 1024` to match the other configurations.
- **HF extension** (from the learning curve): build the solid parts for rows `prefixes["1024"] + 1` to `prefixes["2048"]`.

### Rules that keep the sample clean

- Never edit, re-order or de-duplicate rows of the instance CSV by hand. Never delete the state JSON.
- Keep OD, L, G, the configuration list, `--propagate` (off) and `PARAM_BOUNDS` unchanged for the lifetime of the instance.
- **If the constraints of a configuration change, start a new instance** (or a new configuration stream). Existing rows were filtered with
  the old rules, so the "all feasible points of the first 2^m draws" property would no longer hold.
- The test instance is never extended into training data, and its seed must never equal the training seed.

---

## Sources

- Loeppky, Sacks & Welch (2009), *Choosing the Sample Size of a Computer Experiment: A Practical Guide*, Technometrics 51 —
  [tech report](https://www.stat.ubc.ca/technical-reports-archive/doc/238.pdf)
- Harari et al. (2018), Statistica Sinica — [pdf](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A28n216.pdf)
- SciPy `qmc.Sobol` (balance properties require 2^m points; no thinning or skipping) —
  [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html)
- Multi-fidelity Kriging budget guidelines (University of Southampton) — [eprint](https://eprints.soton.ac.uk/373482)
- Comparison of multi-fidelity frameworks, LF ≈ 10× HF (WCSMO 2015) —
  [abstract](https://web.aeromech.usyd.edu.au:443/WCSMO2015/abstracts/1056.pdf)
- Review of multi-fidelity surrogates, including cases where extra LF data hurts (UF) — [pdf](https://web.mae.ufl.edu/nkim/Papers/paper85.pdf)
- Qian & Wu (2009), *Sliced space-filling designs*, Biometrika — [summary](https://academicnewsletter.sufe.edu.cn/info/448994)
- Yang, Lin, Zhou & He, *Doubly coupled designs* — [arXiv 2203.06335](https://arxiv.org/abs/2203.06335)
