"""
Pre-flight for STEP (.stp) generation: decide, before SolidWorks is touched, which design
points may be built and where each one goes.

SwGen (Automation/SwGen) opens ONE part per run and binds every CSV column to a global
variable of that part.  A mixed (N, T) point list therefore has to be

  * checked against the constraints of each point's own configuration,
  * split into one CSV per configuration, and
  * stripped of the T and N columns (they are not equations in any part - SwGen aborts
    with "no defining equation" if it sees them).

This module does that and mirrors SwGen's naming rules, so batches can be verified
without SolidWorks:

    part file   N<n>[T]A<Fidelity>.SLDPRT        N6AShell, N6TAShell, N0AShell, N0TAShell (or ...Solid)
    STEP stem   <part><R><A><CC><VC>             N6AShellR340A5861CC1800VC400
                each value written as round(value * 100), half-to-even (Commands.cs FileSegment);
                the segment order follows the CSV header order, which is R, A, CC, VC here.
                (the angle label is always "A", also for twisted parts)

A point is rejected (never sent to CAD) when a column is missing / non-numeric / non-finite,
(N, T) is not a valid configuration, a parameter lies outside PARAM_BOUNDS, CC or VC is not a
whole number, it violates the constraints of its configuration, its configuration has no
derived constraints yet (T = 1 placeholders), or its file name collides with an earlier row.

Usage:
    python src/stp_preflight.py points.csv --OD 40 --L 50 --G 3 --fidelity Shell --out batches/run1
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from constraints import (
    LEGACY_CONFIG,
    PARAM_BOUNDS,
    PARAM_COLS,
    VALID_N,
    VALID_T,
    config_label,
    constraint_names,
    evaluate,
)

FIDELITIES = ("Shell", "Solid")
_NAME_SCALE = 100                         # SwGen: round(value * 100)
_DEFAULT_SWGEN = os.path.join("Automation", "SwGen", "bin", "Release", "net48", "SwGen.exe")


# ---------------------------------------------------------------------------
# Naming (mirrors SwGen)
# ---------------------------------------------------------------------------

def _as_config(N, T) -> tuple[int, int]:
    if N not in VALID_N or T not in VALID_T:
        raise ValueError(f"Invalid configuration (N={N!r}, T={T!r}): N must be in {VALID_N} and T in {VALID_T}.")
    return int(N), int(T)


def part_name(N, T, fidelity: str = "Shell") -> str:
    """'N6TAShell' for (N=6, T=1, Shell); the angle variable is 'TA' in twisted parts, 'A' otherwise."""
    if fidelity not in FIDELITIES:
        raise ValueError(f"fidelity must be one of {FIDELITIES}, got {fidelity!r}.")
    N, T = _as_config(N, T)
    return f"N{N}{'TA' if T else 'A'}{fidelity}"


def part_file(N, T, fidelity: str = "Shell") -> str:
    """'N6TAShell.SLDPRT'."""
    return part_name(N, T, fidelity) + ".SLDPRT"


def stp_stem(N, T, fidelity, R, A, CC, VC) -> str:
    """
    File name (without extension) SwGen gives the STEP file for this point.

    round() on a Python float is half-to-even on the exact binary value, which is what
    .NET's Math.Round(x, MidpointRounding.ToEven) does for the same double.
    """
    segments = []
    for label, value in (("R", R), ("A", A), ("CC", CC), ("VC", VC)):
        v = float(value)
        if not math.isfinite(v) or v < 0:
            raise ValueError(f"{label}={value!r} cannot be encoded in a file name.")
        segments.append(f"{label}{round(v * _NAME_SCALE)}")
    return part_name(N, T, fidelity) + "".join(segments)


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

@dataclass
class PreflightResult:
    accepted: pd.DataFrame            # row, N, T, R, A, CC, VC, part, stem
    rejected: pd.DataFrame            # row, N, T, R, A, CC, VC, reasons
    notes: list = field(default_factory=list)


def preflight(points: pd.DataFrame, OD: float, L: float, G: float = 0.0, fidelity: str = "Shell") -> PreflightResult:
    """
    Split `points` into accepted / rejected rows (see the module docstring for the checks).

    Needs the columns R, A, CC, VC.  T and N are optional; without them every point is the
    legacy (N=6, T=0) configuration.  ``row`` is the 1-based data row of the input, the same
    numbering SwGen uses.  Raises ValueError only for structural problems (missing required
    column, unknown fidelity); bad *values* become rejections with a reason.
    """
    if fidelity not in FIDELITIES:
        raise ValueError(f"fidelity must be one of {FIDELITIES}, got {fidelity!r}.")
    missing = [c for c in PARAM_COLS if c not in points.columns]
    if missing:
        raise ValueError(f"points is missing required column(s): {missing}")

    n = len(points)
    rows = np.arange(1, n + 1)
    notes: list[str] = []
    reasons: list[list[str]] = [[] for _ in range(n)]

    defaults = {"N": LEGACY_CONFIG[0], "T": LEGACY_CONFIG[1]}
    num: dict[str, np.ndarray] = {}
    for col in PARAM_COLS + ["T", "N"]:
        if col in points.columns:
            num[col] = pd.to_numeric(points[col], errors="coerce").to_numpy(dtype=float)
        else:
            num[col] = np.full(n, float(defaults[col]))
            notes.append(f"column '{col}' absent - assuming {col}={defaults[col]} (legacy configuration)")
    finite = {col: np.isfinite(vals) for col, vals in num.items()}

    for col in num:
        for i in np.where(~finite[col])[0]:
            reasons[i].append(f"{col} is missing or not a finite number")

    valid_cfg = np.isin(num["N"], VALID_N) & np.isin(num["T"], VALID_T)
    for i in np.where(finite["N"] & finite["T"] & ~valid_cfg)[0]:
        reasons[i].append(f"invalid configuration N={num['N'][i]:g}, T={num['T'][i]:g} "
                          f"(N must be in {VALID_N}, T in {VALID_T})")

    for col in PARAM_COLS:
        lo, hi = PARAM_BOUNDS[col]
        outside = finite[col] & ((num[col] < lo) | (num[col] > hi))
        for i in np.where(outside)[0]:
            reasons[i].append(f"{col}={num[col][i]:g} outside [{lo}, {hi}]")
    for col in ("CC", "VC"):
        fractional = finite[col] & (num[col] != np.round(num[col]))
        for i in np.where(fractional)[0]:
            reasons[i].append(f"{col}={num[col][i]:g} is not an integer")

    # Constraints - only for rows that are numerically sane and in a valid configuration.
    sane = valid_cfg.copy()
    for col in num:
        sane &= finite[col]
    idx = np.where(sane)[0]
    if len(idx):
        frame = pd.DataFrame({col: num[col][idx] for col in num})
        report = evaluate(frame, OD, L, G)
        names = constraint_names()
        for k, i in enumerate(idx):
            if not report["defined"].iloc[k]:
                reasons[i].append(f"no constraints derived yet for (N={int(num['N'][i])}, T={int(num['T'][i])}) "
                                  "- placeholder configuration")
            elif not report["feasible"].iloc[k]:
                violated = [nm for nm in names
                            if pd.notna(report[nm].iloc[k]) and not bool(report[nm].iloc[k])]
                reasons[i].append("violates " + ", ".join(violated))

    # File-name collisions among the rows that survived everything else (SwGen would fail the later row).
    stems: list[str | None] = [None] * n
    first_seen: dict[str, int] = {}
    for i in range(n):
        if reasons[i]:
            continue
        stem = stp_stem(int(num["N"][i]), int(num["T"][i]), fidelity,
                        num["R"][i], num["A"][i], num["CC"][i], num["VC"][i])
        if stem in first_seen:
            reasons[i].append(f"file name {stem} collides with row {first_seen[stem]} "
                              "(values equal after rounding to 0.01)")
        else:
            first_seen[stem] = int(rows[i])
            stems[i] = stem

    base = pd.DataFrame({"row": rows, "N": num["N"], "T": num["T"], **{c: num[c] for c in PARAM_COLS}})
    ok = np.array([not r for r in reasons], dtype=bool) if n else np.zeros(0, dtype=bool)

    accepted = base[ok].copy()
    for col in ("N", "T", "CC", "VC"):
        accepted[col] = accepted[col].astype(int)
    accepted["part"] = [part_file(N, T, fidelity) for N, T in zip(accepted["N"], accepted["T"])]
    accepted["stem"] = [stems[i] for i in np.where(ok)[0]]

    rejected = base[~ok].copy()
    rejected["reasons"] = ["; ".join(reasons[i]) for i in np.where(~ok)[0]]

    return PreflightResult(accepted.reset_index(drop=True), rejected.reset_index(drop=True), notes)


# ---------------------------------------------------------------------------
# SwGen hand-off
# ---------------------------------------------------------------------------

def write_swgen_batches(result: PreflightResult, out_dir: str) -> list[dict]:
    """
    Write one CSV per configuration containing ONLY R, A, CC, VC (never T / N), plus
    rejected.csv when anything was rejected.

    Returns one dict per configuration: config (N, T), part (SLDPRT file name), csv, n, rows
    (1-based input rows in the CSV's order).
    """
    os.makedirs(out_dir, exist_ok=True)
    batches: list[dict] = []
    for (N, T), group in result.accepted.groupby(["N", "T"], sort=True):
        csv_path = os.path.join(out_dir, f"{config_label((N, T))}_points.csv")
        group[PARAM_COLS].to_csv(csv_path, index=False)
        batches.append({
            "config": (int(N), int(T)),
            "part": str(group["part"].iloc[0]),
            "csv": csv_path,
            "n": int(len(group)),
            "rows": [int(r) for r in group["row"]],
        })
    if len(result.rejected):
        result.rejected.to_csv(os.path.join(out_dir, "rejected.csv"), index=False)
    return batches


def swgen_command(batch: dict, parts_dir: str, out_dir: str, swgen_exe: str = _DEFAULT_SWGEN) -> str:
    """The SwGen command line that builds one batch (printed by the CLI, never executed here)."""
    step_dir = os.path.join(out_dir, f"step_{config_label(batch['config'])}")
    return (f'"{swgen_exe}" generate --part "{os.path.join(parts_dir, batch["part"])}" '
            f'--csv "{batch["csv"]}" --out "{step_dir}"')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check design points and split them into per-configuration SwGen batches "
                    "(R, A, CC, VC only).")
    parser.add_argument("csv", help="Points CSV with columns R, A, CC, VC and optionally T, N")
    parser.add_argument("--OD", type=float, required=True, help="Outer diameter constant")
    parser.add_argument("--L",  type=float, required=True, help="Length constant")
    parser.add_argument("--G", "--gap", type=float, default=0.0, help="Gap constant (default 0)")
    parser.add_argument("--fidelity", choices=FIDELITIES, default="Shell",
                        help="Which part family the STEP files are built from (default Shell)")
    parser.add_argument("--out", default=None, help="Output folder (default: <csv name>_stp next to the CSV)")
    parser.add_argument("--parts-dir", default=".", help="Folder holding the SLDPRT files (used in the printed commands)")
    parser.add_argument("--swgen", default=_DEFAULT_SWGEN, help="Path to SwGen.exe (used in the printed commands)")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.csv):
        print(f"ERROR: File not found: {args.csv}")
        return 2
    out_dir = args.out or os.path.splitext(args.csv)[0] + "_stp"

    try:
        result = preflight(pd.read_csv(args.csv), args.OD, args.L, args.G, args.fidelity)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    batches = write_swgen_batches(result, out_dir)

    for note in result.notes:
        print(f"NOTE: {note}")
    print(f"Accepted {len(result.accepted)} / {len(result.accepted) + len(result.rejected)} point(s); output in {out_dir}")
    for b in batches:
        print(f"  {config_label(b['config'])}: {b['n']} point(s) -> {b['part']}   ({b['csv']})")
    if len(result.rejected):
        print(f"Rejected {len(result.rejected)} point(s) (see {os.path.join(out_dir, 'rejected.csv')}):")
        for _, r in result.rejected.head(20).iterrows():
            print(f"  row {int(r['row'])}: {r['reasons']}")
        if len(result.rejected) > 20:
            print(f"  ... and {len(result.rejected) - 20} more")
    if batches:
        print("\nRun each batch (one part per run):")
        for b in batches:
            print("  " + swgen_command(b, args.parts_dir, out_dir, args.swgen))

    return 0 if len(result.rejected) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
