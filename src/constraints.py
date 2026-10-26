"""
Geometric feasibility constraints per (N, T) configuration.

Single source of truth used by sample.py, check_constraints.py, stp_preflight.py
and the tests.  Nothing else in the repo should re-implement these inequalities.

Configuration
-------------
N : 6 = hexagonal cells (traditional),  0 = ellipsoidal holes
T : 0 = untwisted part,                 1 = twist-angle part

A configuration is written (N, T) everywhere in this repo, e.g. (6, 0).

Constraint sets  (OD = outer diameter, L = length, G = gap;  every inequality is
strict, so a point exactly on a boundary is infeasible)
---------------------------------------------------------------------------
(6, 0)   C1   R < pi*OD / (sqrt(3)*CC)
         C2   R < (1/VC)*(L/2 - G) + (sqrt(3)*pi*OD / (12*CC)) * (1 - 1/VC)
         C3   R < (L - 2G) / (VC + 1)

(0, 0)   E1   R < (L - 2G) / (VC + 1)
         E2   R < pi*OD / (2*CC)
         E3   4R^2 < (pi*OD / (2*CC))^2 + ((L - 2(R + G)) / (VC - 1))^2

(6, 1)   C1   as (6, 0)
         C4'  cot(A) <= (-sqrt(3)*Vsep + 2*sqrt(3)*R - pi*OD/(2*CC)) / (Vsep - 2R)   OR   Vsep > 2R
              Vsep = (L - 2(R + G)) / (VC - 1)           (A in degrees; Vsep = 2R makes the ratio undefined -> violated)
         C5   R < (L - 2G)/(VC - 1) * 1 / ( -(cot(A)/(sqrt(3) + cot(A)) - 1) + 2/(VC - 1) )      (vertical packing)
              (at A = 90 deg C5 is exactly the old C3, R < (L - 2G)/(VC + 1); it loosens as A decreases)
         C2, C3 and the old minimum-tilt C4 (A > arctan((v + 2R)/((4*sqrt(3)/3)*v - 2*sqrt(3)*R))) no longer apply to
         (6, 1) (2026-10-16): the rebuild sweep showed them conservative.  C4' replaces the old C4; C5 replaces C3.

(0, 1)   PLACEHOLDER - constraints not derived yet

E3 history: it was first written with ">" (neighbouring holes must overlap).  Real SolidWorks rebuilds of
N0AShell.SLDPRT showed the opposite (see tests/ and src/validate_constraints.py): with ">" the set accepted no
buildable design; with "<" (neighbouring holes do not overlap) every point accepted by E1+E2+E3 built.  It was
flipped on 2026-09-30.  With the flip all of E1, E2, E3 are upper bounds on R.

Placeholder configurations are registered (so they are visible and easy to fill
in) but are treated as *undefined*: they are never sampled and never pass the
STP pre-flight.  To define one, replace its ``None`` entry in CONSTRAINT_SETS
with a tuple of ``Constraint`` objects - everything downstream picks it up.

Domain guard (not one of the derived inequalities)
--------------------------------------------------
A row is violated by *every* constraint of its set when R, CC, VC (and A, when
present) are not all finite, or when R, CC or VC is <= 0.  E3 additionally needs
VC > 1 (E3, C4' and C5 divide by VC - 1) and C4' needs Vsep != 2R.  Undefined
arithmetic therefore reads as "violated" and never raises.  C4' and C5 are the
only constraints that use A; evaluating a (6, 1) row without an A column is an error.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

PI    = math.pi
SQRT3 = math.sqrt(3.0)

# ---------------------------------------------------------------------------
# Design space
# ---------------------------------------------------------------------------

PARAM_COLS  = ["R", "A", "CC", "VC"]
CONFIG_COLS = ["T", "N"]

# Same box for every configuration (assumption - override per config here if the
# ellipsoidal parts get different ranges).
PARAM_BOUNDS = {
    "R":  (2.0,  8.8),
    "A":  (30.0, 90.0),
    "CC": (4,    22),
    "VC": (4,    10),
}

VALID_N = (0, 6)
VALID_T = (0, 1)

LEGACY_CONFIG = (6, 0)
ALL_CONFIGS   = ((6, 0), (6, 1), (0, 0), (0, 1))   # (N, T); order fixes seed offsets


class ConstraintsNotDefinedError(NotImplementedError):
    """Raised when a configuration is a placeholder without derived constraints."""


# ---------------------------------------------------------------------------
# Constraint definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Constraint:
    """One inequality.  ``fn(R, A, CC, VC, OD, L, G)`` returns True where the constraint is satisfied."""

    name: str
    expr: str
    fn: Callable
    uses_A: bool = False        # True for constraints that depend on the angle A (an A column is then required)

    def satisfied(self, R, A, CC, VC, OD: float, L: float, G: float) -> np.ndarray:
        """Vectorised check with the domain guard applied (see module docstring)."""
        if self.uses_A and A is None:
            raise ValueError(f"Constraint {self.name} depends on A, but no A values were supplied.")
        R  = np.atleast_1d(np.asarray(R,  dtype=float))
        CC = np.atleast_1d(np.asarray(CC, dtype=float))
        VC = np.atleast_1d(np.asarray(VC, dtype=float))
        if A is not None:
            A = np.atleast_1d(np.asarray(A, dtype=float))
        with np.errstate(all="ignore"):
            guard = (
                np.isfinite(R) & np.isfinite(CC) & np.isfinite(VC)
                & (R > 0) & (CC > 0) & (VC > 0)
            )
            if A is not None:
                guard &= np.isfinite(A)
            return np.asarray(self.fn(R, A, CC, VC, OD, L, G), dtype=bool) & guard


def _c1(R, A, CC, VC, OD, L, G):
    return R < PI * OD / (SQRT3 * CC)


def _c2(R, A, CC, VC, OD, L, G):
    return R < (1.0 / VC) * (L / 2.0 - G) + (SQRT3 * PI * OD / (12.0 * CC)) * (1.0 - 1.0 / VC)


def _c3(R, A, CC, VC, OD, L, G):
    return R < (L - 2.0 * G) / (VC + 1.0)


def _e2(R, A, CC, VC, OD, L, G):
    return R < PI * OD / (2.0 * CC)


def _e3(R, A, CC, VC, OD, L, G):
    h  = PI * OD / (2.0 * CC)
    vs = (L - 2.0 * (R + G)) / (VC - 1.0)
    return (VC > 1) & (4.0 * R ** 2 < h ** 2 + vs ** 2)          # neighbouring holes do not overlap


def _cot_deg(A):
    """cot(A) for A in degrees (A = 90 -> ~0; A = 0 or 180 -> +-inf, which the callers treat as violated)."""
    return 1.0 / np.tan(np.radians(A))


def _c4p(R, A, CC, VC, OD, L, G):
    """Twisted hex, replaces the minimum tilt:  cot(A) <= (-sqrt3*Vsep + 2*sqrt3*R - pi*OD/(2CC)) / (Vsep - 2R)  OR  Vsep > 2R."""
    vsep = (L - 2.0 * (R + G)) / (VC - 1.0)
    den  = vsep - 2.0 * R
    cot  = _cot_deg(A)
    bound = (-SQRT3 * vsep + 2.0 * SQRT3 * R - PI * OD / (2.0 * CC)) / den
    return (VC > 1) & ((vsep > 2.0 * R) | ((den != 0) & (cot <= bound)))


def _c5(R, A, CC, VC, OD, L, G):
    """Twisted hex, vertical packing:  R < (L-2G)/(VC-1) / ( 1 - cot(A)/(sqrt3 + cot(A)) + 2/(VC-1) )."""
    cot = _cot_deg(A)
    den = -(cot / (SQRT3 + cot) - 1.0) + 2.0 / (VC - 1.0)
    return (VC > 1) & (den > 0) & (R < (L - 2.0 * G) / (VC - 1.0) / den)


C1 = Constraint("C1", "R < pi*OD/(sqrt(3)*CC)", _c1)
C2 = Constraint("C2", "R < (1/VC)(L/2-G) + (sqrt(3)*pi*OD/(12*CC))(1-1/VC)", _c2)
C3 = Constraint("C3", "R < (L-2G)/(VC+1)", _c3)

E1 = Constraint("E1", "R < (L-2G)/(VC+1)", _c3)
E2 = Constraint("E2", "R < pi*OD/(2*CC)", _e2)
E3 = Constraint("E3", "4R^2 < (pi*OD/(2*CC))^2 + ((L-2(R+G))/(VC-1))^2", _e3)

C4P = Constraint("C4'", "cot(A) <= (-sqrt(3)*Vsep + 2*sqrt(3)*R - pi*OD/(2*CC))/(Vsep - 2R)  OR  Vsep > 2R,  Vsep = (L-2(R+G))/(VC-1)", _c4p, uses_A=True)
C5 = Constraint("C5", "R < (L-2G)/(VC-1) / (1 - cot(A)/(sqrt(3)+cot(A)) + 2/(VC-1))", _c5, uses_A=True)

# (N, T) -> constraints.  None = PLACEHOLDER (not derived yet).
CONSTRAINT_SETS: dict[tuple[int, int], tuple[Constraint, ...] | None] = {
    (6, 0): (C1, C2, C3),
    (0, 0): (E1, E2, E3),
    (6, 1): (C1, C4P, C5),
    (0, 1): None,   # TODO: derive the twisted-ellipsoid constraints
}


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def config_label(config: tuple[int, int]) -> str:
    """(6, 0) -> 'N6_T0'  (used in state files and reports)."""
    return f"N{int(config[0])}_T{int(config[1])}"


def implemented_configs() -> list[tuple[int, int]]:
    """Configurations that have derived constraints, in ALL_CONFIGS order."""
    return [c for c in ALL_CONFIGS if CONSTRAINT_SETS.get(c) is not None]


def constraints_for(config: tuple[int, int]) -> tuple[Constraint, ...]:
    """Constraint tuple for a configuration; raises for placeholders / unknown configs."""
    config = (int(config[0]), int(config[1]))
    if config not in CONSTRAINT_SETS:
        raise ValueError(f"Unknown configuration (N={config[0]}, T={config[1]}); "
                         f"N must be in {VALID_N} and T in {VALID_T}.")
    cset = CONSTRAINT_SETS[config]
    if cset is None:
        raise ConstraintsNotDefinedError(
            f"Constraints for (N={config[0]}, T={config[1]}) have not been derived yet "
            f"(placeholder in src/constraints.py::CONSTRAINT_SETS)."
        )
    return cset


def require_implemented(configs: Iterable[tuple[int, int]]) -> None:
    """Raise ConstraintsNotDefinedError if any configuration is a placeholder."""
    for c in configs:
        constraints_for(c)


_CONFIG_RE = re.compile(r"^N?(\d+)[:_]?T?(\d+)$")


def parse_configs(spec: str | Sequence) -> list[tuple[int, int]]:
    """
    Parse a configuration list.

    Accepts a comma separated string - ``"6:0,0:0"``, ``"N6T0,N0T0"``, ``"N6_T0"`` - or
    the keywords ``"all"`` (every implemented configuration) and ``"legacy"`` ((6, 0)).
    A sequence of (N, T) pairs is validated and returned as-is.  Duplicates are removed
    (order kept).  Placeholder configurations are rejected with
    ConstraintsNotDefinedError.
    """
    if isinstance(spec, str):
        items = [s.strip().upper() for s in spec.split(",") if s.strip()]
        if not items:
            raise ValueError("Empty configuration list.")
        configs: list[tuple[int, int]] = []
        for item in items:
            if item == "ALL":
                configs.extend(implemented_configs())
            elif item == "LEGACY":
                configs.append(LEGACY_CONFIG)
            else:
                m = _CONFIG_RE.match(item)
                if not m:
                    raise ValueError(f"Cannot parse configuration '{item}'. "
                                     "Use N:T (e.g. 6:0), NxTy (e.g. N6T0), 'all' or 'legacy'.")
                configs.append((int(m.group(1)), int(m.group(2))))
    else:
        configs = [(int(n), int(t)) for n, t in spec]

    out: list[tuple[int, int]] = []
    for c in configs:
        if c not in CONSTRAINT_SETS:
            raise ValueError(f"Unknown configuration (N={c[0]}, T={c[1]}); "
                             f"N must be in {VALID_N} and T in {VALID_T}.")
        if c not in out:
            out.append(c)
    require_implemented(out)
    return out


def config_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Integer ``N`` and ``T`` columns aligned to df.index.

    Missing columns default to the legacy configuration (N=6, T=0).  Values outside
    N in {0, 6} / T in {0, 1} (or NaN) raise ValueError.
    """
    n = len(df)
    out = pd.DataFrame(index=df.index)
    for col, valid, default in (("N", VALID_N, LEGACY_CONFIG[0]), ("T", VALID_T, LEGACY_CONFIG[1])):
        if col not in df.columns:
            out[col] = np.full(n, default, dtype=np.int64)
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        bad = vals.isna() | ~vals.isin(valid)
        if bad.any():
            shown = sorted({str(v) for v in df.loc[bad, col].tolist()})[:5]
            raise ValueError(f"Column '{col}' must contain only {valid}; found {shown}.")
        out[col] = vals.astype(np.int64)
    return out[["N", "T"]]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _check_constants(OD: float, L: float, G: float) -> None:
    for name, val in (("OD", OD), ("L", L), ("G", G)):
        if not np.isfinite(val):
            raise ValueError(f"{name} must be finite, got {val}.")
    if OD <= 0 or L <= 0:
        raise ValueError(f"OD and L must be positive (OD={OD}, L={L}).")
    if G < 0:
        raise ValueError(f"G must be >= 0, got {G}.")


def constraint_names() -> list[str]:
    """Names of every constraint in every implemented set, in registry order."""
    names: list[str] = []
    for c in ALL_CONFIGS:
        for con in CONSTRAINT_SETS.get(c) or ():
            if con.name not in names:
                names.append(con.name)
    return names


def evaluate(df: pd.DataFrame, OD: float, L: float, G: float = 0.0) -> pd.DataFrame:
    """
    Evaluate every row against the constraint set of its own (N, T) configuration.

    Requires columns R, CC, VC (A is optional).  ``N`` / ``T`` are optional and default
    to (6, 0).

    Returns a frame indexed like ``df`` with
        N, T      integer configuration
        <name>    one nullable-boolean column per constraint (C1, C2, C3, C4', C5, E1, E2, E3):
                  True = satisfied, False = violated, <NA> = not applicable to the row
        defined   False for placeholder configurations (constraints not derived)
        feasible  True only if the configuration is defined and every applicable
                  constraint is satisfied
    """
    _check_constants(OD, L, G)
    missing = [c for c in ("R", "CC", "VC") if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    cfg = config_columns(df)
    n = len(df)
    R  = df["R"].to_numpy(dtype=float)
    CC = df["CC"].to_numpy(dtype=float)
    VC = df["VC"].to_numpy(dtype=float)
    A  = df["A"].to_numpy(dtype=float) if "A" in df.columns else None

    out = cfg.copy()
    columns = {name: pd.array([pd.NA] * n, dtype="boolean") for name in constraint_names()}
    defined  = np.zeros(n, dtype=bool)
    feasible = np.zeros(n, dtype=bool)

    for (cn, ct), cset in CONSTRAINT_SETS.items():
        rows = ((cfg["N"] == cn) & (cfg["T"] == ct)).to_numpy()
        if not rows.any() or cset is None:
            continue
        defined[rows] = True
        ok = np.ones(int(rows.sum()), dtype=bool)
        A_rows = None if A is None else A[rows]
        for con in cset:
            sat = con.satisfied(R[rows], A_rows, CC[rows], VC[rows], OD, L, G)
            columns[con.name][rows] = sat
            ok &= sat
        feasible[rows] = ok

    for name, col in columns.items():
        out[name] = col
    out["defined"]  = defined
    out["feasible"] = feasible
    return out


def feasible_mask(df: pd.DataFrame, OD: float, L: float, G: float = 0.0) -> pd.Series:
    """Boolean Series: row is in a defined configuration and satisfies all its constraints."""
    return evaluate(df, OD, L, G)["feasible"]


def is_feasible(N: int, T: int, R: float, A: float, CC: float, VC: float,
                OD: float, L: float, G: float = 0.0) -> bool:
    """Scalar convenience wrapper around evaluate()."""
    row = pd.DataFrame([{"R": R, "A": A, "CC": CC, "VC": VC, "T": T, "N": N}])
    return bool(evaluate(row, OD, L, G)["feasible"].iloc[0])
