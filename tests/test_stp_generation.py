"""
Edge cases and constraints for STEP (.stp) generation.

Layers, from cheap to expensive
-------------------------------
1. Constraints per (N, T) configuration  (src/constraints.py)
     - every inequality at its boundary, checked against an independent scalar reference
     - undefined arithmetic, NaN / inf, configuration routing, placeholders
2. Pre-flight (src/stp_preflight.py): which points may reach SolidWorks
     - bounds, integrality, duplicates and file-name collisions, split per configuration
3. Naming contract shared with SwGen (part files, STEP file stems, analysis.py regex)
4. SwGen itself  - opt-in, needs SolidWorks:  pytest -m solidworks
     set SWGEN_EXE (default: Automation/SwGen/bin/Release/net48/SwGen.exe)
         SWGEN_PARTS_DIR (folder with N6AShell.SLDPRT, N0AShell.SLDPRT, ...)
         SWGEN_FIDELITY (Shell | Solid, default Shell)

Constants used throughout: OD = 40, L = 50, G = 3 (the values of the existing sample designs).
"""

import json
import math
import os
import re
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

import constraints as C
import sample as S
import stp_preflight as P

OD, L, G = 40.0, 50.0, 3.0
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HEX = (6, 0)
ELLIPSE = (0, 0)
TWIST_HEX = (6, 1)
IMPLEMENTED = [HEX, TWIST_HEX, ELLIPSE]


def feasible(N, T, R, A, CC, VC, OD=OD, L=L, G=G):
    return C.is_feasible(N, T, R, A, CC, VC, OD, L, G)


# A point that every implemented configuration accepts: R=3 is below all upper bounds at CC=12, VC=5
# (E3: 4.99, E2: 5.24, C2: 5.61) and for (6, 1) Vsep = 9.5 > 2R, so C4' holds and C5 (bound 8.8 at A=60) holds too.
BOTH_FEASIBLE = dict(R=3.0, A=60.0, CC=12, VC=5)


# ---------------------------------------------------------------------------
# Independent references (plain math, no numpy) and closed-form boundaries
# ---------------------------------------------------------------------------

def reference(N, T, R, A, CC, VC, OD=OD, L=L, G=G):
    """Per-constraint verdicts computed row-by-row with plain Python floats."""
    s3, pi = math.sqrt(3.0), math.pi
    if (N, T) == HEX:
        return {
            "C1": R < pi * OD / (s3 * CC),
            "C2": R < (1 / VC) * (L / 2 - G) + (s3 * pi * OD / (12 * CC)) * (1 - 1 / VC),
            "C3": R < (L - 2 * G) / (VC + 1),
        }
    if (N, T) == TWIST_HEX:
        vs = (L - 2 * (R + G)) / (VC - 1)
        cot = 1.0 / math.tan(math.radians(A))
        den = vs - 2 * R
        c4p = vs > 2 * R or (den != 0 and cot <= (-s3 * vs + 2 * s3 * R - pi * OD / (2 * CC)) / den)
        return {
            "C1": R < pi * OD / (s3 * CC),
            "C4'": bool(c4p),
            "C5": R < (L - 2 * G) / (VC - 1) / (-(cot / (s3 + cot) - 1) + 2 / (VC - 1)),
        }
    if (N, T) == ELLIPSE:
        h = pi * OD / (2 * CC)
        vs = (L - 2 * (R + G)) / (VC - 1)
        return {
            "E1": R < (L - 2 * G) / (VC + 1),
            "E2": R < h,
            "E3": 4 * R * R < h * h + vs * vs,          # neighbouring holes do not overlap
        }
    raise AssertionError((N, T))


def c4p_amin(R, CC, VC, OD=OD):
    """Smallest A (degrees) for which C4' holds when Vsep < 2R (independent of constraints.py): cot is decreasing in A,
    so cot(A) <= bound  <=>  A >= acot(bound) = atan2(1, bound).  nan where Vsep >= 2R (no tilt condition: > holds, = undefined)."""
    vs = (L - 2 * (R + G)) / (VC - 1)
    den = vs - 2 * R
    if den >= 0:
        return float("nan")
    bound = (-math.sqrt(3) * vs + 2 * math.sqrt(3) * R - math.pi * OD / (2 * CC)) / den
    return math.degrees(math.atan2(1.0, bound))


def c5_bound(A, VC):
    """Upper bound on R of C5 (plain math, independent of constraints.py)."""
    cot = 1.0 / math.tan(math.radians(A))
    return (L - 2 * G) / (VC - 1) / (-(cot / (math.sqrt(3) + cot) - 1) + 2 / (VC - 1))


def hex_bounds(CC, VC):
    """Upper bounds on R of the three (6, 0) constraints (same expressions as constraints.py)."""
    return {
        "C1": C.PI * OD / (C.SQRT3 * CC),
        "C2": (1.0 / VC) * (L / 2.0 - G) + (C.SQRT3 * C.PI * OD / (12.0 * CC)) * (1.0 - 1.0 / VC),
        "C3": (L - 2.0 * G) / (VC + 1.0),
    }


def e3_root(CC, VC):
    """R at which E3 holds with equality: 4R^2(m^2-1) + 4uR - (h^2 m^2 + u^2) = 0, m = VC-1, u = L-2G.
    E3 (non-overlap of neighbouring holes) holds for R below this root."""
    h = C.PI * OD / (2.0 * CC)
    u, m = L - 2.0 * G, VC - 1.0
    a = m * m - 1.0
    return (-u + math.sqrt(u * u + a * (h * h * m * m + u * u))) / (2.0 * a)


def ellipse_bounds(CC, VC):
    """E1, E2 and E3 are all upper bounds on R."""
    return {
        "E1": (L - 2.0 * G) / (VC + 1.0),
        "E2": C.PI * OD / (2.0 * CC),
        "E3": e3_root(CC, VC),
    }


def _grid():
    return [(CC, VC) for CC in range(4, 23) for VC in range(4, 11)]


def _binding(name, bounds_fn):
    """(CC, VC) where upper bound `name` is the tightest of its configuration's bounds by >= 2 %."""
    for CC, VC in _grid():
        b = bounds_fn(CC, VC)
        others = [v for k, v in b.items() if k != name]
        if b[name] > 2.0 and b[name] < 0.98 * min(others):
            return CC, VC
    raise AssertionError(f"no grid point where {name} is binding")


# ---------------------------------------------------------------------------
# 1. Constraints per configuration
# ---------------------------------------------------------------------------

class TestBoundaries:
    @pytest.mark.parametrize("name", ["C1", "C2", "C3"])
    def test_hex_constraint_boundary(self, name):
        CC, VC = _binding(name, hex_bounds)
        b = hex_bounds(CC, VC)[name]
        assert feasible(6, 0, b * (1 - 1e-9), 60, CC, VC)
        assert not feasible(6, 0, b * (1 + 1e-9), 60, CC, VC)
        assert not feasible(6, 0, b, 60, CC, VC), "equality must be infeasible (strict inequality)"

    def test_twisted_hex_c1_boundary(self):
        """C1 is the only constraint shared with (6, 0): at CC = 22, VC = 4, A = 90 it binds and the others hold."""
        b = hex_bounds(22, 4)["C1"]
        assert feasible(6, 1, b * (1 - 1e-6), 90, 22, 4)
        assert not feasible(6, 1, b * (1 + 1e-6), 90, 22, 4)
        assert not feasible(6, 1, b, 90, 22, 4), "equality must be infeasible (strict inequality)"

    @pytest.mark.parametrize("A, CC, VC", [(90, 4, 5), (90, 4, 8), (75, 4, 7), (75, 4, 10), (60, 4, 10)])
    def test_vertical_packing_boundary(self, A, CC, VC):
        """C5 is a closed-form upper bound on R; the other two constraints hold on both sides of it."""
        b = c5_bound(A, VC)
        assert 2 < b < C.PARAM_BOUNDS["R"][1]
        assert feasible(6, 1, b * (1 - 1e-9), A, CC, VC)
        assert not feasible(6, 1, b * (1 + 1e-9), A, CC, VC)
        assert not C.C5.satisfied(b, A, CC, VC, OD, L, G)[0], "equality must be infeasible (strict inequality)"
        assert C.C4P.satisfied(b * (1 + 1e-9), A, CC, VC, OD, L, G)[0] and C.C1.satisfied(b * (1 + 1e-9), A, CC, VC, OD, L, G)[0]

    @pytest.mark.parametrize("R, CC, VC", [(3.5, 12, 10), (4.0, 8, 9), (4.0, 6, 9), (3.5, 10, 9), (3.5, 10, 10), (3.0, 16, 10)])
    def test_tilt_boundary_of_c4p(self, R, CC, VC):
        """Where Vsep < 2R, C4' is a lower bound on A (cot decreases with A); C1 and C5 hold around it."""
        a_min = c4p_amin(R, CC, VC)
        assert 30 < a_min < 90
        for a in (a_min * (1 + 1e-9), 90):
            assert C.C1.satisfied(R, a, CC, VC, OD, L, G)[0] and C.C5.satisfied(R, a, CC, VC, OD, L, G)[0], "premise"
        assert feasible(6, 1, R, a_min * (1 + 1e-9), CC, VC)
        assert not feasible(6, 1, R, a_min * (1 - 1e-9), CC, VC)
        assert not feasible(6, 1, R, 30, CC, VC) and feasible(6, 1, R, 90, CC, VC)     # the A box limits

    def test_c4p_needs_no_tilt_where_vsep_exceeds_2r(self):
        """Vsep > 2R (the OR branch) accepts every tilt, including the 30 deg box limit."""
        R, CC, VC = 2.5, 12, 5
        assert (L - 2 * (R + G)) / (VC - 1) > 2 * R
        assert all(feasible(6, 1, R, A, CC, VC) for A in (30, 45, 60, 90))

    @pytest.mark.parametrize("name", ["E1", "E2", "E3"])
    def test_ellipse_constraint_boundary(self, name):
        CC, VC = _binding(name, ellipse_bounds)
        b = ellipse_bounds(CC, VC)[name]
        assert feasible(0, 0, b * (1 - 1e-9), 60, CC, VC)
        assert not feasible(0, 0, b * (1 + 1e-9), 60, CC, VC)
        if name != "E3":                     # E3's root is a closed-form solution, not the same floating-point expression
            assert not feasible(0, 0, b, 60, CC, VC), "equality must be infeasible (strict inequality)"

    @pytest.mark.parametrize("CC, VC", [(4, 4), (6, 5), (12, 5), (22, 10), (8, 7)])
    def test_e3_root_is_where_neighbouring_holes_touch(self, CC, VC):
        R = e3_root(CC, VC)
        h = math.pi * OD / (2 * CC)
        vs = (L - 2 * (R + G)) / (VC - 1)
        assert 2 * R == pytest.approx(math.hypot(h, vs), rel=1e-9)

    def test_e3_means_neighbouring_holes_do_not_overlap(self):
        """E3 holds exactly when the centres of staggered neighbours are further apart than 2R."""
        rng = np.random.default_rng(1)
        for _ in range(500):
            R, CC, VC = rng.uniform(2, 8.8), int(rng.integers(4, 23)), int(rng.integers(4, 11))
            h = math.pi * OD / (2 * CC)
            vs = (L - 2 * (R + G)) / (VC - 1)
            assert bool(C.E3.satisfied(R, 60, CC, VC, OD, L, G)[0]) == (2 * R < math.hypot(h, vs))

    def test_every_constraint_is_exercised(self):
        """Guards the boundary tests above: each upper bound must be the binding one somewhere."""
        for name in ("C1", "C2", "C3"):
            _binding(name, hex_bounds)
        for A, CC, VC in ((90, 4, 10), (60, 4, 10)):                 # C5 binds (see test_vertical_packing_boundary)
            assert 2 < c5_bound(A, VC) < min(hex_bounds(CC, VC)["C1"], C.PARAM_BOUNDS["R"][1])
        for name in ("E1", "E2", "E3"):
            _binding(name, ellipse_bounds)

    def test_configurations_are_genuinely_different(self):
        # R just above the half pitch with many cells: fine for the hexagons, rejected by E2 (holes overlap in a row).
        assert feasible(6, 0, 3.0, 60, 22, 4)
        assert not feasible(0, 0, 3.0, 60, 22, 4)
        # Many rows with a shallow tilt: fine untwisted, rejected by C4' of the twist ...
        assert feasible(6, 0, 3.5, 60, 12, 10) and not feasible(6, 1, 3.5, 60, 12, 10)
        # ... and fine again once the tilt is large enough (C4' needs A >= 85.40 deg here).
        assert feasible(6, 1, 3.5, 86, 12, 10)
        # C5 relaxes C3 for tilted cells: R = 4.5 at A = 60, CC = 4, VC = 10 breaks C3 (bound 4.0) but not C5 (bound 5.03).
        assert not feasible(6, 0, 4.5, 60, 4, 10) and feasible(6, 1, 4.5, 60, 4, 10)

    def test_the_saved_default_designs_of_the_parts_are_feasible(self):
        """Regression for the E3 flip: the designs stored in N0AShell / N6AShell are buildable, so they must pass."""
        assert feasible(0, 0, 4.0, 60.0, 12, 5)          # N0AShell: R=4, A=60, CC=12, VC=5
        assert feasible(6, 0, 4.0, 90.0, 16, 5)          # N6AShell: R=4, A=90, CC=16, VC=5

    def test_the_old_e3_inequality_is_gone(self):
        """With E3 as first written (4R^2 > d^2) this point was accepted; SolidWorks cannot build it (Wrap1)."""
        assert not feasible(0, 0, 5.1, 60.0, 12, 5)


class TestTwistedConstraints:
    def test_c4p_and_c5_match_the_independent_reference(self):
        rng = np.random.default_rng(2)
        for _ in range(2000):
            R, A = float(rng.uniform(2, 8.8)), float(rng.uniform(30, 90))
            CC, VC = int(rng.integers(4, 23)), int(rng.integers(4, 11))
            ref = reference(6, 1, R, A, CC, VC)
            assert bool(C.C4P.satisfied(R, A, CC, VC, OD, L, G)[0]) == ref["C4'"]
            assert bool(C.C5.satisfied(R, A, CC, VC, OD, L, G)[0]) == ref["C5"]
            assert bool(C.C1.satisfied(R, A, CC, VC, OD, L, G)[0]) == ref["C1"]

    def test_c5_is_the_old_c3_at_a_right_angle_and_loosens_with_a_smaller_tilt(self):
        rng = np.random.default_rng(5)
        R, CC, VC = rng.uniform(2, 8.8, 5000), rng.integers(4, 23, 5000), rng.integers(4, 11, 5000)
        assert (C.C5.satisfied(R, 90.0, CC, VC, OD, L, G) == C.C3.satisfied(R, 90.0, CC, VC, OD, L, G)).all()
        for VC_ in range(4, 11):
            bounds = [c5_bound(A, VC_) for A in (30, 45, 60, 75, 90)]
            assert bounds == sorted(bounds, reverse=True) and len(set(bounds)) == 5
            assert bounds[-1] == pytest.approx((L - 2 * G) / (VC_ + 1))

    def test_vsep_equal_to_2r_is_violated_without_a_warning(self):
        """Vsep = 2R makes the C4' ratio a division by zero (and the OR branch needs '>'): violated, silently."""
        R, VC = 5.5, 4                                # (50 - 2*(5.5 + 3)) / 3 = 11 = 2R exactly
        assert (L - 2 * (R + G)) / (VC - 1) == 2 * R
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert not C.C4P.satisfied(R, 90.0, 12, VC, OD, L, G)[0]

    def test_a_is_required_for_twisted_rows_only(self):
        no_a = pd.DataFrame({"R": [3.0], "CC": [12], "VC": [5]})
        assert C.evaluate(no_a.assign(T=0, N=6), OD, L, G)["feasible"].tolist() == [True]       # (6, 0) does not need A
        with pytest.raises(ValueError, match="C4'"):
            C.evaluate(no_a.assign(T=1, N=6), OD, L, G)
        assert C.C4P.uses_A and C.C5.uses_A and not C.C1.uses_A and not C.E3.uses_A

    def test_twist_is_not_a_subset_of_untwisted_any_more(self):
        """C5 loosens C3 for tilted cells, so (6, 1) accepts some designs that (6, 0) rejects - and vice versa."""
        rng = np.random.default_rng(3)
        n = 20000
        df = pd.DataFrame({"R": rng.uniform(2, 8.8, n), "A": rng.uniform(30, 90, n),
                           "CC": rng.integers(4, 23, n), "VC": rng.integers(4, 11, n)})
        plain = C.feasible_mask(df.assign(T=0, N=6), OD, L, G)
        twist = C.feasible_mask(df.assign(T=1, N=6), OD, L, G)
        assert (twist & ~plain).any() and (plain & ~twist).any()
        report = C.evaluate(df.assign(T=1, N=6), OD, L, G)
        assert (twist == (report["C1"].astype(bool) & report["C4'"].astype(bool) & report["C5"].astype(bool))).all()
        assert report["C2"].isna().all() and report["C3"].isna().all()

    def test_tilt_decides_the_report_columns(self):
        df = pd.DataFrame({"R": [3.5, 3.5], "A": [60, 90], "CC": [12, 12], "VC": [10, 10], "T": [1, 1], "N": [6, 6]})
        report = C.evaluate(df, OD, L, G)
        assert report[["C1", "C5"]].astype(bool).all().all()
        assert report["C4'"].astype(bool).tolist() == [False, True]
        assert report["feasible"].tolist() == [False, True]


class TestReferenceAgreement:
    @pytest.mark.parametrize("cfg", [HEX, TWIST_HEX, ELLIPSE])
    def test_vectorised_matches_scalar_reference_on_dense_grid(self, cfg):
        N, T = cfg
        rows = [
            {"R": R, "A": A, "CC": CC, "VC": VC, "T": T, "N": N}
            for R in np.linspace(2.0, 8.8, 35)
            for CC in range(4, 23)
            for VC in range(4, 11)
            for A in (30.0, 45.0, 60.0, 75.0, 90.0)
        ]
        df = pd.DataFrame(rows)
        report = C.evaluate(df, OD, L, G)

        expected = {name: [] for name in reference(N, T, 3.0, 60, 12, 5)}
        for r in rows:
            for name, ok in reference(N, T, r["R"], r["A"], r["CC"], r["VC"]).items():
                expected[name].append(ok)
        for name, exp in expected.items():
            assert report[name].astype(bool).tolist() == exp, name
        assert report["feasible"].tolist() == [all(v) for v in zip(*expected.values())]
        assert 0.0 < report["feasible"].mean() < 1.0      # the grid really contains both outcomes

    def test_acceptance_rates_are_sane(self):
        """Share of the design box accepted: ~37 % (6, 0), ~34 % (6, 1), ~32 % (0, 0) - a coarse regression guard."""
        rng = np.random.default_rng(0)
        n = 200_000
        df = pd.DataFrame({
            "R": rng.uniform(2, 8.8, n), "A": rng.uniform(30, 90, n),
            "CC": rng.integers(4, 23, n), "VC": rng.integers(4, 11, n),
        })
        rates = {cfg: C.feasible_mask(df.assign(N=cfg[0], T=cfg[1]), OD, L, G).mean() for cfg in IMPLEMENTED}
        assert 0.35 < rates[HEX] < 0.39
        assert 0.32 < rates[TWIST_HEX] < 0.36
        assert 0.30 < rates[ELLIPSE] < 0.335


class TestUndefinedArithmetic:
    @pytest.mark.parametrize("cfg", [HEX, TWIST_HEX, ELLIPSE])
    @pytest.mark.parametrize("column, value", [
        ("CC", 0), ("CC", -4), ("VC", 0), ("VC", -1), ("R", 0.0), ("R", -3.0),
        ("R", float("nan")), ("CC", float("nan")), ("VC", float("inf")), ("A", float("nan")),
        ("R", float("inf")), ("CC", float("-inf")), ("A", float("inf")),
    ])
    def test_undefined_or_non_finite_is_violated_without_error(self, cfg, column, value):
        point = dict(BOTH_FEASIBLE)
        assert feasible(*cfg, **point)               # sanity: feasible until we break one input
        point[column] = value
        with warnings.catch_warnings():
            warnings.simplefilter("error")           # no RuntimeWarning from divide-by-zero / invalid
            assert not feasible(*cfg, **point)

    def test_vc_equal_one_is_undefined_for_e3(self):
        # VC - 1 = 0 in the denominator of E3 (E1 and E2 alone would pass).
        assert C.E1.satisfied(3.0, 60, 12, 1, OD, L, G)[0]
        assert C.E2.satisfied(3.0, 60, 12, 1, OD, L, G)[0]
        assert not C.E3.satisfied(3.0, 60, 12, 1, OD, L, G)[0]
        assert not feasible(0, 0, 3.0, 60, 12, 1)

    def test_rows_that_do_not_fit_are_rejected_by_e3_alone(self):
        """(L-2(R+G)) < 0 is squared inside E3; the inequality must still reject such R without help from E1."""
        for R in (23.0, 30.0, 100.0):                # L - 2(R+G) < 0
            assert not C.E3.satisfied(R, 60, 12, 5, OD, L, G)[0]
            assert not feasible(0, 0, R, 60, 12, 5)

    def test_a_column_is_optional(self):
        df = pd.DataFrame({"R": [3.0], "CC": [12], "VC": [5]})
        assert C.evaluate(df, OD, L, G)["feasible"].tolist() == [True]

    @pytest.mark.parametrize("bad", [
        dict(OD=0.0), dict(OD=-1.0), dict(L=0.0), dict(L=float("nan")), dict(G=-0.5), dict(G=float("inf")),
    ])
    def test_invalid_constants_raise(self, bad):
        kw = dict(OD=OD, L=L, G=G)
        kw.update(bad)
        with pytest.raises(ValueError):
            C.evaluate(pd.DataFrame({"R": [3.0], "CC": [12], "VC": [5]}), **kw)


class TestGapRegression:
    def test_gap_changes_the_verdict(self):
        # R = 9 with VC = 4: C3 bound is 50/5 = 10 for G = 0 but 44/5 = 8.8 for G = 3.
        assert feasible(6, 0, 9.0, 60, 4, 4, G=0.0)
        assert not feasible(6, 0, 9.0, 60, 4, 4, G=3.0)

    def test_default_gap_is_zero(self):
        df = pd.DataFrame({"R": [9.0], "CC": [4], "VC": [4]})
        assert C.feasible_mask(df, OD, L).tolist() == [True]
        assert C.feasible_mask(df, OD, L, G=3.0).tolist() == [False]


# ---------------------------------------------------------------------------
# Configuration routing and placeholders
# ---------------------------------------------------------------------------

class TestConfigurationRouting:
    def test_missing_configuration_columns_mean_legacy_hex(self):
        df = pd.DataFrame({"R": [3.0, 8.5], "A": [60, 60], "CC": [12, 12], "VC": [5, 5]})
        explicit = df.assign(T=0, N=6)
        assert C.feasible_mask(df, OD, L, G).tolist() == C.feasible_mask(explicit, OD, L, G).tolist() == [True, False]
        report = C.evaluate(df, OD, L, G)
        assert report["N"].tolist() == [6, 6] and report["T"].tolist() == [0, 0]

    def test_each_row_uses_its_own_constraint_set(self):
        df = pd.DataFrame({
            "R": [3.0, 3.0, 3.0], "A": [60, 60, 60], "CC": [12, 12, 12], "VC": [5, 5, 5],
            "T": [0, 0, 1], "N": [6, 0, 6],
        })
        report = C.evaluate(df, OD, L, G)
        assert report.loc[0, ["C1", "C2", "C3"]].notna().all() and report.loc[0, ["C4'", "C5", "E1", "E2", "E3"]].isna().all()
        assert report.loc[1, ["E1", "E2", "E3"]].notna().all() and report.loc[1, ["C1", "C2", "C3", "C4'", "C5"]].isna().all()
        assert report.loc[2, ["C1", "C4'", "C5"]].notna().all() and report.loc[2, ["C2", "C3", "E1", "E2", "E3"]].isna().all()

    @pytest.mark.parametrize("cfg", [(0, 1)])
    def test_placeholder_configurations_are_undefined_and_never_feasible(self, cfg):
        df = pd.DataFrame({"R": [3.0], "A": [60], "CC": [12], "VC": [5], "T": [cfg[1]], "N": [cfg[0]]})
        report = C.evaluate(df, OD, L, G)
        assert report["defined"].tolist() == [False]
        assert report["feasible"].tolist() == [False]
        with pytest.raises(C.ConstraintsNotDefinedError):
            C.constraints_for(cfg)
        assert issubclass(C.ConstraintsNotDefinedError, NotImplementedError)

    def test_only_derived_configurations_are_implemented(self):
        assert C.implemented_configs() == [(6, 0), (6, 1), (0, 0)]

    @pytest.mark.parametrize("column, value", [("N", 3), ("N", 8), ("T", 2), ("T", -1), ("N", float("nan")), ("T", "x")])
    def test_invalid_configuration_values_raise(self, column, value):
        df = pd.DataFrame({"R": [3.0], "A": [60], "CC": [12], "VC": [5], "T": [0], "N": [6]})
        df[column] = [value]
        with pytest.raises(ValueError):
            C.evaluate(df, OD, L, G)

    def test_configuration_columns_accept_numeric_strings_and_floats(self):
        df = pd.DataFrame({"R": [3.0, 3.0], "A": [60, 60], "CC": [12, 12], "VC": [5, 5],
                           "T": ["0", 0.0], "N": ["6", 6.0]})
        assert C.evaluate(df, OD, L, G)["feasible"].tolist() == [True, True]

    def test_empty_frame(self):
        df = pd.DataFrame({"R": [], "A": [], "CC": [], "VC": [], "T": [], "N": []})
        report = C.evaluate(df, OD, L, G)
        assert len(report) == 0 and "feasible" in report.columns

    def test_missing_required_column(self):
        with pytest.raises(ValueError, match="CC"):
            C.evaluate(pd.DataFrame({"R": [3.0], "VC": [5]}), OD, L, G)


class TestParseConfigs:
    @pytest.mark.parametrize("spec, expected", [
        ("all", [(6, 0), (6, 1), (0, 0)]),
        ("legacy", [(6, 0)]),
        ("6:1", [(6, 1)]),
        ("N6T1,N0T0", [(6, 1), (0, 0)]),
        ("6:0,0:0", [(6, 0), (0, 0)]),
        ("N0T0,N6T0", [(0, 0), (6, 0)]),
        ("n6_t0", [(6, 0)]),
        ("6:0,6:0,N6T0", [(6, 0)]),
        ("legacy,0:0", [(6, 0), (0, 0)]),
    ])
    def test_valid_specs(self, spec, expected):
        assert C.parse_configs(spec) == expected

    @pytest.mark.parametrize("spec", ["0:1", "N0T1", "all,0:1", "6:1,0:1"])
    def test_placeholders_are_rejected(self, spec):
        with pytest.raises(C.ConstraintsNotDefinedError):
            C.parse_configs(spec)

    @pytest.mark.parametrize("spec", ["", "banana", "3:0", "6:2", "N6T"])
    def test_junk_is_rejected(self, spec):
        with pytest.raises(ValueError):
            C.parse_configs(spec)


# ---------------------------------------------------------------------------
# 2. Pre-flight
# ---------------------------------------------------------------------------

GOOD_HEX = dict(R=3.41, A=47.51, CC=8, VC=9, T=0, N=6)       # first row of Sample_OD40L50G3_New.csv


def frame(*rows, **defaults):
    base = dict(GOOD_HEX)
    base.update(defaults)
    return pd.DataFrame([{**base, **r} for r in rows])


def reasons_of(result, row):
    return result.rejected.set_index("row").loc[row, "reasons"]


class TestPreflight:
    def test_accepts_feasible_points_and_names_them(self):
        df = pd.DataFrame([GOOD_HEX, dict(BOTH_FEASIBLE, T=0, N=0), dict(BOTH_FEASIBLE, T=1, N=6)])
        res = P.preflight(df, OD, L, G, "Shell")
        assert res.rejected.empty
        assert res.accepted["row"].tolist() == [1, 2, 3]
        assert res.accepted["part"].tolist() == ["N6AShell.SLDPRT", "N0AShell.SLDPRT", "N6TAShell.SLDPRT"]
        assert res.accepted["stem"].tolist() == [
            "N6AShellR341A4751CC800VC900",
            "N0AShellR300A6000CC1200VC500",
            "N6TAShellR300A6000CC1200VC500",
        ]

    def test_fidelity_changes_the_part_and_the_stem(self):
        res = P.preflight(frame({}), OD, L, G, "Solid")
        assert res.accepted["part"].tolist() == ["N6ASolid.SLDPRT"]
        assert res.accepted["stem"].iloc[0].startswith("N6ASolidR")

    def test_unknown_fidelity_and_missing_columns_are_structural_errors(self):
        with pytest.raises(ValueError):
            P.preflight(frame({}), OD, L, G, "Mesh")
        with pytest.raises(ValueError, match="VC"):
            P.preflight(pd.DataFrame({"R": [3.0], "A": [60], "CC": [12]}), OD, L, G)

    def test_missing_configuration_columns_default_to_legacy_with_a_note(self):
        df = pd.DataFrame([dict(R=3.41, A=47.51, CC=8, VC=9)])
        res = P.preflight(df, OD, L, G)
        assert res.accepted[["N", "T"]].values.tolist() == [[6, 0]]
        assert any("legacy" in n for n in res.notes)

    @pytest.mark.parametrize("override, fragment", [
        (dict(CC=12.5), "CC=12.5 is not an integer"),
        (dict(VC=6.2), "VC=6.2 is not an integer"),
        (dict(R=1.99), "R=1.99 outside"),
        (dict(R=8.81), "R=8.81 outside"),
        (dict(A=29.99), "A=29.99 outside"),
        (dict(A=90.01), "A=90.01 outside"),
        (dict(CC=3), "CC=3 outside"),
        (dict(CC=23), "CC=23 outside"),
        (dict(VC=3), "VC=3 outside"),
        (dict(VC=11), "VC=11 outside"),
        (dict(R=float("nan")), "R is missing or not a finite number"),
        (dict(A=float("inf")), "A is missing or not a finite number"),
        (dict(R="abc"), "R is missing or not a finite number"),
        (dict(N=3), "invalid configuration N=3"),
        (dict(T=2), "invalid configuration"),
        (dict(N=float("nan")), "N is missing or not a finite number"),
    ])
    def test_bad_values_are_rejected_with_a_reason(self, override, fragment):
        res = P.preflight(frame(override), OD, L, G)
        assert res.accepted.empty
        assert fragment in reasons_of(res, 1)

    @pytest.mark.parametrize("override", [
        dict(R=2.0, A=30.0, CC=4, VC=4),        # every parameter at its lower bound
        dict(R=2.0, A=90.0, CC=22, VC=4),       # A at the upper bound, CC at the upper bound
        dict(R=2.0, A=30.0, CC=22, VC=10),      # VC at the upper bound
    ])
    def test_box_corners_are_inclusive(self, override):
        res = P.preflight(frame(override), OD, L, G)
        assert len(res.accepted) == 1, res.rejected.to_string()

    def test_constraint_violations_name_the_violated_constraints(self):
        res = P.preflight(frame(dict(R=8.5, CC=12, VC=5),                   # hex: above all three bounds
                                dict(R=3.0, CC=12, VC=5, N=0),              # ellipse: fine
                                dict(R=3.0, CC=22, VC=4, N=0),              # ellipse: R >= the half pitch 2.86 (E2) only
                                dict(R=5.1, CC=12, VC=5, N=0),              # ellipse: neighbouring holes overlap (E3) only
                                dict(R=3.5, A=60.0, CC=12, VC=10, N=6, T=1),  # twisted hex: tilt below C4's minimum (85.4) only
                                dict(R=3.5, A=90.0, CC=12, VC=10, N=6, T=1),  # twisted hex: tilt large enough
                                dict(R=6.0, A=90.0, CC=4, VC=10, N=6, T=1)),  # twisted hex: R above the vertical-packing bound (C5)
                          OD, L, G)
        assert res.accepted["row"].tolist() == [2, 6]
        assert reasons_of(res, 1) == "violates C1, C2, C3"
        assert reasons_of(res, 3) == "violates E2"
        assert reasons_of(res, 4) == "violates E3"
        assert reasons_of(res, 5) == "violates C4'"
        assert reasons_of(res, 7) == "violates C5"

    @pytest.mark.parametrize("cfg", [(0, 1)])
    def test_placeholder_configurations_never_reach_cad(self, cfg):
        res = P.preflight(frame(dict(N=cfg[0], T=cfg[1])), OD, L, G)
        assert res.accepted.empty
        assert "placeholder" in reasons_of(res, 1)

    def test_duplicates_and_rounding_collisions_reject_the_later_row(self):
        df = frame({}, {}, dict(R=3.404), dict(R=3.405), dict(R=3.406))
        # rows: 1 = good, 2 = identical, 3/4 = R rounds to 340, 5 = R rounds to 341 (collides with row 1)
        res = P.preflight(df, OD, L, G)
        assert res.accepted["row"].tolist() == [1, 3]
        assert "collides with row 1" in reasons_of(res, 2)
        assert "collides with row 3" in reasons_of(res, 4)      # 3.405 -> 340.49999999999994 -> 340
        assert "collides with row 1" in reasons_of(res, 5)      # 3.406 -> 341 == GOOD_HEX's 3.41 -> 341

    def test_same_point_in_different_configurations_gets_distinct_stems(self):
        # A point feasible in every implemented set, so only the naming rule decides: the configuration is in the stem.
        assert all(feasible(N, T, **BOTH_FEASIBLE) for N, T in IMPLEMENTED)
        res = P.preflight(pd.DataFrame([dict(BOTH_FEASIBLE, T=0, N=6), dict(BOTH_FEASIBLE, T=0, N=0),
                                        dict(BOTH_FEASIBLE, T=1, N=6)]), OD, L, G)
        assert res.rejected.empty
        assert res.accepted["stem"].tolist() == ["N6AShellR300A6000CC1200VC500", "N0AShellR300A6000CC1200VC500",
                                                  "N6TAShellR300A6000CC1200VC500"]

    def test_row_numbers_are_one_based_and_survive_rejections(self):
        df = frame({}, dict(CC=12.5), {}, dict(R=0.5), dict(R=3.0, A=61.0, CC=12, VC=5))
        res = P.preflight(df, OD, L, G)
        assert res.rejected["row"].tolist() == [2, 3, 4]      # row 3 duplicates row 1
        assert res.accepted["row"].tolist() == [1, 5]

    def test_empty_single_and_all_rejected_inputs(self):
        empty = P.preflight(pd.DataFrame({"R": [], "A": [], "CC": [], "VC": []}), OD, L, G)
        assert empty.accepted.empty and empty.rejected.empty
        assert {"row", "N", "T", "R", "A", "CC", "VC", "part", "stem"} <= set(empty.accepted.columns)
        assert {"row", "reasons"} <= set(empty.rejected.columns)

        single = P.preflight(frame({}), OD, L, G)
        assert len(single.accepted) == 1 and single.rejected.empty

        bad = P.preflight(frame(dict(CC=12.5), dict(R=-1)), OD, L, G)
        assert bad.accepted.empty and len(bad.rejected) == 2

    def test_accepted_points_really_satisfy_the_constraints(self):
        rng = np.random.default_rng(3)
        n = 400
        df = pd.DataFrame({
            "R": np.round(rng.uniform(1.5, 9.3, n), 2), "A": np.round(rng.uniform(28, 92, n), 2),
            "CC": rng.integers(3, 24, n), "VC": rng.integers(3, 12, n),
            "T": rng.choice([0, 0, 0, 1], n), "N": rng.choice([0, 6], n),
        })
        res = P.preflight(df, OD, L, G)
        assert len(res.accepted) > 0 and len(res.rejected) > 0
        assert len(res.accepted) + len(res.rejected) == n
        assert C.feasible_mask(res.accepted[["R", "A", "CC", "VC", "T", "N"]], OD, L, G).all()
        twisted = res.accepted[res.accepted["T"] == 1]
        assert len(twisted) > 0 and (twisted["N"] == 6).all()        # only (6, 1) is derived; (0, 1) is never accepted
        assert res.accepted["stem"].is_unique


class TestSwGenHandOff:
    def test_batches_carry_only_the_four_geometry_columns(self, tmp_path):
        df = pd.DataFrame([
            dict(R=3.41, A=47.51, CC=8, VC=9, T=0, N=6),
            dict(BOTH_FEASIBLE, T=0, N=0),
            dict(R=3.0, A=61.0, CC=12, VC=5, T=0, N=6),
            dict(R=9.9, A=60.0, CC=12, VC=5, T=0, N=6),        # rejected
        ])
        res = P.preflight(df, OD, L, G)
        batches = P.write_swgen_batches(res, str(tmp_path))

        assert [b["config"] for b in batches] == [(0, 0), (6, 0)]
        assert [b["part"] for b in batches] == ["N0AShell.SLDPRT", "N6AShell.SLDPRT"]
        assert [b["n"] for b in batches] == [1, 2]
        assert [b["rows"] for b in batches] == [[2], [1, 3]]

        for b in batches:
            raw = pd.read_csv(b["csv"], dtype=str)
            assert list(raw.columns) == ["R", "A", "CC", "VC"]      # SwGen aborts on T / N columns
            assert raw["CC"].str.fullmatch(r"\d+").all() and raw["VC"].str.fullmatch(r"\d+").all()   # written as ints
        assert (tmp_path / "rejected.csv").is_file()
        assert "violates" in pd.read_csv(tmp_path / "rejected.csv")["reasons"].iloc[0]

    def test_no_rejected_csv_when_everything_passes(self, tmp_path):
        res = P.preflight(frame({}), OD, L, G)
        P.write_swgen_batches(res, str(tmp_path))
        assert not (tmp_path / "rejected.csv").exists()

    def test_nothing_accepted_writes_no_batch(self, tmp_path):
        res = P.preflight(frame(dict(CC=12.5)), OD, L, G)
        assert P.write_swgen_batches(res, str(tmp_path)) == []
        assert [p.name for p in tmp_path.iterdir()] == ["rejected.csv"]

    def test_command_line_targets_the_right_part(self, tmp_path):
        res = P.preflight(frame({}), OD, L, G)
        batch = P.write_swgen_batches(res, str(tmp_path))[0]
        cmd = P.swgen_command(batch, parts_dir="C:/parts", out_dir=str(tmp_path), swgen_exe="SwGen.exe")
        assert cmd.startswith('"SwGen.exe" generate --part "C:/parts')
        assert "N6AShell.SLDPRT" in cmd and batch["csv"] in cmd and "step_N6_T0" in cmd

    def test_cli_exit_codes(self, tmp_path, capsys):
        good = tmp_path / "good.csv"
        frame({}).to_csv(good, index=False)
        assert P.main([str(good), "--OD", "40", "--L", "50", "--G", "3", "--out", str(tmp_path / "o1")]) == 0
        mixed = tmp_path / "mixed.csv"
        frame({}, dict(CC=12.5)).to_csv(mixed, index=False)
        assert P.main([str(mixed), "--OD", "40", "--L", "50", "--G", "3", "--out", str(tmp_path / "o2")]) == 1
        assert P.main([str(tmp_path / "missing.csv"), "--OD", "40", "--L", "50"]) == 2
        no_cols = tmp_path / "nocols.csv"
        pd.DataFrame({"R": [3.0]}).to_csv(no_cols, index=False)
        assert P.main([str(no_cols), "--OD", "40", "--L", "50"]) == 2
        capsys.readouterr()


class TestSamplerFeedsPreflight:
    def test_every_sampled_point_passes_preflight(self):
        df = S.generate_sobol(k=32, seed=11, OD=OD, L=L, G=G, configs=IMPLEMENTED)
        res = P.preflight(df, OD, L, G)
        assert res.rejected.empty, res.rejected.to_string()
        assert len(res.accepted) == len(df) and res.accepted["stem"].is_unique
        assert set(res.accepted["part"]) == {"N6AShell.SLDPRT", "N6TAShell.SLDPRT", "N0AShell.SLDPRT"}


# ---------------------------------------------------------------------------
# 3. Naming contract
# ---------------------------------------------------------------------------

class TestNaming:
    @pytest.mark.parametrize("N, T, fid, expected", [
        (6, 0, "Shell", "N6AShell"), (6, 1, "Shell", "N6TAShell"),
        (0, 0, "Shell", "N0AShell"), (0, 1, "Shell", "N0TAShell"),
        (6, 0, "Solid", "N6ASolid"), (6, 1, "Solid", "N6TASolid"),
        (0, 0, "Solid", "N0ASolid"), (0, 1, "Solid", "N0TASolid"),
    ])
    def test_part_names(self, N, T, fid, expected):
        assert P.part_name(N, T, fid) == expected
        assert P.part_file(N, T, fid) == expected + ".SLDPRT"

    @pytest.mark.parametrize("args", [(3, 0, "Shell"), (6, 2, "Shell"), (6, 0, "Mesh"), (None, 0, "Shell"), (6, 0.5, "Shell")])
    def test_invalid_part_arguments_raise(self, args):
        with pytest.raises(ValueError):
            P.part_name(*args)

    def test_half_to_even_rounding_matches_swgen(self):
        # SwGen: Math.Round(value * 100, MidpointRounding.ToEven); 0.125/0.375/0.625 * 100 are exact halves.
        assert "R12A" in P.stp_stem(6, 0, "Shell", 0.125, 60, 12, 5)     # 12.5 -> 12 (not 13)
        assert "R38A" in P.stp_stem(6, 0, "Shell", 0.375, 60, 12, 5)     # 37.5 -> 38
        assert "R62A" in P.stp_stem(6, 0, "Shell", 0.625, 60, 12, 5)     # 62.5 -> 62
        assert "R340A" in P.stp_stem(6, 0, "Shell", 3.405, 60, 12, 5)    # 3.405 * 100 = 340.49999999999994

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
    @pytest.mark.parametrize("position", ["R", "A", "CC", "VC"])
    def test_unencodable_values_raise(self, position, bad):
        vals = dict(R=3.0, A=60.0, CC=12, VC=5)
        vals[position] = bad
        with pytest.raises(ValueError):
            P.stp_stem(6, 0, "Shell", **vals)

    def test_stems_are_windows_safe_and_match_the_expected_shape(self):
        rng = np.random.default_rng(5)
        pattern = re.compile(r"^N(0|6)(T)?A(Shell|Solid)R\d+A\d+CC\d+VC\d+$")
        for _ in range(200):
            N, T = int(rng.choice([0, 6])), int(rng.choice([0, 1]))
            fid = str(rng.choice(["Shell", "Solid"]))
            stem = P.stp_stem(N, T, fid, rng.uniform(2, 8.8), rng.uniform(30, 90), int(rng.integers(4, 23)), int(rng.integers(4, 11)))
            assert pattern.match(stem), stem
            assert re.fullmatch(r"[A-Za-z0-9]+", stem) and len(stem) < 60

    def test_readme_example_name(self):
        # Automation/SwGen/README.md: N6ASolidR340A5861CC1800VC400
        assert P.stp_stem(6, 0, "Solid", 3.40, 58.61, 18, 4) == "N6ASolidR340A5861CC1800VC400"

    def test_stem_round_trips_through_analysis_parser(self):
        analysis = _load_analysis()
        rng = np.random.default_rng(9)
        for _ in range(50):
            N, T = int(rng.choice([0, 6])), int(rng.choice([0, 1]))
            R, A = round(float(rng.uniform(2, 8.8)), 2), round(float(rng.uniform(30, 90)), 2)
            CC, VC = int(rng.integers(4, 23)), int(rng.integers(4, 11))
            parsed = analysis.parse_param_string(P.stp_stem(N, T, "Shell", R, A, CC, VC) + "_FD")
            assert parsed["N"] == str(N)
            assert parsed["T"] == T
            assert parsed["R"] == pytest.approx(R) and parsed["A"] == pytest.approx(A)
            assert parsed["CC"] == pytest.approx(CC) and parsed["VC"] == pytest.approx(VC)


def _load_analysis():
    try:
        import analysis                                    # imports the plotting stack (Qt backend)
    except Exception as exc:                               # pragma: no cover - environment dependent
        pytest.skip(f"analysis.py cannot be imported here: {exc}")
    return analysis




# ---------------------------------------------------------------------------
# 4. SwGen + SolidWorks (opt-in:  pytest -m solidworks)
# ---------------------------------------------------------------------------
# These tests launch and close their own SolidWorks instance, so they skip if one is already running
# (they must never touch an interactive session).  Behaviour tests (row handling, naming, columns) use the
# part's own default values as the known-good point, so they do not depend on the constraint definitions;
# `test_points_accepted_by_the_constraints_build` is the one that checks the constraints against real geometry.

def _swgen_exe():
    exe = os.environ.get("SWGEN_EXE") or os.path.join(
        REPO_ROOT, "Automation", "SwGen", "bin", "Release", "net48", "SwGen.exe")
    return exe if os.path.isfile(exe) else None


def _parts_dir():
    path = os.environ.get("SWGEN_PARTS_DIR")
    return path if path and os.path.isdir(path) else None


FIDELITY = os.environ.get("SWGEN_FIDELITY", "Shell")


def _solidworks_processes():
    """Running SLDWORKS.exe processes (exact image name; SolidWorks' background helpers do not match)."""
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq SLDWORKS.exe", "/FO", "CSV", "/NH"],
                         capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if line.strip().lower().startswith('"sldworks.exe"')]


def _run_swgen(*args, timeout=900):
    proc = subprocess.run([_swgen_exe(), *map(str, args)], capture_output=True, text=True, timeout=timeout)
    try:
        payload = json.loads(proc.stdout) if proc.stdout.strip() else {}
    except json.JSONDecodeError:
        payload = {"unparsed_stdout": proc.stdout}
    # A SolidWorks left running after SwGen exits is reused (silently, in a half-dead state) by the next
    # run, where equation edits then have no effect - so every invocation must clean up after itself.
    assert not _solidworks_processes(), "SwGen left a SolidWorks process running after it exited"
    return proc.returncode, payload, proc.stderr


def _results(out_dir):
    with open(os.path.join(out_dir, "swgen_results.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


_DEFAULTS = {}


def _defaults(part):
    """The part's own R, A, CC, VC values (read once per part with `SwGen equations`)."""
    if part not in _DEFAULTS:
        rc, payload, err = _run_swgen("equations", "--part", part)
        assert rc == 0, err
        values = {}
        for e in payload["equations"]:
            m = re.match(r'\s*"(R|A|CC|VC)"\s*=', e["equation"])
            if m and e["global_variable"]:
                values[m.group(1)] = e["value"]
        assert set(values) == {"R", "A", "CC", "VC"}, f"could not read the defaults from {part}: {values}"
        _DEFAULTS[part] = values
    return _DEFAULTS[part]


def _csv_row(d, **override):
    v = {**d, **override}
    return f"{v['R']},{v['A']},{int(v['CC'])},{int(v['VC'])}"


@pytest.mark.solidworks
class TestSwGen:
    @pytest.fixture(autouse=True)
    def _require_environment(self):
        if sys.platform != "win32":
            pytest.skip("SwGen needs Windows + SolidWorks")
        if _swgen_exe() is None:
            pytest.skip("SwGen.exe not found (build it or set SWGEN_EXE)")
        if _solidworks_processes():
            pytest.skip("SolidWorks is already running - close it first (these tests launch and close their "
                        "own instance and must not touch an interactive session)")

    def _parts(self):
        parts = _parts_dir()
        if parts is None:
            pytest.skip("set SWGEN_PARTS_DIR to the folder holding the SLDPRT files")
        found = [(cfg, os.path.join(parts, P.part_file(cfg[0], cfg[1], FIDELITY))) for cfg in C.implemented_configs()]
        found = [(cfg, path) for cfg, path in found if os.path.isfile(path)]
        if not found:
            pytest.skip(f"no {FIDELITY} part of an implemented configuration in {parts}")
        return found

    def _part(self, cfg):
        for c, path in self._parts():
            if c == cfg:
                return path
        pytest.skip(f"{P.part_file(cfg[0], cfg[1], FIDELITY)} not found in SWGEN_PARTS_DIR")

    def _any_part(self):
        return self._parts()[0][1]

    def test_probe_connects_and_leaves_nothing_behind(self):
        rc, payload, _ = _run_swgen("probe")
        assert rc == 0 and payload.get("ok") is True and payload.get("revision")

    def test_back_to_back_runs_apply_their_equations(self, tmp_path):
        """Regression: a lingering SolidWorks made the next run reconnect to a zombie whose equation edits did nothing."""
        part = self._any_part()
        d = _defaults(part)
        for k, delta in enumerate((0.37, 0.21)):
            csv = tmp_path / f"points{k}.csv"
            csv.write_text("R,A,CC,VC\n" + _csv_row(d, R=round(d["R"] + delta, 2)) + "\n")
            rc, summary, err = _run_swgen("generate", "--part", part, "--csv", csv, "--out", tmp_path / f"out{k}")
            row = _results(str(tmp_path / f"out{k}"))[0]
            assert row["readback"]["R"] == pytest.approx(d["R"] + delta), (row["errors"], err)
            assert rc == 0 and row["status"] == "ok", row["errors"]

    @pytest.mark.parametrize("cfg", C.implemented_configs())
    def test_points_accepted_by_the_constraints_build(self, cfg, tmp_path):
        """Every point the constraints of this configuration accept must rebuild and export in SolidWorks."""
        part = self._part(cfg)
        pts = S.generate_sobol(k=4, seed=21, OD=OD, L=L, G=G, configs=[cfg])[["R", "A", "CC", "VC"]]
        csv = tmp_path / "points.csv"
        pts.to_csv(csv, index=False)
        out = tmp_path / "step"

        rc, summary, err = _run_swgen("generate", "--part", part, "--csv", csv, "--out", out)
        results = _results(str(out))
        failed = [(r["params"], r["errors"]) for r in results if r["status"] != "ok"]
        assert not failed, (f"the constraints of (N={cfg[0]}, T={cfg[1]}) accepted points that SolidWorks could "
                            f"not build:\n" + "\n".join(f"  {p}: {e}" for p, e in failed))
        assert rc == 0, err

        for (_, r), res in zip(pts.iterrows(), results):
            stem = P.stp_stem(cfg[0], cfg[1], FIDELITY, r["R"], r["A"], r["CC"], r["VC"])
            assert res["name"] == stem
            stp = out / (stem + ".stp")
            assert stp.is_file() and stp.stat().st_size > 0
            assert res["readback"]["R"] == pytest.approx(r["R"], rel=1e-6)

    def test_short_csv_row_fails_without_a_file(self, tmp_path):
        part = self._any_part()
        d = _defaults(part)
        csv = tmp_path / "points.csv"
        csv.write_text("R,A,CC,VC\n" + _csv_row(d) + "\n3.0,60.0,12\n")             # second row one column short
        out = tmp_path / "step"
        rc, summary, _ = _run_swgen("generate", "--part", part, "--csv", csv, "--out", out)
        results = _results(str(out))
        assert rc == 1 and summary["ok"] == 1 and summary["failed"] == 1
        assert results[1]["status"] == "failed" and "malformed CSV row" in " ".join(results[1]["errors"])
        assert len(list(out.glob("*.stp"))) == 1

    def test_rows_rounding_to_the_same_name_fail_the_second(self, tmp_path):
        part = self._any_part()
        d = _defaults(part)
        csv = tmp_path / "points.csv"
        csv.write_text("R,A,CC,VC\n" + _csv_row(d) + "\n" + _csv_row(d, R=round(d["R"] + 0.004, 3)) + "\n")   # both -> same name
        out = tmp_path / "step"
        rc, summary, _ = _run_swgen("generate", "--part", part, "--csv", csv, "--out", out)
        results = _results(str(out))
        assert rc == 1 and summary["ok"] == 1 and summary["failed"] == 1
        assert "collides with row 1" in " ".join(results[1]["errors"])
        assert len(list(out.glob("*.stp"))) == 1

    def test_configuration_columns_abort_swgen_unless_skipped(self, tmp_path):
        part = self._any_part()
        d = _defaults(part)
        csv = tmp_path / "points.csv"
        csv.write_text("R,A,CC,VC,T,N\n" + _csv_row(d) + ",0,6\n")

        rc, payload, _ = _run_swgen("generate", "--part", part, "--csv", csv, "--out", tmp_path / "a")
        assert rc == 2 and "no defining equation" in payload.get("fatal", "")
        assert not list((tmp_path / "a").glob("*.stp"))

        rc, summary, _ = _run_swgen("generate", "--part", part, "--csv", csv, "--out", tmp_path / "b",
                                    "--skip-columns", "P,T,N")
        assert rc == 0 and summary["ok"] == 1
