"""
src/validate_constraints.py: comparing constraints with SwGen rebuild results (synthetic results files, no SolidWorks).
"""

import json

import numpy as np
import pandas as pd
import pytest

import constraints as C
import validate_constraints as V

OD, L, G = 40.0, 50.0, 3.0


def write_results(tmp_path, pts, built, extra_lines=()):
    path = tmp_path / "swgen_results.jsonl"
    with open(path, "w") as f:
        for i, (r, ok) in enumerate(zip(pts.to_dict("records"), built), start=1):
            f.write(json.dumps({
                "row": i, "params": {k: r[k] for k in ("R", "A", "CC", "VC")},
                "status": "ok" if ok else "failed",
                "errors": [] if ok else ["ForceRebuild3 returned false"],
                "rebuild_errors": [] if ok else [{"feature": "Wrap1", "code": 1}],
            }) + "\n")
        for line in extra_lines:
            f.write(line + "\n")
    return str(path)


def sat(con, pts, cfg=(0, 0)):
    return con.satisfied(pts["R"], pts["A"], pts["CC"], pts["VC"], OD, L, G)


class TestCompare:
    def test_constraints_that_match_solidworks_agree_everywhere(self, tmp_path):
        pts = V.random_box_points(400, seed=1)
        accepted = C.feasible_mask(pts.assign(T=0, N=0), OD, L, G).to_numpy()
        rep = V.compare(write_results(tmp_path, pts, accepted), (0, 0), OD, L, G)
        assert rep["agreement"] == rep["n"] == 400
        assert rep["accepted_but_failed"] == 0 and rep["rejected_but_built"] == 0
        assert rep["accepted"] == rep["accepted_and_built"] == int(accepted.sum()) > 0
        assert all(m["violated_by_built"] == 0 for m in rep["per_constraint"].values())
        assert rep["false_accepts"] == [] and rep["false_rejects"] == []

    def test_a_reversed_inequality_is_visible_in_the_report(self, tmp_path):
        """If SolidWorks builds exactly when E1 and E2 hold but E3 does NOT, E3 as written is backwards."""
        pts = V.random_box_points(600, seed=2)
        built = sat(C.E1, pts) & sat(C.E2, pts) & ~sat(C.E3, pts)
        rep = V.compare(write_results(tmp_path, pts, built), (0, 0), OD, L, G)
        assert rep["built"] == int(built.sum()) > 0
        assert rep["per_constraint"]["E3"]["violated_by_built"] == rep["built"]      # every built row violates E3
        assert rep["per_constraint"]["E1"]["violated_by_built"] == 0
        assert rep["per_constraint"]["E2"]["violated_by_built"] == 0
        assert rep["accepted_and_built"] == 0 and rep["rejected_but_built"] == rep["built"]

    def test_a_missing_constraint_shows_up_as_accepted_but_failed(self, tmp_path):
        pts = V.random_box_points(600, seed=3)
        accepted = C.feasible_mask(pts.assign(T=0, N=0), OD, L, G).to_numpy()
        built = accepted & (pts["A"].to_numpy() > 40)                         # reality also needs A > 40
        rep = V.compare(write_results(tmp_path, pts, built), (0, 0), OD, L, G)
        assert rep["accepted_but_failed"] > 0 and rep["rejected_but_built"] == 0
        assert {"row", "R", "A", "CC", "VC", "failed_features"} <= set(rep["false_accepts"][0])
        assert rep["false_accepts"][0]["failed_features"] == "Wrap1"

    def test_only_violation_of_failed_rows_is_attributed_per_constraint(self, tmp_path):
        pts = V.random_box_points(800, seed=4)
        accepted = C.feasible_mask(pts.assign(T=0, N=0), OD, L, G).to_numpy()
        rep = V.compare(write_results(tmp_path, pts, accepted), (0, 0), OD, L, G)
        total_only = sum(m["only_violation_of_failed"] for m in rep["per_constraint"].values())
        assert 0 < total_only <= rep["failed"]

    def test_hex_configuration_and_report_printing(self, tmp_path, capsys):
        pts = V.random_box_points(100, seed=5)
        accepted = C.feasible_mask(pts.assign(T=0, N=6), OD, L, G).to_numpy()
        rep = V.compare(write_results(tmp_path, pts, accepted), (6, 0), OD, L, G)
        assert set(rep["per_constraint"]) == {"C1", "C2", "C3"}
        V.print_report(rep)
        out = capsys.readouterr().out
        assert "Configuration (N=6, T=0)" in out and "agreement: 100/100" in out

    def test_rows_without_parameters_are_ignored_and_an_empty_file_is_an_error(self, tmp_path):
        pts = V.random_box_points(10, seed=6)
        junk = json.dumps({"row": 99, "status": "failed", "errors": ["malformed CSV row: expected 4 columns, found 3"]})
        path = write_results(tmp_path, pts, [True] * 10, extra_lines=[junk, ""])
        assert len(V.load_results(path)) == 10
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        with pytest.raises(ValueError):
            V.compare(str(empty), (0, 0), OD, L, G)

    def test_placeholder_configurations_cannot_be_compared(self, tmp_path):
        path = write_results(tmp_path, V.random_box_points(5, seed=7), [True] * 5)
        with pytest.raises(C.ConstraintsNotDefinedError):
            V.compare(path, (0, 1), OD, L, G)


class TestPointsAndCli:
    def test_random_box_points_are_in_the_box_and_reproducible(self):
        a, b = V.random_box_points(300, seed=8), V.random_box_points(300, seed=8)
        pd.testing.assert_frame_equal(a, b)
        for col, (lo, hi) in C.PARAM_BOUNDS.items():
            assert a[col].between(lo, hi).all(), col
        assert (a["CC"] % 1 == 0).all() and a["CC"].min() == 4 and a["CC"].max() == 22

    def test_cli_points_and_compare(self, tmp_path, capsys):
        points = tmp_path / "points.csv"
        assert V.main(["points", "--n", "50", "--out", str(points)]) == 0
        pts = pd.read_csv(points)
        assert list(pts.columns) == ["R", "A", "CC", "VC"] and len(pts) == 50

        results = write_results(tmp_path, pts, C.feasible_mask(pts.assign(T=0, N=0), OD, L, G).to_numpy())
        assert V.main(["compare", results, "--config", "0:0", "--OD", "40", "--L", "50", "--G", "3"]) == 0
        assert "agreement: 50/50" in capsys.readouterr().out
        assert V.main(["compare", results, "--config", "0:1", "--OD", "40", "--L", "50"]) == 2
        assert V.main(["compare", results, "--config", "junk", "--OD", "40", "--L", "50"]) == 2
