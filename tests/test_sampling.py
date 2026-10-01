"""
Sampler tests (src/sample.py): per-configuration streams, constraints, persistence, subsampling.

K always means "K points per configuration"; T and N are stratification factors, not Sobol dimensions.
"""

import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

import check_constraints as CK
import constraints as C
import sample as S

OD, L, G = 40.0, 50.0, 3.0
BOTH = [(6, 0), (0, 0)]


def feasible(df):
    return C.feasible_mask(df, OD, L, G)


def count(df, N, T):
    return int(((df["N"] == N) & (df["T"] == T)).sum())


class TestOutputShape:
    def test_default_is_the_legacy_single_configuration(self):
        df = S.generate_sobol(k=16, seed=1, OD=OD, L=L, G=G)
        assert list(df.columns) == ["R", "A", "CC", "VC", "T", "N"]
        assert len(df) == 16 and (df["N"] == 6).all() and (df["T"] == 0).all()

    @pytest.mark.parametrize("cfg", BOTH)
    def test_every_sampled_point_is_feasible_for_its_configuration(self, cfg):
        df = S.generate_sobol(k=32, seed=3, OD=OD, L=L, G=G, configs=[cfg])
        assert len(df) == 32
        assert (df["N"] == cfg[0]).all() and (df["T"] == cfg[1]).all()
        assert feasible(df).all()

    def test_k_is_per_configuration(self):
        df = S.generate_sobol(k=16, seed=2, OD=OD, L=L, G=G, configs=BOTH)
        assert len(df) == 32 and count(df, 6, 0) == 16 and count(df, 0, 0) == 16
        assert feasible(df).all()

    def test_values_stay_inside_the_design_box_and_integers_are_integers(self):
        df = S.generate_sobol(k=64, seed=4, OD=OD, L=L, G=G, configs=BOTH)
        for col, (lo, hi) in C.PARAM_BOUNDS.items():
            assert df[col].between(lo, hi).all(), col
        assert (df["CC"] % 1 == 0).all() and (df["VC"] % 1 == 0).all()
        assert (df["R"].round(2) == df["R"]).all() and (df["A"].round(2) == df["A"]).all()

    def test_unconstrained_sampling_ignores_feasibility_but_keeps_configuration_columns(self):
        df = S.generate_sobol(k=32, seed=5, configs=BOTH)
        assert len(df) == 64 and not feasible(df).all()

    def test_k_is_rounded_up_to_a_power_of_two(self, capsys):
        df = S.generate_sobol(k=10, seed=6, OD=OD, L=L, G=G, configs=[(6, 0)])
        assert len(df) == 16
        assert "not a power of 2" in capsys.readouterr().out


class TestPlaceholderConfigurations:
    @pytest.mark.parametrize("cfgs", [[(0, 1)], [(6, 1)], [(6, 0), (6, 1)]])
    def test_sampling_is_refused(self, cfgs):
        with pytest.raises(C.ConstraintsNotDefinedError):
            S.generate_sobol(k=8, seed=1, OD=OD, L=L, G=G, configs=cfgs)
        with pytest.raises(C.ConstraintsNotDefinedError):
            S.sample_with_constraints(8, OD, L, G, seed=1, configs=cfgs)
        with pytest.raises(C.ConstraintsNotDefinedError):
            S.generate_lhs_configs(8, seed=1, configs=cfgs)

    @pytest.mark.parametrize("cfg", [(6, 1), (0, 1)])
    def test_propagation_is_refused(self, cfg):
        with pytest.raises(C.ConstraintsNotDefinedError):
            S.propagate_domains(OD, L, G, cfg)

    def test_apply_constraints_never_silently_passes_or_drops_placeholder_rows(self):
        df = pd.DataFrame({"R": [3.0], "A": [60], "CC": [12], "VC": [5], "T": [1], "N": [6]})
        with pytest.raises(C.ConstraintsNotDefinedError):
            S.apply_constraints(df, OD, L, G)

    def test_cli_rejects_a_placeholder_configuration(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["sample.py", "--sobol", "8", "--configs", "0:1"])
        with pytest.raises(SystemExit, match="have not been derived"):
            S.main()


class TestGapAndConstraintFiltering:
    def test_gap_is_applied_even_without_propagation(self):
        """Regression: --G used to be ignored for Sobol unless --propagate was given (G silently 0)."""
        with_g = S.generate_sobol(k=128, seed=5, OD=OD, L=L, G=G, configs=[(6, 0)])
        assert feasible(with_g).all()

        without_g = S.generate_sobol(k=128, seed=5, OD=OD, L=L, G=0.0, configs=[(6, 0)])
        assert not feasible(without_g).all(), "G=0 sample should contain points that violate G=3"

    def test_gap_can_come_from_a_legacy_domains_dict(self):
        """optimizer.py / active_sampler.py pass G only through domains['G']."""
        domains = {"R": (2.0, 8.8), "A": (30.0, 90.0), "CC": (4, 22), "VC": (4, 10), "OD": OD, "L": L, "G": G}
        df = S.generate_sobol(k=64, OD=OD, L=L, domains=domains)
        assert feasible(df).all() and (df["N"] == 6).all()

    def test_apply_constraints_keeps_legacy_four_column_frames_four_columns(self):
        df = pd.DataFrame({"R": [3.0, 8.5], "A": [60, 60], "CC": [12, 12], "VC": [5, 5]})
        out = S.apply_constraints(df, OD, L, G)
        assert list(out.columns) == ["R", "A", "CC", "VC"] and out["R"].tolist() == [3.0]

    def test_apply_constraints_handles_empty_frames(self):
        out = S.apply_constraints(pd.DataFrame({"R": [], "CC": [], "VC": []}), OD, L, G)
        assert len(out) == 0

    def test_attempt_budget_shortfall_is_reported(self, capsys):
        df = S.generate_sobol(k=16, seed=1, OD=OD, L=L, G=G, max_attempts=1, configs=[(0, 0)])
        assert len(df) < 16
        assert "Warning" in capsys.readouterr().out


class TestReproducibility:
    def test_same_seed_same_sample_different_seed_different_sample(self):
        a = S.generate_sobol(k=16, seed=8, OD=OD, L=L, G=G, configs=BOTH)
        b = S.generate_sobol(k=16, seed=8, OD=OD, L=L, G=G, configs=BOTH)
        c = S.generate_sobol(k=16, seed=9, OD=OD, L=L, G=G, configs=BOTH)
        pd.testing.assert_frame_equal(a, b)
        assert not a[["R", "A"]].equals(c[["R", "A"]])

    def test_a_configuration_keeps_its_stream_whichever_others_are_selected(self):
        both = S.generate_sobol(k=16, seed=7, OD=OD, L=L, G=G, configs=BOTH)
        hexonly = S.generate_sobol(k=16, seed=7, OD=OD, L=L, G=G, configs=[(6, 0)])
        ellipse = S.generate_sobol(k=16, seed=7, OD=OD, L=L, G=G, configs=[(0, 0)])
        pd.testing.assert_frame_equal(both[both["N"] == 6].reset_index(drop=True), hexonly)
        pd.testing.assert_frame_equal(both[both["N"] == 0].reset_index(drop=True), ellipse)

    def test_configurations_do_not_share_a_point_stream(self):
        both = S.generate_sobol(k=32, seed=7, OD=OD, L=L, G=G, configs=BOTH)
        hex_rows = both[both["N"] == 6][["R", "A"]].to_numpy()
        ell_rows = both[both["N"] == 0][["R", "A"]].to_numpy()
        assert not np.array_equal(hex_rows, ell_rows)

    def test_legacy_stream_is_reproduced_up_to_the_documented_c2_difference(self):
        """(6, 0) with the same seed = the old sampler's stream, except that the new closed-form C2 (no |2R-a|
        branch) additionally accepts a few CC=4, small-R points the old rule rejected."""
        old = pd.read_csv(os.path.join(os.path.dirname(S.__file__), "src_data", "Sample_OD40L50G3_New.csv"))
        new = S.generate_sobol(k=1024, seed=42, OD=OD, L=L, G=G, configs=[(6, 0)],
                               domains={(6, 0): S.propagate_domains(OD, L, G, (6, 0))})
        key = ["R", "A", "CC", "VC"]
        old_keys = set(map(tuple, old[key].round(2).to_numpy()))
        new_keys = set(map(tuple, new[key].round(2).to_numpy()))
        # the sequence is a shifted-by-a-few-points copy of the old one: almost everything is shared
        assert len(old_keys & new_keys) / len(new_keys) > 0.98
        extra = new[~new[key].round(2).apply(tuple, axis=1).isin(old_keys)]
        # the only points the closed-form C2 adds are the small-R, CC=4, VC=9-10 corner (2R <= a branch)
        assert ((extra["CC"] == 4) & (extra["VC"] >= 9) & (extra["R"] < 2.4)).all()


class TestPropagation:
    def test_hex_and_ellipse_domains_for_the_standard_case(self):
        for cfg in BOTH:
            d = S.propagate_domains(OD, L, G, cfg)
            assert d["R"] == (2.0, 8.8) and d["CC"] == (4, 22) and d["VC"] == (4, 10) and d["A"] == (30.0, 90.0)

    def test_small_outer_diameter_tightens_cc_differently_per_configuration(self):
        hex_dom = S.propagate_domains(10.0, L, G, (6, 0))
        ell_dom = S.propagate_domains(10.0, L, G, (0, 0))
        assert hex_dom["CC"] == (4, 9)        # CC < pi*OD / (sqrt(3)*R_lo)
        assert ell_dom["CC"] == (4, 7)        # CC < pi*OD / (2*R_lo)

    @pytest.mark.parametrize("cfg", BOTH)
    def test_infeasible_geometry_is_reported(self, cfg):
        with pytest.raises(ValueError):
            S.propagate_domains(2.0, L, G, cfg)

    def test_sampling_inside_propagated_domains(self):
        domains = {cfg: S.propagate_domains(OD, L, G, cfg) for cfg in BOTH}
        df = S.generate_sobol(k=16, seed=3, OD=OD, L=L, G=G, domains=domains, configs=BOTH)
        assert len(df) == 32 and feasible(df).all()

    def test_a_flat_domains_dict_cannot_serve_several_configurations(self):
        flat = S.propagate_domains(OD, L, G, (6, 0))
        with pytest.raises(ValueError, match="mapping"):
            S.generate_sobol(k=8, seed=1, OD=OD, L=L, G=G, domains=flat, configs=BOTH)


class TestPersistence:
    def test_resume_continues_every_stream_without_repeating_points(self, tmp_path):
        inst = str(tmp_path / "run")
        first = S.generate_sobol(k=8, seed=7, instance=inst, OD=OD, L=L, G=G, configs=BOTH)
        state1 = json.load(open(inst + ".sobol_state.json"))
        second = S.generate_sobol(k=8, seed=7, instance=inst, OD=OD, L=L, G=G, configs=BOTH)
        state2 = json.load(open(inst + ".sobol_state.json"))

        assert len(first) == 16 and len(second) == 32
        pd.testing.assert_frame_equal(second.iloc[:16].reset_index(drop=True), first.reset_index(drop=True))
        for key in ("N6_T0", "N0_T0"):
            assert state2["configs"][key]["num_generated"] > state1["configs"][key]["num_generated"] > 0
        assert not second.duplicated(subset=["R", "A", "CC", "VC", "T", "N"]).any()
        assert state2["version"] == 2 and state2["seed"] == 7 and state2["G"] == G
        assert feasible(second).all()
        assert pd.read_csv(inst + ".csv").shape == (32, 6)

    def test_resume_ignores_a_different_seed_argument(self, tmp_path):
        a, b = str(tmp_path / "a"), str(tmp_path / "b")
        S.generate_sobol(k=8, seed=7, instance=a, OD=OD, L=L, G=G, configs=BOTH)
        for suffix in (".csv", ".sobol_state.json"):
            shutil.copy(a + suffix, b + suffix)
        ra = S.generate_sobol(k=8, seed=7, instance=a, OD=OD, L=L, G=G, configs=BOTH)
        rb = S.generate_sobol(k=8, seed=999, instance=b, OD=OD, L=L, G=G, configs=BOTH)
        pd.testing.assert_frame_equal(ra, rb)

    def test_a_new_configuration_can_join_an_existing_instance(self, tmp_path):
        inst = str(tmp_path / "grow")
        S.generate_sobol(k=8, seed=7, instance=inst, OD=OD, L=L, G=G, configs=[(6, 0)])
        full = S.generate_sobol(k=8, seed=7, instance=inst, OD=OD, L=L, G=G, configs=[(0, 0)])
        state = json.load(open(inst + ".sobol_state.json"))
        assert set(state["configs"]) == {"N6_T0", "N0_T0"}
        assert count(full, 6, 0) == 8 and count(full, 0, 0) == 8

    def test_version_1_state_and_four_column_csv_are_migrated(self, tmp_path):
        inst = str(tmp_path / "legacy")
        old = S.generate_sobol(k=8, seed=42, OD=OD, L=L, G=G, configs=[(6, 0)])
        old[["R", "A", "CC", "VC"]].to_csv(inst + ".csv", index=False)               # 4 columns, as before
        json.dump({"num_generated": 32, "seed": 42, "OD": OD, "L": L, "propagated": True,
                   "domains": {"R": [2.0, 8.8], "A": [30.0, 90.0], "CC": [4, 22], "VC": [4, 10]}},
                  open(inst + ".sobol_state.json", "w"))

        full = S.generate_sobol(k=8, seed=1234, instance=inst, OD=OD, L=L, G=G, configs=[(6, 0)])
        assert list(full.columns) == ["R", "A", "CC", "VC", "T", "N"] and len(full) == 16
        assert (full.iloc[:8]["T"] == 0).all() and (full.iloc[:8]["N"] == 6).all()

        state = json.load(open(inst + ".sobol_state.json"))
        assert state["version"] == 2 and state["seed"] == 42               # the original seed wins
        assert state["configs"]["N6_T0"]["num_generated"] >= 32 + 8

    def test_an_unseeded_instance_persists_a_seed_so_that_resume_is_possible(self, tmp_path):
        inst = str(tmp_path / "unseeded")
        S.generate_sobol(k=8, instance=inst, OD=OD, L=L, G=G, configs=[(6, 0)])
        seed = json.load(open(inst + ".sobol_state.json"))["seed"]
        assert isinstance(seed, int)
        S.generate_sobol(k=8, instance=inst, OD=OD, L=L, G=G, configs=[(6, 0)])
        assert json.load(open(inst + ".sobol_state.json"))["seed"] == seed


class TestLatinHypercube:
    def test_constrained_lhs_per_configuration(self):
        df = S.sample_with_constraints(20, OD, L, G, seed=2, configs=BOTH)
        assert len(df) == 40 and count(df, 6, 0) == 20 and count(df, 0, 0) == 20
        assert feasible(df).all()

    def test_default_lhs_is_legacy_hex(self):
        df = S.sample_with_constraints(10, OD, L, G, seed=2)
        assert (df["N"] == 6).all() and (df["T"] == 0).all() and len(df) == 10

    def test_unconstrained_lhs_has_configuration_columns(self):
        df = S.generate_lhs_configs(5, seed=1, configs=BOTH)
        assert len(df) == 10 and list(df.columns) == ["R", "A", "CC", "VC", "T", "N"]
        assert sorted(df["N"].unique()) == [0, 6]


class TestSubsampling:
    @pytest.fixture()
    def sample(self):
        return S.generate_sobol(k=32, seed=13, OD=OD, L=L, G=G, configs=BOTH)

    @pytest.mark.parametrize("strategy", ["random", "maxmin"])
    def test_subsample_is_spread_across_configurations(self, sample, strategy):
        sub = S.stratified_subsample(sample, 10, strategy=strategy, seed=1)
        assert len(sub) == 10 and count(sub, 6, 0) == 5 and count(sub, 0, 0) == 5

    @pytest.mark.parametrize("strategy", ["random", "maxmin"])
    def test_subsample_of_one_configuration_and_edge_sizes(self, sample, strategy):
        hex_only = sample[sample["N"] == 6].reset_index(drop=True)
        assert len(S.stratified_subsample(hex_only, 7, strategy=strategy, seed=1)) == 7
        assert len(S.stratified_subsample(sample, 1, strategy=strategy, seed=1)) == 1
        assert len(S.stratified_subsample(sample, len(sample), strategy=strategy)) == len(sample)

    def test_subsample_errors(self, sample):
        with pytest.raises(ValueError):
            S.stratified_subsample(sample, len(sample) + 1)
        with pytest.raises(ValueError):
            S.stratified_subsample(sample, 5, strategy="bogus")

    def test_subsample_works_on_legacy_four_column_frames(self, sample):
        legacy = sample[sample["N"] == 6][["R", "A", "CC", "VC"]].reset_index(drop=True)
        assert len(S.stratified_subsample(legacy, 6, strategy="maxmin")) == 6

    @pytest.mark.parametrize("n, sizes, expected", [
        (10, {"a": 30, "b": 10}, {"a": 8, "b": 2}),
        (5, {"a": 1, "b": 100}, {"a": 1, "b": 4}),
        (2, {"a": 50, "b": 50}, {"a": 1, "b": 1}),
        (1, {"a": 50, "b": 50}, None),
    ])
    def test_allocation(self, n, sizes, expected):
        alloc = S._allocate(n, sizes)
        assert sum(alloc.values()) == n
        assert all(0 <= alloc[k] <= sizes[k] for k in sizes)
        if expected is not None:
            assert alloc == expected


class TestCli:
    def test_end_to_end_sobol_and_check_constraints(self, monkeypatch, tmp_path, capsys):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["sample.py", "--sobol", "8", "--OD", "40", "--L", "50", "--G", "3",
                                          "--seed", "1", "--sobol-output", "out.csv"])
        S.main()
        df = pd.read_csv("out.csv")
        assert list(df.columns) == ["R", "A", "CC", "VC", "T", "N"] and len(df) == 16      # 8 x 2 default configs
        assert sorted(df["N"].unique()) == [0, 6] and (df["T"] == 0).all()

        out = CK.check_constraints("out.csv", G=3.0, L=50.0, OD=40.0)
        assert out is None                                           # nothing violated
        capsys.readouterr()

    def test_legacy_configs_flag(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["sample.py", "--k", "10", "--configs", "legacy", "--OD", "40",
                                          "--L", "50", "--G", "3", "--seed", "1", "--output", "lhs.csv"])
        S.main()
        df = pd.read_csv("lhs.csv")
        assert len(df) == 10 and (df["N"] == 6).all()

    def test_lhs_with_subsample(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["sample.py", "--k", "20", "--OD", "40", "--L", "50", "--G", "3",
                                          "--seed", "1", "--propagate", "--subsample", "6",
                                          "--subsample-strategy", "maxmin", "--output", "lhs.csv",
                                          "--subsample-output", "sub.csv"])
        S.main()
        sub = pd.read_csv("sub.csv")
        assert len(pd.read_csv("lhs.csv")) == 40 and len(sub) == 6
        assert sorted(sub["N"].unique()) == [0, 6]


class TestCheckConstraintsScript:
    def test_reports_violations_and_unchecked_rows(self, tmp_path, capsys):
        csv = tmp_path / "mixed.csv"
        pd.DataFrame([
            dict(R=3.0, A=60, CC=12, VC=5, T=0, N=6),          # ok
            dict(R=8.5, A=60, CC=12, VC=5, T=0, N=6),          # violates C1, C2, C3
            dict(R=5.1, A=60, CC=12, VC=5, T=0, N=0),          # ok
            dict(R=3.0, A=60, CC=12, VC=5, T=0, N=0),          # violates E3
            dict(R=3.0, A=60, CC=12, VC=5, T=1, N=6),          # placeholder -> unchecked
        ]).to_csv(csv, index=False)
        out = CK.check_constraints(str(csv), G=3.0, L=50.0, OD=40.0, output_path=str(tmp_path / "v.csv"))
        text = capsys.readouterr().out

        assert out["SampleRow"].tolist() == [2, 4, 5]
        assert out["Unchecked"].tolist() == [False, False, True]
        assert out.loc[out["SampleRow"] == 2, ["Violates_C1", "Violates_C2", "Violates_C3"]].iloc[0].tolist() == [True] * 3
        assert out.loc[out["SampleRow"] == 4, "Violates_E3"].iloc[0] == True        # noqa: E712
        assert "UNCHECKED" in text and (tmp_path / "v.csv").is_file()

    def test_legacy_csv_without_configuration_columns(self, tmp_path, capsys):
        csv = tmp_path / "legacy.csv"
        pd.DataFrame({"R": [3.0, 8.5], "A": [60, 60], "CC": [12, 12], "VC": [5, 5]}).to_csv(csv, index=False)
        out = CK.check_constraints(str(csv), G=3.0, L=50.0, OD=40.0, output_path=str(tmp_path / "v.csv"))
        assert out["SampleRow"].tolist() == [2]
        assert "legacy" in capsys.readouterr().out

    def test_bad_inputs_exit(self, tmp_path):
        with pytest.raises(SystemExit):
            CK.check_constraints(str(tmp_path / "nope.csv"), G=3.0, L=50.0, OD=40.0)
        csv = tmp_path / "bad.csv"
        pd.DataFrame({"R": [3.0]}).to_csv(csv, index=False)
        with pytest.raises(SystemExit):
            CK.check_constraints(str(csv), G=3.0, L=50.0, OD=40.0)
        csv2 = tmp_path / "badcfg.csv"
        pd.DataFrame({"R": [3.0], "CC": [12], "VC": [5], "N": [3]}).to_csv(csv2, index=False)
        with pytest.raises(SystemExit):
            CK.check_constraints(str(csv2), G=3.0, L=50.0, OD=40.0)
