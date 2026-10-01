"""
Surrogate pipeline tests (src/ml_models): six inputs [R, A, CC, VC, T, N], legacy compatibility, per-config metrics.

Everything runs on small synthetic CSVs in tmp_path - no confidential data, and nothing is written to models/.
"""

import argparse
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import constraints as C
from ml_models import data_loader as D
from ml_models import evaluate as EV
from ml_models import gp_model as GPM
from ml_models import mlp_model as M
from ml_models import predict as PR
from ml_models import train_gp, train_mlp

ALL_CONFIGS = [(6, 0), (0, 0), (6, 1), (0, 1)]


def synth(n_per_config, configs, seed=0, with_config_columns=True, extra_columns=True):
    """Synthetic processed-data frame: smooth targets that depend on the configuration."""
    rng = np.random.default_rng(seed)
    frames = []
    for N, T in configs:
        R, A = rng.uniform(2, 8.8, n_per_config), rng.uniform(30, 90, n_per_config)
        CC, VC = rng.integers(4, 23, n_per_config), rng.integers(4, 11, n_per_config)
        df = pd.DataFrame({
            "R": R.round(2), "A": A.round(2), "CC": CC, "VC": VC,
            "SEA": 2 + 0.3 * R - 0.02 * A + 0.05 * CC + 0.1 * VC + 1.5 * (N == 0) + 0.7 * T
                   + rng.normal(0, 0.05, n_per_config),
            "CFE": np.clip(0.6 + 0.02 * R + 0.1 * (N == 0) + rng.normal(0, 0.01, n_per_config), 0, 1),
        })
        if with_config_columns:
            df["T"], df["N"] = T, N
        if extra_columns:
            df["AUC"], df["Volume"], df["Obj"] = 1.0, 2.0, 3.0
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def write(tmp_path, name, df):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# Constants that are deliberately duplicated must not drift apart
# ---------------------------------------------------------------------------

class TestConstants:
    def test_feature_columns(self):
        assert D.FEATURE_COLS == ["R", "A", "CC", "VC", "T", "N"]
        assert D.FEATURE_COLS == C.PARAM_COLS + C.CONFIG_COLS
        assert D.LEGACY_FEATURE_COLS == C.PARAM_COLS
        assert D.CONFIG_COLS == C.CONFIG_COLS

    def test_configuration_values_match_constraints_module(self):
        assert D.VALID_N == C.VALID_N and D.VALID_T == C.VALID_T
        assert D.LEGACY_CONFIG_DEFAULTS == {"N": C.LEGACY_CONFIG[0], "T": C.LEGACY_CONFIG[1]}

    def test_predict_bounds_match_the_design_box(self):
        assert PR._BOUNDS == {k: tuple(float(v) for v in C.PARAM_BOUNDS[k]) for k in ("R", "A", "CC", "VC")}

    def test_model_widths_follow_the_feature_list(self):
        assert M.N_INPUTS == GPM.N_FEATURES == len(D.FEATURE_COLS) == 6
        assert M.LAYER_SIZES == [6, 64, 32, 16, 2]

    def test_parameter_counts(self):
        net = M.SurrogateNet()
        assert net.total_params() == 3090
        net.freeze_until(M.N_FREEZE_FOR_FINETUNE)
        assert net.trainable_params() == 562                     # unchanged by the two extra inputs
        assert M.SurrogateNet([4, 64, 32, 16, 2]).total_params() == 2962


# ---------------------------------------------------------------------------
# Reading data
# ---------------------------------------------------------------------------

class TestReadPdCsv:
    def test_legacy_csv_defaults_to_hex_untwisted(self, tmp_path, capsys):
        path = write(tmp_path, "legacy.csv", synth(10, [(6, 0)], with_config_columns=False))
        df = D.read_pd_csv(path)
        assert list(df.columns) == D.FEATURE_COLS + D.TARGET_COLS
        assert (df["N"] == 6).all() and (df["T"] == 0).all() and len(df) == 10
        assert "assuming N=6" in capsys.readouterr().out

    def test_bang_suffix_and_numeric_strings_are_coerced(self, tmp_path):
        df = synth(3, [(0, 0)])
        df["N"] = ["0!", "0", "0"]                    # PD files mark modified-mesh variants as '0!'
        assert D.read_pd_csv(write(tmp_path, "bang.csv", df))["N"].tolist() == [0, 0, 0]

    @pytest.mark.parametrize("column, value", [("N", 8), ("N", 3), ("T", 2), ("T", -1), ("N", "x")])
    def test_invalid_configuration_values_raise(self, tmp_path, column, value):
        df = synth(4, [(6, 0)])
        df[column] = df[column].astype(object)
        df.loc[1, column] = value
        with pytest.raises(ValueError, match=column):
            D.read_pd_csv(write(tmp_path, "bad.csv", df))

    def test_missing_configuration_value_raises_but_missing_feature_drops_the_row(self, tmp_path):
        df = synth(4, [(6, 0)])
        nan_n = df.copy()
        nan_n.loc[1, "N"] = np.nan
        with pytest.raises(ValueError, match="N"):
            D.read_pd_csv(write(tmp_path, "nan_n.csv", nan_n))
        nan_r = df.copy()
        nan_r.loc[2, "R"] = np.nan
        assert len(D.read_pd_csv(write(tmp_path, "nan_r.csv", nan_r))) == 3

    def test_missing_required_column_and_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="SEA"):
            D.read_pd_csv(write(tmp_path, "x.csv", synth(4, [(6, 0)]).drop(columns=["SEA"])))
        with pytest.raises(FileNotFoundError):
            D.read_pd_csv(str(tmp_path / "nope.csv"))

    def test_several_files_are_concatenated(self, tmp_path):
        a = write(tmp_path, "a.csv", synth(5, [(6, 0)]))
        b = write(tmp_path, "b.csv", synth(7, [(0, 0)], seed=1))
        df = D.read_pd_csv([a, b])
        assert len(df) == 12 and sorted(df["N"].unique()) == [0, 6]


# ---------------------------------------------------------------------------
# Splitting and scaling
# ---------------------------------------------------------------------------

class TestLoadData:
    def test_shapes_scalers_and_labels(self, tmp_path):
        lf = write(tmp_path, "lf.csv", synth(80, ALL_CONFIGS))
        hf = write(tmp_path, "hf.csv", synth(15, ALL_CONFIGS, seed=1))
        s = D.load_data(seed=0, lf_csv=lf, hf_csv=hf)
        assert s.X_lf_train.shape[1] == s.X_hf_test.shape[1] == 6
        assert len(s.X_lf_train) + len(s.X_lf_val) == 320 and len(s.X_hf_finetune) + len(s.X_hf_test) == 60
        assert s.feature_cols == D.FEATURE_COLS
        for labels, X in ((s.config_lf_train, s.X_lf_train), (s.config_lf_val, s.X_lf_val),
                          (s.config_hf_finetune, s.X_hf_finetune), (s.config_hf_test, s.X_hf_test)):
            assert len(labels) == len(X)
        # scalers fit on LF train only
        assert np.allclose(s.X_lf_train.mean(axis=0), 0, atol=1e-9)
        assert np.allclose(s.x_scaler.inverse_transform(s.X_lf_train)[:, 5].min(), 0)

    def test_splits_are_stratified_by_configuration(self, tmp_path):
        lf = write(tmp_path, "lf.csv", synth(40, ALL_CONFIGS))
        hf = write(tmp_path, "hf.csv", synth(15, ALL_CONFIGS, seed=1))
        s = D.load_data(seed=3, lf_csv=lf, hf_csv=hf)
        expected = {"N6_T0", "N0_T0", "N6_T1", "N0_T1"}
        for labels in (s.config_lf_train, s.config_lf_val, s.config_hf_finetune, s.config_hf_test):
            assert set(labels) == expected
        assert sorted(np.unique(s.config_hf_test, return_counts=True)[1]) == [3, 3, 3, 3]

    def test_stratification_falls_back_when_a_configuration_is_too_small(self, tmp_path, capsys):
        lf = write(tmp_path, "lf.csv", synth(40, [(6, 0), (0, 0)]))
        hf_df = pd.concat([synth(30, [(6, 0), (0, 0)], seed=1), synth(1, [(0, 1)], seed=2)], ignore_index=True)
        s = D.load_data(seed=0, lf_csv=lf, hf_csv=write(tmp_path, "hf.csv", hf_df))
        assert "cannot stratify the HF split" in capsys.readouterr().out
        assert len(s.X_hf_test) == 13

    def test_legacy_data_gives_constant_zero_configuration_inputs(self, tmp_path):
        lf = write(tmp_path, "lf.csv", synth(60, [(6, 0)], with_config_columns=False))
        hf = write(tmp_path, "hf.csv", synth(20, [(6, 0)], seed=1, with_config_columns=False))
        s = D.load_data(seed=0, lf_csv=lf, hf_csv=hf)
        assert np.all(s.X_lf_train[:, 4:] == 0) and np.all(s.X_hf_test[:, 4:] == 0)
        assert s.x_scaler.scale_[4] == 1.0 and s.x_scaler.scale_[5] == 1.0    # zero variance -> scale 1
        assert np.isfinite(s.x_scaler.transform([[3, 60, 12, 5, 1, 0]])).all()  # T=1 / N=0 later on stay finite

    def test_legacy_data_splits_exactly_like_the_four_input_pipeline(self, tmp_path):
        lf_df = synth(100, [(6, 0)], with_config_columns=False)
        hf_df = synth(40, [(6, 0)], seed=1, with_config_columns=False)
        s = D.load_data(seed=42, lf_csv=write(tmp_path, "lf.csv", lf_df), hf_csv=write(tmp_path, "hf.csv", hf_df))
        X_tr, X_val, Y_tr, Y_val = train_test_split(
            lf_df[D.LEGACY_FEATURE_COLS].values.astype(float), lf_df[D.TARGET_COLS].values.astype(float),
            test_size=0.2, random_state=42)
        assert np.allclose(s.x_scaler.inverse_transform(s.X_lf_train)[:, :4], X_tr)
        assert np.allclose(s.y_scaler.inverse_transform(s.Y_lf_train), Y_tr)
        assert np.allclose(s.x_scaler.inverse_transform(s.X_lf_val)[:, :4], X_val)
        # and the first four standardised columns are the legacy standardisation
        legacy_scaler = StandardScaler().fit(X_tr)
        assert np.allclose(s.X_lf_train[:, :4], legacy_scaler.transform(X_tr))

    def test_missing_default_files_report_a_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            D.load_data(lf_csv=str(tmp_path / "no_lf.csv"), hf_csv=str(tmp_path / "no_hf.csv"))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TestModels:
    def test_mlp_forward_and_roundtrip_for_six_and_legacy_four_inputs(self, tmp_path):
        for widths in ([6, 64, 32, 16, 2], [4, 64, 32, 16, 2]):
            net = M.SurrogateNet(widths)
            path = str(tmp_path / f"net{widths[0]}.pt")
            net.save(path, metadata={"feature_cols": D.FEATURE_COLS[: widths[0]]})
            loaded = M.SurrogateNet.load(path)
            x = torch.rand(5, widths[0])
            assert loaded(x).shape == (5, 2)
            assert torch.allclose(net(x), loaded(x))
            assert json.load(open(os.path.splitext(path)[0] + ".json"))["layer_sizes"] == widths

    def test_gp_with_six_features_roundtrips(self, tmp_path):
        rng = np.random.default_rng(0)
        X, Y = rng.normal(size=(30, 6)), rng.normal(size=(30, 2))
        gp = GPM.MultiFidelityGP(n_restarts=0)
        assert gp.n_features == 6
        gp.fit_lf(X, Y)
        gp.fit_hf_correction(X[:12], Y[:12] + 0.3)
        assert gp.gp_sea_lf.kernel_.k1.k2.length_scale.shape == (6,)      # one ARD length scale per input
        gp.save(str(tmp_path / "gp"))
        loaded = GPM.MultiFidelityGP.load(str(tmp_path / "gp"))
        assert loaded.n_features == 6
        assert np.allclose(gp.predict(X[:5])[0], loaded.predict(X[:5])[0])

    def test_gp_metadata_without_n_features_is_a_legacy_four_input_model(self, tmp_path):
        rng = np.random.default_rng(1)
        gp = GPM.MultiFidelityGP(n_restarts=0, n_features=4)
        gp.fit_lf(rng.normal(size=(20, 4)), rng.normal(size=(20, 2)))
        gp.save(str(tmp_path / "gp"))
        meta_path = tmp_path / "gp" / "gp_metadata.json"
        meta = json.load(open(meta_path))
        del meta["n_features"]
        json.dump(meta, open(meta_path, "w"))
        loaded = GPM.MultiFidelityGP.load(str(tmp_path / "gp"))
        assert loaded.n_features == 4
        # the not-yet-fitted delta GPs are rebuilt with the legacy width (an unfitted kernel keeps a plain list)
        assert np.shape(loaded.gp_sea_delta.kernel.k1.k2.length_scale) == (4,)


# ---------------------------------------------------------------------------
# Prediction interface
# ---------------------------------------------------------------------------

def make_models_dir(path, n_features, with_gp=False):
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(2, 8.8, 60), rng.uniform(30, 90, 60), rng.integers(4, 23, 60),
                         rng.integers(4, 11, 60), rng.integers(0, 2, 60), rng.choice([0, 6], 60)])[:, :n_features]
    x_sc, y_sc = StandardScaler().fit(X), StandardScaler().fit(rng.normal(size=(60, 2)))
    joblib.dump({"x_scaler": x_sc, "y_scaler": y_sc, "feature_cols": D.FEATURE_COLS[:n_features]},
                os.path.join(path, "scalers.pkl"))
    torch.manual_seed(0)
    M.SurrogateNet([n_features, 8, 4, 2]).save(os.path.join(path, "mlp_finetuned_hf.pt"))
    if with_gp:
        gp = GPM.MultiFidelityGP(n_restarts=0, n_features=n_features)
        Xs = x_sc.transform(X)
        gp.fit_lf(Xs[:30], rng.normal(size=(30, 2)))
        gp.fit_hf_correction(Xs[:10], rng.normal(size=(10, 2)))
        gp.save(os.path.join(path, "gp"))
    return str(path)


X4 = np.array([[3.5, 60.0, 12.0, 6.0], [5.0, 45.0, 8.0, 5.0]])
LEGACY_TN = np.array([[0.0, 6.0], [0.0, 6.0]])


class TestPredict:
    def test_six_column_input_and_padding_of_legacy_four_column_input(self, tmp_path):
        d = make_models_dir(tmp_path, 6, with_gp=True)
        explicit = np.hstack([X4, LEGACY_TN])
        sea6, cfe6 = PR.predict_mlp(explicit, models_dir=d)
        with pytest.warns(UserWarning, match="legacy configuration"):
            sea4, cfe4 = PR.predict_mlp(X4, models_dir=d)
        assert np.allclose(sea4, sea6) and np.allclose(cfe4, cfe6)

        gp6 = PR.predict_gp(explicit, models_dir=d)
        with pytest.warns(UserWarning, match="legacy configuration"):
            gp4 = PR.predict_gp(X4, models_dir=d)
        assert all(np.allclose(a, b) for a, b in zip(gp4, gp6))
        assert (gp6[1] > 0).all() and (gp6[3] > 0).all()

    def test_other_configurations_change_the_prediction(self, tmp_path):
        d = make_models_dir(tmp_path, 6)
        base = np.hstack([X4, LEGACY_TN])
        other = np.hstack([X4, np.array([[0.0, 0.0], [1.0, 6.0]])])
        assert not np.allclose(PR.predict_mlp(base, models_dir=d)[0], PR.predict_mlp(other, models_dir=d)[0])

    def test_categorical_inputs_are_validated_strictly(self, tmp_path):
        d = make_models_dir(tmp_path, 6)
        for tn in ([0.0, 3.0], [2.0, 6.0], [0.0, float("nan")], [0.5, 6.0]):
            with pytest.raises(ValueError, match="must contain only"):
                PR.predict_mlp(np.hstack([X4, np.tile(tn, (2, 1))]), models_dir=d)

    def test_wrong_number_of_columns(self, tmp_path):
        d = make_models_dir(tmp_path, 6)
        for bad in (np.ones((2, 5)), np.ones((2, 3)), np.ones((2, 7))):
            with pytest.raises(ValueError, match="columns"):
                PR.predict_mlp(bad, models_dir=d)

    def test_continuous_inputs_outside_the_domain_warn_but_predict(self, tmp_path):
        d = make_models_dir(tmp_path, 6)
        X = np.hstack([np.array([[12.0, 60.0, 12.0, 6.0]]), [[0.0, 6.0]]])
        with pytest.warns(UserWarning, match="outside the training domain"):
            sea, cfe = PR.predict_mlp(X, models_dir=d)
        assert np.isfinite(sea).all()

    def test_a_single_row_vector_is_accepted(self, tmp_path):
        d = make_models_dir(tmp_path, 6)
        sea, cfe = PR.predict_mlp([3.5, 60.0, 12.0, 6.0, 0, 6], models_dir=d)
        assert sea.shape == (1,) and cfe.shape == (1,)

    def test_legacy_four_input_model_only_serves_the_legacy_configuration(self, tmp_path):
        d = make_models_dir(tmp_path, 4)
        with warnings.catch_warnings():
            warnings.simplefilter("error")                        # 4 columns on a 4-input model: no warning
            sea4, _ = PR.predict_mlp(X4, models_dir=d)
        sea6, _ = PR.predict_mlp(np.hstack([X4, LEGACY_TN]), models_dir=d)
        assert np.allclose(sea4, sea6)
        with pytest.raises(ValueError, match="legacy 4-input"):
            PR.predict_mlp(np.hstack([X4, np.array([[0.0, 0.0], [0.0, 6.0]])]), models_dir=d)

    def test_objective_uses_the_padded_input_too(self, tmp_path):
        d = make_models_dir(tmp_path, 6, with_gp=True)
        with pytest.warns(UserWarning):
            obj = PR.predict_objective(X4, k=1.0, model="mlp", models_dir=d)
        assert obj.shape == (2,)


# ---------------------------------------------------------------------------
# Model-directory guard and metrics
# ---------------------------------------------------------------------------

class TestGuardAndMetrics:
    def _write_scalers(self, path, n_features):
        x_sc = StandardScaler().fit(np.random.default_rng(0).normal(size=(10, n_features)))
        joblib.dump({"x_scaler": x_sc, "y_scaler": StandardScaler().fit(np.zeros((3, 2)) + [[0, 1], [1, 0], [2, 2]])},
                    os.path.join(path, "scalers.pkl"))

    def test_guard_refuses_to_overwrite_a_model_with_a_different_input_width(self, tmp_path):
        self._write_scalers(tmp_path, 4)
        assert D.existing_model_n_features(str(tmp_path)) == 4
        with pytest.raises(SystemExit, match="4-input"):
            D.guard_model_dir(str(tmp_path), 6)
        D.guard_model_dir(str(tmp_path), 6, force=True)
        D.guard_model_dir(str(tmp_path), 4)

    def test_guard_allows_empty_and_unreadable_directories(self, tmp_path):
        D.guard_model_dir(str(tmp_path), 6)
        (tmp_path / "scalers.pkl").write_text("not a pickle")
        assert D.existing_model_n_features(str(tmp_path)) is None
        D.guard_model_dir(str(tmp_path), 6)

    def test_feature_cols_default_to_legacy_when_not_recorded(self, tmp_path):
        self._write_scalers(tmp_path, 4)
        assert D.load_feature_cols(str(tmp_path)) == D.LEGACY_FEATURE_COLS

    def test_metrics_by_config_skips_single_row_groups(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=(10, 2))
        labels = np.array(["N6_T0"] * 5 + ["N0_T0"] * 4 + ["N0_T1"])
        out = EV.compute_metrics_by_config(y, y, labels)
        assert set(out) == {"N6_T0", "N0_T0"}
        assert out["N6_T0"]["n"] == 5 and out["N0_T0"]["SEA"]["R2"] == 1.0

    def test_printing_metrics_by_config_is_quiet_when_there_is_nothing_to_show(self, capsys):
        EV.print_metrics_by_config({"MLP": {}})
        assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Training entry points (tiny, into tmp_path)
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_plots(monkeypatch):
    """set_plots_dir() mutates module state; restore it so later tests (and models/plots) are untouched."""
    monkeypatch.setattr(EV, "_PLOTS_DIR", EV._PLOTS_DIR)


class TestTrainingScripts:
    def _data(self, tmp_path, n_lf, n_hf):
        cfgs = [(6, 0), (0, 0)]
        return (write(tmp_path, "lf.csv", synth(n_lf, cfgs, seed=1)),
                write(tmp_path, "hf.csv", synth(n_hf, cfgs, seed=2)))

    def test_mlp_training_writes_six_input_artifacts_and_per_config_metrics(self, tmp_path, isolated_plots):
        lf, hf = self._data(tmp_path, 80, 30)
        models = tmp_path / "models"
        args = argparse.Namespace(phase="both", seed=0, epochs_p1=3, epochs_p2=3, models_dir=str(models),
                                  lf_csv=[lf], hf_csv=[hf], force=False)
        train_mlp.main(args)

        for f in ("mlp_pretrained_lf.pt", "mlp_finetuned_hf.pt", "scalers.pkl", "mlp_comparison_metrics.json",
                  "mlp_comparison_by_config.json", os.path.join("plots", "Parity_MLP_LF_only.png"),
                  os.path.join("plots", "LC_Phase1_MLP.png")):
            assert (models / f).is_file(), f
        meta = json.load(open(models / "mlp_finetuned_hf.json"))
        assert meta["layer_sizes"][0] == 6 and meta["feature_cols"] == D.FEATURE_COLS
        assert set(meta["hf_test_by_config"]) == {"N6_T0", "N0_T0"}
        by_cfg = json.load(open(models / "mlp_comparison_by_config.json"))
        assert set(by_cfg) == {"MLP (LF only)", "MLP (TL)"} and set(by_cfg["MLP (TL)"]) == {"N6_T0", "N0_T0"}
        assert D.existing_model_n_features(str(models)) == 6
        assert D.load_feature_cols(str(models)) == D.FEATURE_COLS
        assert EV._PLOTS_DIR == str(models / "plots")

    def test_mlp_training_refuses_to_overwrite_a_legacy_model_dir(self, tmp_path, isolated_plots):
        lf, hf = self._data(tmp_path, 40, 20)
        models = tmp_path / "models"
        models.mkdir()
        x_sc = StandardScaler().fit(np.random.default_rng(0).normal(size=(10, 4)))
        joblib.dump({"x_scaler": x_sc, "y_scaler": StandardScaler().fit(np.ones((2, 2)) * [[1, 2], [3, 4]])},
                    models / "scalers.pkl")
        args = argparse.Namespace(phase=1, seed=0, epochs_p1=1, epochs_p2=1, models_dir=str(models),
                                  lf_csv=[lf], hf_csv=[hf], force=False)
        with pytest.raises(SystemExit, match="4-input"):
            train_mlp.main(args)
        assert D.existing_model_n_features(str(models)) == 4          # untouched

    def test_gp_training_writes_six_feature_artifacts(self, tmp_path, isolated_plots):
        lf, hf = self._data(tmp_path, 30, 15)
        models = tmp_path / "models"
        args = argparse.Namespace(phase="both", seed=0, restarts=0, models_dir=str(models),
                                  lf_csv=[lf], hf_csv=[hf], force=False)
        train_gp.main(args)

        meta = json.load(open(models / "gp" / "gp_metadata.json"))
        assert meta["n_features"] == 6 and meta["is_hf_fitted"] is True
        assert (models / "gp_comparison_metrics.json").is_file()
        assert set(json.load(open(models / "gp_comparison_by_config.json"))["GP (TL)"]) == {"N6_T0", "N0_T0"}
        assert (models / "plots" / "Parity_GP_TL.png").is_file()

        sea, sea_std, cfe, cfe_std = PR.predict_gp(np.array([[3.5, 60.0, 12.0, 6.0, 0, 0]]), models_dir=str(models))
        assert np.isfinite([sea, sea_std, cfe, cfe_std]).all()
