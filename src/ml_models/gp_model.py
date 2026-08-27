"""
Multi-fidelity Gaussian Process surrogate using the Kennedy-O'Hagan (2000)
additive correction framework.

Architecture
------------
Two independent GPs (one for SEA, one for CFE) are maintained at each fidelity
level.  The multi-fidelity prediction decomposes as:

    mu_HF(x)  = mu_LF(x)  + mu_delta(x)
    var_HF(x) = var_LF(x) + var_delta(x)      (independence assumption)

where delta(x_i) = Y_HF(x_i) - GP_LF.predict(x_i) is the fidelity residual.

Kernel
------
C(amplitude) * RBF(length_scale=[l_R, l_A, l_CC, l_VC]) + WhiteKernel(noise)

A separate length scale per input dimension (ARD — Automatic Relevance
Determination) allows the kernel to discover that CC and VC (integer-valued)
contribute differently than R and A (continuous floats).  Hyperparameters are
optimized by maximizing the log marginal likelihood (Rasmussen & Williams, 2006,
§5.4.1) using L-BFGS-B with multiple random restarts.

Why two independent GPs and not a single scalarized GP?
-------------------------------------------------------
Predicting the scalarized objective Obj = SEA - k*(1-CFE) directly conflates
two separate physical quantities with different noise structures.  Keeping SEA
and CFE separate allows inspecting each output's uncertainty independently,
enables flexible post-hoc definition of the k weighting, and matches the MLP's
two-output structure for a fair comparison.  The objective is always computable
as a derived quantity from the two GP predictions.

References
----------
Kennedy, M.C., & O'Hagan, A. (2000). Predicting the output from a complex
computer code when fast approximations are available. Biometrika, 87(1), 1-13.

Rasmussen, C.E., & Williams, C.K.I. (2006). Gaussian Processes for Machine
Learning. MIT Press.  (Chapters 2, 4, 5)

Forrester, A.I.J., Sóbester, A., & Keane, A.J. (2007). Multi-fidelity
optimization via surrogate modelling. Proc. Royal Society A, 463, 3251-3269.

Álvarez, M.A., Rosasco, L., & Lawrence, N.D. (2012). Kernels for vector-valued
functions: A review. Foundations and Trends in ML, 4(3), 195-266.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import joblib
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel

N_FEATURES = 4   # R, A, CC, VC


def _make_kernel(amplitude: float = 1.0, noise: float = 0.01) -> object:
    """
    Build one ARD-RBF + WhiteKernel instance.

    A fresh kernel object is required per GaussianProcessRegressor because
    sklearn's GP optimizer mutates the kernel in-place.  Using the same object
    for multiple GPs would corrupt their hyperparameter optimization.

    amplitude : starting ConstantKernel value (LF: 1.0, delta: 0.5)
    noise     : starting noise level (both models: 0.01 in standardized units)
    """
    return (
        C(amplitude, constant_value_bounds=(1e-3, 1e3))
        * RBF(length_scale=[1.0] * N_FEATURES, length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=noise, noise_level_bounds=(1e-5, 1.0))
    )


class MultiFidelityGP:
    """
    Two-output multi-fidelity GP surrogate implementing the Kennedy-O'Hagan (2000)
    additive correction framework.

    Usage (typical)
    ---------------
    >>> gp = MultiFidelityGP()
    >>> gp.fit_lf(X_lf_std, Y_lf_std)         # Phase 1
    >>> gp.fit_hf_correction(X_hf_std, Y_hf_std)  # Phase 2
    >>> sea_mu, sea_std, cfe_mu, cfe_std = gp.predict(X_new_std, return_std=True)
    >>> gp.save("models/gp")
    """

    def __init__(self, n_restarts: int = 5, alpha: float = 1e-6):
        """
        Parameters
        ----------
        n_restarts : number of L-BFGS-B restarts for kernel hyperparameter
                     optimization.  More restarts improve the chance of finding
                     the global optimum of the log marginal likelihood at the
                     cost of longer fitting time.  Default 5 is adequate for
                     n ≈ 100–1000 points.
        alpha      : diagonal jitter added to the Gram matrix for numerical
                     stability (equivalent to a minimum noise floor).
                     1e-6 is sklearn's recommended default.
        """
        self.n_restarts = n_restarts
        self.alpha = alpha
        self._build_gps()

    def _build_gps(self) -> None:
        kw = dict(n_restarts_optimizer=self.n_restarts, alpha=self.alpha, normalize_y=False)
        # LF GPs — amplitude=1.0 matches standardized target variance
        self.gp_sea_lf    = GaussianProcessRegressor(kernel=_make_kernel(1.0, 0.01), **kw)
        self.gp_cfe_lf    = GaussianProcessRegressor(kernel=_make_kernel(1.0, 0.01), **kw)
        # Correction GPs — amplitude=0.5 reflects that delta is typically smaller
        # than the raw response; this gives the optimizer a better starting point.
        self.gp_sea_delta = GaussianProcessRegressor(kernel=_make_kernel(0.5, 0.01), **kw)
        self.gp_cfe_delta = GaussianProcessRegressor(kernel=_make_kernel(0.5, 0.01), **kw)
        self.is_lf_fitted = False
        self.is_hf_fitted = False
        self._lf_kernel_str:    dict[str, str] = {}
        self._delta_kernel_str: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit_lf(self, X_lf: np.ndarray, Y_lf: np.ndarray) -> None:
        """
        Fit GP_LF_SEA and GP_LF_CFE on the low-fidelity (shell) training data.

        Parameters
        ----------
        X_lf : (n_lf, 4) standardized inputs  [R, A, CC, VC]
        Y_lf : (n_lf, 2) standardized targets  [SEA, CFE]
        """
        print(f"[gp] Fitting LF-SEA GP  (n={len(X_lf)}) ...")
        self.gp_sea_lf.fit(X_lf, Y_lf[:, 0])
        print(f"[gp] Fitting LF-CFE GP  (n={len(X_lf)}) ...")
        self.gp_cfe_lf.fit(X_lf, Y_lf[:, 1])
        self.is_lf_fitted = True
        self._lf_kernel_str = {
            "SEA": str(self.gp_sea_lf.kernel_),
            "CFE": str(self.gp_cfe_lf.kernel_),
        }
        print(f"  SEA kernel: {self.gp_sea_lf.kernel_}")
        print(f"  CFE kernel: {self.gp_cfe_lf.kernel_}")

    def compute_deltas(self, X_hf: np.ndarray, Y_hf: np.ndarray) -> np.ndarray:
        """
        Compute fidelity residuals  delta_i = Y_HF_i - GP_LF.predict(X_HF_i).

        This works for all 123 HF points — including the 5 that have no exact
        counterpart in the LF dataset — because GP_LF.predict() generalizes
        across the input space rather than performing a table lookup.

        Returns
        -------
        (n_hf, 2) array  [delta_SEA, delta_CFE]  in standardized units
        """
        if not self.is_lf_fitted:
            raise RuntimeError("Call fit_lf() before compute_deltas().")

        sea_lf = self.gp_sea_lf.predict(X_hf)
        cfe_lf = self.gp_cfe_lf.predict(X_hf)
        delta_sea = Y_hf[:, 0] - sea_lf
        delta_cfe = Y_hf[:, 1] - cfe_lf

        print(
            f"[gp] Delta SEA (std): mean={delta_sea.mean():.4f}, "
            f"std={delta_sea.std():.4f}, "
            f"range=[{delta_sea.min():.4f}, {delta_sea.max():.4f}]"
        )
        print(
            f"[gp] Delta CFE (std): mean={delta_cfe.mean():.4f}, "
            f"std={delta_cfe.std():.4f}, "
            f"range=[{delta_cfe.min():.4f}, {delta_cfe.max():.4f}]"
        )
        return np.column_stack([delta_sea, delta_cfe])

    def fit_hf_correction(self, X_hf: np.ndarray, Y_hf: np.ndarray) -> None:
        """
        Compute fidelity deltas then fit the correction GPs.

        Parameters
        ----------
        X_hf : (n_hf, 4) standardized HF fine-tune inputs
        Y_hf : (n_hf, 2) standardized HF fine-tune targets
        """
        deltas = self.compute_deltas(X_hf, Y_hf)
        print(f"[gp] Fitting delta-SEA GP  (n={len(X_hf)}) ...")
        self.gp_sea_delta.fit(X_hf, deltas[:, 0])
        print(f"[gp] Fitting delta-CFE GP  (n={len(X_hf)}) ...")
        self.gp_cfe_delta.fit(X_hf, deltas[:, 1])
        self.is_hf_fitted = True
        self._delta_kernel_str = {
            "SEA": str(self.gp_sea_delta.kernel_),
            "CFE": str(self.gp_cfe_delta.kernel_),
        }
        print(f"  delta-SEA kernel: {self.gp_sea_delta.kernel_}")
        print(f"  delta-CFE kernel: {self.gp_cfe_delta.kernel_}")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False,
        use_correction: bool = True,
    ) -> tuple:
        """
        Predict SEA and CFE (in standardized units) at query locations X.

        Parameters
        ----------
        X              : (n, 4) standardized inputs
        return_std     : if True, also return posterior standard deviations
        use_correction : if False, return LF-only predictions (baseline comparison)

        Returns
        -------
        return_std=False : (sea_mean, cfe_mean)
        return_std=True  : (sea_mean, sea_std, cfe_mean, cfe_std)

        Variance combination (Kennedy & O'Hagan, 2000)
        -----------------------------------------------
        var_HF(x) = var_LF(x) + var_delta(x)
        The two GPs (LF and delta) are conditioned on disjoint information
        (LF data and HF residuals respectively), making their posteriors
        approximately independent — a standard assumption in the multi-fidelity
        literature (Forrester et al., 2007).
        """
        if not self.is_lf_fitted:
            raise RuntimeError("Call fit_lf() before predict().")

        sea_mu, sea_std = self.gp_sea_lf.predict(X, return_std=True)
        cfe_mu, cfe_std = self.gp_cfe_lf.predict(X, return_std=True)

        if use_correction and self.is_hf_fitted:
            d_sea_mu, d_sea_std = self.gp_sea_delta.predict(X, return_std=True)
            d_cfe_mu, d_cfe_std = self.gp_cfe_delta.predict(X, return_std=True)
            sea_mu  = sea_mu  + d_sea_mu
            cfe_mu  = cfe_mu  + d_cfe_mu
            sea_std = np.sqrt(sea_std**2 + d_sea_std**2)
            cfe_std = np.sqrt(cfe_std**2 + d_cfe_std**2)

        if return_std:
            return sea_mu, sea_std, cfe_mu, cfe_std
        return sea_mu, cfe_mu

    def objective(
        self,
        X: np.ndarray,
        k: float | None = None,
        y_scaler=None,
    ) -> np.ndarray:
        """
        Scalarized objective Obj = SEA - k * (1 - CFE) from GP predictions.

        If y_scaler is supplied, SEA and CFE are inverse-transformed to original
        units before computing the objective (matching analysis.py convention).
        k defaults to mean(SEA_pred) when not supplied.
        """
        sea_mu, cfe_mu = self.predict(X, return_std=False)

        if y_scaler is not None:
            tmp = y_scaler.inverse_transform(np.column_stack([sea_mu, cfe_mu]))
            sea_mu, cfe_mu = tmp[:, 0], tmp[:, 1]

        if k is None:
            k = float(sea_mu.mean())
        return sea_mu - k * (1.0 - cfe_mu)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """
        Persist all four GPs and metadata JSON to directory.

        If the correction GPs have not been fitted yet (is_hf_fitted=False),
        only the LF GPs and metadata are saved.
        """
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.gp_sea_lf, os.path.join(directory, "gp_sea_lf.pkl"))
        joblib.dump(self.gp_cfe_lf, os.path.join(directory, "gp_cfe_lf.pkl"))

        if self.is_hf_fitted:
            joblib.dump(self.gp_sea_delta, os.path.join(directory, "gp_sea_delta.pkl"))
            joblib.dump(self.gp_cfe_delta, os.path.join(directory, "gp_cfe_delta.pkl"))

        meta = {
            "is_lf_fitted":      self.is_lf_fitted,
            "is_hf_fitted":      self.is_hf_fitted,
            "n_restarts":        self.n_restarts,
            "alpha":             self.alpha,
            "lf_kernel_params":  self._lf_kernel_str,
            "delta_kernel_params": self._delta_kernel_str,
            "saved_at":          datetime.now().isoformat(timespec="seconds"),
        }
        with open(os.path.join(directory, "gp_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[gp] Saved to {directory}/")

    @classmethod
    def load(cls, directory: str) -> "MultiFidelityGP":
        """Reconstruct a MultiFidelityGP from a previously saved directory."""
        meta_path = os.path.join(directory, "gp_metadata.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"GP metadata not found: {meta_path}")

        with open(meta_path) as f:
            meta = json.load(f)

        obj = cls.__new__(cls)
        obj.n_restarts = meta["n_restarts"]
        obj.alpha      = meta["alpha"]
        obj._lf_kernel_str    = meta.get("lf_kernel_params",    {})
        obj._delta_kernel_str = meta.get("delta_kernel_params", {})

        obj.gp_sea_lf = joblib.load(os.path.join(directory, "gp_sea_lf.pkl"))
        obj.gp_cfe_lf = joblib.load(os.path.join(directory, "gp_cfe_lf.pkl"))
        obj.is_lf_fitted = meta["is_lf_fitted"]

        kw = dict(n_restarts_optimizer=obj.n_restarts, alpha=obj.alpha, normalize_y=False)
        if meta["is_hf_fitted"]:
            obj.gp_sea_delta = joblib.load(os.path.join(directory, "gp_sea_delta.pkl"))
            obj.gp_cfe_delta = joblib.load(os.path.join(directory, "gp_cfe_delta.pkl"))
        else:
            obj.gp_sea_delta = GaussianProcessRegressor(kernel=_make_kernel(0.5), **kw)
            obj.gp_cfe_delta = GaussianProcessRegressor(kernel=_make_kernel(0.5), **kw)
        obj.is_hf_fitted = meta["is_hf_fitted"]

        print(f"[gp] Loaded from {directory}/  (saved {meta.get('saved_at', 'unknown')})")
        return obj
