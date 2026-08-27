import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
import optuna

def optimize_parameter(fd_df: pd.DataFrame, material_name: str, param_value: float, auc: float, cfe: float,
                       param_name: str,
                       degree: int = 3,
                       n_trials: int = 200) -> float:
    """
    Given an F-D curve DataFrame, one parameter value, and the computed AUC and CFE,
    builds a simple surrogate model and returns the best parameter that maximizes both.

    Parameters
    ----------
    fd_df : pd.DataFrame
        The Force-Displacement data. Columns: ["Displacement", "Force"]
    param_value : float
        The design parameter corresponding to this curve (e.g., twist angle)
    auc : float
        Area under the curve (energy absorbed)
    cfe : float
        Crushing Force Efficiency (0 < CFE < 1)
    param_name : str, optional
        Name of the design parameter
    degree : int, optional
        Polynomial degree for regression model
    n_trials : int, optional
        Number of Optuna trials to search best parameter

    Returns
    -------
    float
        The best estimated parameter that maximizes predicted AUC and CFE
    """

    # Assemble minimal dataset: here we just reuse one point, but in practice you'd have many (param_value_i, AUC_i, CFE_i)
    df = pd.DataFrame({
        param_name: [param_value],
        "AUC": [auc],
        "CFE": [cfe]
    })

    # If you have multiple F–D curves/parameter samples, append them here.
    # For now, we’ll simulate a small synthetic variation around the single value.
    # (This just allows the model to interpolate instead of fitting 1 point.)
    spread = np.linspace(-5, 5, 7)
    df_extra = pd.DataFrame({
        param_name: param_value + spread,
        "AUC": auc + np.random.normal(0, 5, len(spread)),
        "CFE": np.clip(cfe + np.random.normal(0, 0.05, len(spread)), 0, 1)
    })
    df = pd.concat([df, df_extra], ignore_index=True)

    X = df[[param_name]].values
    Y = df[["AUC", "CFE"]].values

    # Train simple polynomial regression (1D)
    surrogate = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        MultiOutputRegressor(Ridge(alpha=1.0))
    )
    surrogate.fit(X, Y)

    # Normalize targets for balanced optimization
    auc_mean, auc_std = df["AUC"].mean(), df["AUC"].std() + 1e-8
    cfe_mean, cfe_std = df["CFE"].mean(), df["CFE"].std() + 1e-8
    lo, hi = float(df[param_name].min()), float(df[param_name].max())

    def objective(trial):
        p = trial.suggest_float(param_name, lo, hi)
        auc_pred, cfe_pred = surrogate.predict(np.array([[p]]))[0]
        if not (0.0 < cfe_pred < 1.0):
            return -1e9
        # Scalarization of two objectives (equal weight)
        score = (auc_pred - auc_mean)/auc_std + (cfe_pred - cfe_mean)/cfe_std
        return score

    # Run Optuna search
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_p = study.best_params[param_name]
    best_auc, best_cfe = surrogate.predict(np.array([[best_p]]))[0]

    print(f"\nOptimization Results for {material_name}:")
    print(f"\nBest {param_name}: {best_p:.3f}")
    print(f"Predicted AUC: {best_auc:.3f}")
    print(f"Predicted CFE: {best_cfe:.4f}")

    return best_p

