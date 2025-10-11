# 3rd Party Libraries
import numpy as np
import scipy.integrate as scipy

# Internal Libraries
from enum import Enum

class OperationsInterface:

    class Mode(Enum):
        FvD = 0
        AvC = 1

    @staticmethod
    def cfe(X: list) -> float:
        if not X:
            return 0.0
        x_max = max(X)
        if x_max == 0:
            return 0.0
        return float(np.median(X) / x_max)

    @staticmethod
    def _prep_xy(df: list, x_col: str, y_col: str):
        d = df[[x_col, y_col]].dropna()
        if d.shape[0] < 2:
            return None, None
        x = d[x_col].to_numpy(dtype=float, copy=False)
        y = d[y_col].to_numpy(dtype=float, copy=False)
        order = np.argsort(x)
        return x[order], y[order]

    @staticmethod
    def _clip_to_xmax(x: np.ndarray, y: np.ndarray, x_max: float):
        if x_max <= x[0]:
            return None, None  # area is 0 up to x_max
        if x_max >= x[-1]:
            return x, y        # no clipping needed

        i = np.searchsorted(x, x_max, side="left")
        # exact match
        if x[i] == x_max:
            return x[:i+1], y[:i+1]
        # i could be 0 only if x_max < x[0], which we handled above
        x_clip = np.concatenate([x[:i], [x_max]])
        y_clip = np.concatenate([y[:i], [np.interp(x_max, x[i-1:i+1], y[i-1:i+1])]])
        return x_clip, y_clip

    @staticmethod
    def auc_trapz(df: list, x_col: str = "Displacement", y_col: str = "Force", x_max=None) -> float:
        x, y = OperationsInterface._prep_xy(df, x_col, y_col)
        if x is None:
            return 0.0
        if x_max is not None:
            x, y = OperationsInterface._clip_to_xmax(x, y, float(x_max))
            if x is None:
                return 0.0
        return float(scipy.trapezoid(y, x))

    @staticmethod
    def auc_simps(df: list, x_col: str = "Displacement", y_col: str = "Force", x_max=None) -> float:
        x, y = OperationsInterface._prep_xy(df, x_col, y_col)
        if x is None:
            return 0.0
        if x_max is not None:
            x, y = OperationsInterface._clip_to_xmax(x, y, float(x_max))
            if x is None:
                return 0.0
        # Simpson needs >= 2 points; SciPy handles nonuniform spacing.
        return float(scipy.simpson(y, x))