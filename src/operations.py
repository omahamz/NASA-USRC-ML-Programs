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
    def max(X: list) -> float:
        return max(X) if len(X) > 0 else 0.0
    
    @staticmethod
    def cfe(X: list) -> float:
        if not X:
            return 0.0
        x_max = OperationsInterface.max(X)
        if x_max == 0:
            return 0.0
        return np.median(X) / x_max

    @staticmethod
    def auc_trapz(df: list, x_col: str ="Displacement", y_col: str="Force", x_max=None) -> float:
        x = df[x_col].values
        y = df[y_col].values
        order = np.argsort(x)
        x, y = x[order], y[order]
        if x_max is not None:
            # clip to x_max (linear interpolate last point)
            if x_max < x[-1]:
                i = np.searchsorted(x, x_max)
                x_clip = np.concatenate([x[:i], [x_max]])
                y_clip = np.concatenate([y[:i], [np.interp(x_max, x[i-1:i+1], y[i-1:i+1])]])
                x, y = x_clip, y_clip
        return scipy.trapezoid(y, x)

    @staticmethod
    def auc_simps(df: list, x_col: str ="Displacement", y_col: str="Force", x_max=None) -> float:
        x = df[x_col].values
        y = df[y_col].values
        order = np.argsort(x)
        x, y = x[order], y[order]
        if x_max is not None:
            if x_max < x[-1]:
                i = np.searchsorted(x, x_max)
                x_clip = np.concatenate([x[:i], [x_max]])
                y_clip = np.concatenate([y[:i], [np.interp(x_max, x[i-1:i+1], y[i-1:i+1])]])
                x, y = x_clip, y_clip
        return scipy.simpsons(y, x)