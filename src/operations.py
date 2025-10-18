# 3rd Party Libraries
import numpy as np
import scipy.integrate
import pandas as pd

# Internal Libraries
from enum import Enum

class OperationsInterface:

    class Mode(Enum):
        FvD = 0
        AvC = 1

    @staticmethod
    def cfe_median(df) -> float:
        if df.shape[0] < 2:
            return 0.0
        
        # Extract columns and handle finite values
        x = df.iloc[:, 0].to_numpy(dtype=float)
        y = df.iloc[:, 1].to_numpy(dtype=float)

        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if x.size < 2:
            return 0.0

        # Sort by x
        order = np.argsort(x)
        x, y = x[order], y[order]

        # Voronoi-style dx weights per point
        w = np.empty_like(x, dtype=float)
        if x.size == 2:
            w[:] = (x[1] - x[0]) / 2.0
        else:
            w[0] = (x[1] - x[0]) / 2.0
            w[1:-1] = (x[2:] - x[:-2]) / 2.0
            w[-1] = (x[-1] - x[-2]) / 2.0
        
        w = np.clip(w, 0.0, None)
        if w.sum() == 0:
            return 0.0

        # Weighted median of y
        idx = np.argsort(y)
        y_s, w_s = y[idx], w[idx]
        cdf = np.cumsum(w_s) / w_s.sum()
        y_wmed = y_s[np.searchsorted(cdf, 0.5)]

        y_max = np.max(y)
        if y_max == 0:
            return 0.0
        
        return y_wmed / y_max
    
    # OLD VERSION, is instead weighted mean
    @staticmethod
    def cfe_mean(df) -> float:
        # Expect dataframe with two columns: X and Y
        if df.shape[0] < 2:
            return 0.0
        
        x = df.iloc[:, 0].to_numpy(dtype=float)
        y = df.iloc[:, 1].to_numpy(dtype=float)
        
        # Sort in case x is unordered
        order = np.argsort(x)
        x, y = x[order], y[order]

        # Compute weighted mean using trapezoidal integration
        # (y[i] + y[i+1]) / 2 * (x[i+1] - x[i])
        auc = scipy.integrate.trapezoid(y, x)
        mean_y = auc / (x[-1] - x[0])

        y_max = np.max(y)
        if y_max == 0:
            return 0.0
        return mean_y / y_max

    @staticmethod
    def auc_trapz(df, x_col: str, y_col: str) -> float:
        # Input validation
        if df.empty or x_col not in df.columns or y_col not in df.columns:
            return 0.0
        
        x = df[x_col].to_numpy()
        y = df[y_col].to_numpy()
        
        # Handle finite values only
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        
        # Need at least 2 points for trapezoid integration
        if len(x) < 2:
            return 0.0
        
        # Sort by x-values (important for proper integration)
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        
        # Calculate area using trapezoid rule
        try:
            area = float(scipy.integrate.trapezoid(y_sorted, x_sorted))
            return area
        except (ValueError, TypeError):
            return 0.0
        
    # OLD VERSION, does not handle unsorted or NaN values
    # def auc_trapz(df, x_col: str, y_col: str) -> float:
    #     x = df[x_col]
    #     y = df[y_col]
    #     return float(scipy.trapezoid(y, x))