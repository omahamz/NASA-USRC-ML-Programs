import numpy as np
from scipy.integrate import simpson

class OperationsInterface:

    @staticmethod
    def find_peak(X) -> float:
        x_max = 0
        for x in X:
            if x > x_max:
                x_max = x
        return x_max

    @staticmethod
    def auc_trapz(x, y, x_max=None) -> float:
        x, y = np.asarray(x), np.asarray(y)
        order = np.argsort(x)
        x, y = x[order], y[order]
        if x_max is not None:
            # clip to x_max (linear interpolate last point)
            if x_max < x[-1]:
                i = np.searchsorted(x, x_max)
                x_clip = np.concatenate([x[:i], [x_max]])
                y_clip = np.concatenate([y[:i],[np.interp(x_max, x[i-1:i+1], y[i-1:i+1])]])
                x, y = x_clip, y_clip
        return np.trapezoid(y, x)

    @staticmethod
    def auc_simps(df, x_col, y_col, x_max=None) -> float:
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
        return simpson(y, x)