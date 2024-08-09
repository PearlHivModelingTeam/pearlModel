import numpy as np
import pandas as pd


def restricted_cubic_spline_var(x: pd.Series, t: pd.Series, i: int) -> pd.Series:
    """Return a pandas Series of the i'th restricted cubic spline variable for numpy array x with knots t.
    Adapted from: https:github.com/harrelfe/Hmisc/blob/master/R/rcspline.eval.s
    """
    if i < 1 or i > 2:
        raise ValueError("i must equal 1 or 2")
    kd = (t.iloc[3] - t.iloc[0]) ** (2 / 3)
    y = (
        np.maximum(0, (x - t.iloc[i - 1]) / kd) ** 3
        - (np.maximum(0, (x - t.iloc[2]) / kd) ** 3)
        * (t.iloc[3] - t.iloc[i - 1])
        / (t.iloc[3] - t.iloc[2])
        + (np.maximum(0, (x - t.iloc[3]) / kd) ** 3)
        * (t.iloc[2] - t.iloc[i - 1])
        / (t.iloc[3] - t.iloc[2])
    )
    return y


def restricted_quadratic_spline_var(x: pd.Series, t: np.array, i: int) -> pd.Series:
    """Return a pandas Series of the i'th restricted quadratic spline variable for numpy array x with knots t."""
    if i < 1 or i > 3:
        raise ValueError("i must be 1, 2, or 3")
    y = (np.maximum(0, x - t[i - 1]) ** 2 - np.maximum(0, x - t[3]) ** 2) / (t[3] - t[0])
    return y
