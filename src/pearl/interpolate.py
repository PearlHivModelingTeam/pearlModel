"""
Module for interpolations.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd


def restricted_cubic_spline_var(x: pd.Series, knots: pd.Series, i: int) -> pd.Series:
    """
    Return a pandas Series of the i'th restricted cubic spline variable for numpy array x with
    knots.
    Adapted from: https:github.com/harrelfe/Hmisc/blob/master/R/rcspline.eval.s

    Parameters
    ----------
    x : pd.Series
        Pandas Series to apply spline interpolation to.
    knots : pd.Series
        Pandas Series containing the values of the knots.
    i : int
        The knot for calculation.

    Returns
    -------
    pd.Series
        The interpolated series.

    Raises
    ------
    ValueError
        Raises an error if i is not 1 or 2 as this is undefined for cubic splines.
    """
    if i < 1 or i > 2:
        raise ValueError("i must equal 1 or 2")
    kd = (knots.iloc[3] - knots.iloc[0]) ** (2 / 3)
    y = (
        np.maximum(0, (x - knots.iloc[i - 1]) / kd) ** 3
        - (np.maximum(0, (x - knots.iloc[2]) / kd) ** 3)
        * (knots.iloc[3] - knots.iloc[i - 1])
        / (knots.iloc[3] - knots.iloc[2])
        + (np.maximum(0, (x - knots.iloc[3]) / kd) ** 3)
        * (knots.iloc[2] - knots.iloc[i - 1])
        / (knots.iloc[3] - knots.iloc[2])
    )
    return y


def restricted_quadratic_spline_var(x: pd.Series, knots: NDArray[Any], i: int) -> pd.Series:
    """
    Return a pandas Series of the i'th restricted cubic spline variable for numpy array x with
    knots.
    Adapted from: https:github.com/harrelfe/Hmisc/blob/master/R/rcspline.eval.s

    Parameters
    ----------
    x : pd.Series
        Pandas Series to apply spline interpolation to.
    knots : pd.Series
        Pandas Series containing the values of the knots.
    i : int
        The knot for calculation.

    Returns
    -------
    pd.Series
        The interpolated series.

    Raises
    ------
    ValueError
        Raises an error if i is not 1, 2, or 3 as this is undefined for quadratic splines.
    """
    if i < 1 or i > 3:
        raise ValueError("i must be 1, 2, or 3")
    y = (np.maximum(0, x - knots[i - 1]) ** 2 - np.maximum(0, x - knots[3]) ** 2) / (
        knots[3] - knots[0]
    )
    return y
