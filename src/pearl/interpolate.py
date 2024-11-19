"""
Module for interpolations.
"""

from typing import Any

from numba import njit
import numpy as np
from numpy.typing import NDArray


@njit  # type: ignore[misc]
def restricted_cubic_spline_var(x: NDArray[Any], knots: NDArray[Any], i: int) -> Any:
    """
    Return a pandas Series of the i'th restricted cubic spline variable for numpy array x with
    knots.
    Adapted from: https:github.com/harrelfe/Hmisc/blob/master/R/rcspline.eval.s

    Parameters
    ----------
    x : np.array
        Array to apply spline interpolation to.
    knots : np.array
        Array containing the values of the knots.
    i : int
        The knot for calculation.

    Returns
    -------
    np.array
        The interpolated array.

    Raises
    ------
    ValueError
        Raises an error if i is not 1 or 2 as this is undefined for cubic splines.
    """
    if i < 1 or i > 2:
        raise ValueError("i must equal 1 or 2")
    kd = (knots[3] - knots[0]) ** (2 / 3)
    y = (
        np.maximum(0, (x - knots[i - 1]) / kd) ** 3
        - (np.maximum(0, (x - knots[2]) / kd) ** 3)
        * (knots[3] - knots[i - 1])
        / (knots[3] - knots[2])
        + (np.maximum(0, (x - knots[3]) / kd) ** 3)
        * (knots[2] - knots[i - 1])
        / (knots[3] - knots[2])
    )
    return y


@njit  # type: ignore[misc]
def restricted_quadratic_spline_var(x: NDArray[Any], knots: NDArray[Any], i: int) -> Any:
    """
    Return a pandas Series of the i'th restricted cubic spline variable for numpy array x with
    knots.
    Adapted from: https:github.com/harrelfe/Hmisc/blob/master/R/rcspline.eval.s

    Parameters
    ----------
    x : np.array
        Array to apply spline interpolation to.
    knots : np.array
        Array containing the values of the knots.
    i : int
        The knot for calculation.

    Returns
    -------
    np.array
        The interpolated array.

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
