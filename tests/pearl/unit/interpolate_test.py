import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal
from pytest import fixture

from pearl.interpolate import (
    restricted_cubic_spline_var,
    restricted_quadratic_spline_var,
)


@fixture
def cubic_test_series() -> pd.Series:
    return pd.Series([18, 35, 52, 70])


@fixture
def cubic_knots() -> pd.Series:
    return pd.Series([24.0, 37.0, 45.0, 59.0])


@fixture
def expected_first_cubic_knot() -> pd.Series:
    return pd.Series([0.0, 1.0865306122448988, 17.220000000000006, 49.200000000000045])


@fixture
def expected_second_cubic_knot() -> pd.Series:
    return pd.Series([0.0, 0.0, 2.3151020408163285, 9.913469387755109])


def test_restricted_cubic_spline_var(
    cubic_test_series,
    cubic_knots,
    expected_first_cubic_knot,
    expected_second_cubic_knot,
):
    """
    It should return the expected value.
    """
    first_knot = restricted_cubic_spline_var(cubic_test_series, cubic_knots, 1)
    second_knot = restricted_cubic_spline_var(cubic_test_series, cubic_knots, 2)

    assert_series_equal(first_knot, expected_first_cubic_knot)
    assert_series_equal(second_knot, expected_second_cubic_knot)


def test_restricted_cubic_spline_var_bad_i(cubic_test_series, cubic_knots):
    """
    It should raise a value error when i != 1 or 2.
    """
    with pytest.raises(ValueError):
        restricted_cubic_spline_var(cubic_test_series, cubic_knots, 3)
    with pytest.raises(ValueError):
        restricted_cubic_spline_var(cubic_test_series, cubic_knots, 0)


@fixture
def quadratic_test_series():
    return pd.Series([18, 35, 52, 70])


@fixture
def quadratic_knots():
    return np.array([1.0, 4, 7.0, 13.0])


@fixture
def expected_first_quadratic_knot():
    return pd.Series([0.0, 3.4571428571428573, 22.4, 57.0])


@fixture
def expected_second_quadratic_knot():
    return pd.Series([0.0, 0.0, 6.428571428571429, 27.65714285714286])


@fixture
def expected_third_quadratic_knot():
    return pd.Series([0.0, 0.0, 1.4, 14.4])


def test_restricted_quadratic_spline_var(
    quadratic_test_series,
    cubic_knots,
    expected_first_quadratic_knot,
    expected_second_quadratic_knot,
    expected_third_quadratic_knot,
):
    """
    It should return the expected value.
    """
    first_knot = restricted_quadratic_spline_var(quadratic_test_series, cubic_knots, 1)
    second_knot = restricted_quadratic_spline_var(quadratic_test_series, cubic_knots, 2)
    third_knot = restricted_quadratic_spline_var(quadratic_test_series, cubic_knots, 3)

    assert_series_equal(first_knot, expected_first_quadratic_knot)
    assert_series_equal(second_knot, expected_second_quadratic_knot)
    assert_series_equal(third_knot, expected_third_quadratic_knot)


def test_restricted_quadratic_spline_var_bad_i(quadratic_test_series, cubic_knots):
    """
    It should raise a value error when i != 1 or 2.
    """
    with pytest.raises(ValueError):
        restricted_quadratic_spline_var(quadratic_test_series, cubic_knots, 4)
    with pytest.raises(ValueError):
        restricted_quadratic_spline_var(cubic_test_series, cubic_knots, 0)
