"""
test module for events.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pytest import fixture

from pearl.model import Parameters
from pearl.population.events import calculate_cd4_decrease, calculate_cd4_increase


@pytest.mark.xfail
def test_create_mortality_in_care_pop_matrix():
    raise NotImplementedError


@pytest.mark.xfail
def test_create_mortality_out_care_pop_matrix():
    raise NotImplementedError


@fixture
def test_cd4_increase_pop():
    population = {
        "intercept": [1, 1, 1, 1, 1],
        "age_cat": [1.0, 2.0, 3.0, 4.0, 5.0],
        "year": [2009, 2009, 2009, 2009, 2009],
        "last_h1yy": [2009, 2008, 2001, 2007, 2003],
        "last_init_sqrtcd4n": [
            7.785768284900243,
            13.921099280684002,
            6.670803203549613,
            3.962003815707643,
            11.188837871609948,
        ],
    }
    return pd.DataFrame(population)


@fixture
def param_file_path():
    return Path("tests/pearl/assets/parameters.h5")


@fixture
def test_parameters(param_file_path):
    return Parameters(
        path=param_file_path,
        output_folder=None,
        replication=42,
        group_name="msm_black_male",
        new_dx="base",
        final_year=2015,
        mortality_model="by_sex_race_risk",
        mortality_threshold_flag=1,
        idu_threshold="2x",
        seed=42,
    )


@fixture
def expected_cd4_increase():
    return np.array(
        [
            13.321834633018316,
            14.5111149282341,
            19.867717597510353,
            15.451144615518423,
            18.613086718713745,
        ]
    )


def test_calculate_cd4_increase(test_cd4_increase_pop, test_parameters, expected_cd4_increase):
    """
    It should return the expected numbers
    """
    cd4_increase = calculate_cd4_increase(test_cd4_increase_pop, test_parameters)
    assert np.allclose(cd4_increase, expected_cd4_increase)


@fixture
def test_cd4_decrease_pop():
    population = {
        "intercept": [1, 1, 1, 1, 1],
        "year": [2009, 2009, 2009, 2009, 2009],
        "ltfu_year": [2009, 2008, 2001, 2007, 2003],
        "sqrtcd4n_exit": [
            16.459867166915828,
            26.136887305474033,
            19.082228482285846,
            23.125814322560394,
            26.30349530000121,
        ],
    }
    return pd.DataFrame(population)


@fixture
def expected_cd4_decrease():
    return np.array(
        [
            11.286071129634552,
            19.81473466999056,
            11.645263381341982,
            16.58330381833863,
            18.207905726917033,
        ]
    )


def test_calculate_cd4_decrease(test_cd4_decrease_pop, test_parameters, expected_cd4_decrease):
    """
    It should return the expected numbers
    """
    cd4_decrease = calculate_cd4_decrease(test_cd4_decrease_pop, test_parameters)
    assert np.allclose(cd4_decrease, expected_cd4_decrease)
