"""
Tests for model.py
"""

from pathlib import Path

from pytest import fixture

from pearl.model import Pearl
from pearl.parameters import Parameters


@fixture
def param_file_path():
    return Path("tests/pearl/assets/parameters.h5")


@fixture
def test_parameters(param_file_path):
    return Parameters(
        path=param_file_path,
        output_folder=None,
        group_name="msm_black_male",
        new_dx="base",
        final_year=2015,
        mortality_model="by_sex_race_risk",
        mortality_threshold_flag=1,
        idu_threshold="2x",
        seed=42,
    )


@fixture
def test_pearl(test_parameters):
    return Pearl(test_parameters, "msm_black_male", 42)


def test_init_year(test_pearl):
    assert test_pearl.year == 2010
