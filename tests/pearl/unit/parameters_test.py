"""
Test module for parameters.py
"""

from pathlib import Path

from pandas.testing import assert_frame_equal
from pytest import fixture

from pearl.parameters import Parameters


@fixture
def param_file_path():
    return Path("tests/pearl/assets/parameters.h5")


@fixture
def test_parameter_1(param_file_path):
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
        sa_variables=[
            "dm_prevalence",
            "dm_prevalence_prev",
            "dm_incidence",
            "pre_art_bmi",
            "post_art_bmi",
            "art_initiators",
        ],
    )


@fixture
def test_parameter_2(param_file_path):
    return Parameters(
        path=param_file_path,
        output_folder=None,
        replication=42,
        group_name="idu_white_female",
        new_dx="base",
        final_year=2015,
        mortality_model="by_sex_race_risk",
        mortality_threshold_flag=1,
        idu_threshold="2x",
        seed=121294,
        sa_variables=[
            "dm_prevalence",
            "dm_prevalence_prev",
            "dm_incidence",
            "pre_art_bmi",
            "post_art_bmi",
            "art_initiators",
        ],
    )


def test_parameter_random_init_values_are_equal(test_parameter_1, test_parameter_2):
    """
    The initiation random variables for sensitivity analysis should be the same
    """
    # drop the group and seed columns so that we can fairly assert the rest is equal
    param_1 = test_parameter_1.param_dataframe.drop(["group", "seed"], axis=1)
    param_2 = test_parameter_2.param_dataframe.drop(["group", "seed"], axis=1)

    assert_frame_equal(param_1, param_2)
