"""
Test module for multimorbidity.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from pytest import fixture

from pearl.multimorbidity import create_comorbidity_pop_matrix
from pearl.parameters import Parameters


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
def test_population():
    population = {
        "year": pd.Series([2002, 2004, 2006, 2011, 2013]),
        "h1yy": pd.Series([2000, 2002, 2004, 2009, 2011]),
        "status": pd.Series([0, 1, 2, 3, 4]),
        "pre_art_bmi": pd.Series([28.4, 23.8, 27.3, 17.0, 32.2]),
        "post_art_bmi": pd.Series([30.6, 24.7, 31.0, 19.2, 35.6]),
        "age": [18, 19, 20, 21, 22],
        "init_sqrtcd4n": pd.Series(
            [
                11.038756528100167,
                23.32942023071808,
                23.73959591750536,
                20.67559217085155,
                18.336389525364968,
            ]
        ),
        "anx": [True, True, False, True, False],
        "dpr": [True, True, False, True, False],
        "time_since_art": [5, 6, 7, 8, 9],
        "hcv": [False, False, True, False, False],
        "intercept": pd.Series([1.0, 1.0, 1.0, 1.0, 1.0]),
        "smoking": [True, True, True, False, False],
        "last_init_sqrtcd4n": pd.Series(
            [
                7.785768284900243,
                13.921099280684002,
                6.670803203549613,
                3.962003815707643,
                11.188837871609948,
            ]
        ),
        "last_h1yy": pd.Series([2009, 2008, 2005, 2010, 2012]),
        "dm": [True, True, False, True, False],
        "ht": [True, True, False, True, False],
        "lipid": [True, True, False, True, False],
        "ckd": [True, True, False, True, False],
    }
    population["delta_bmi"] = population["pre_art_bmi"] - population["post_art_bmi"]
    return pd.DataFrame(population)


@fixture
def expected_anx_result():
    return np.array([18.0, 11.038756528100167, 1.0, 2.0, 0.0, 1.0, 1.0, 2002.0])


def test_create_comorbidity_pop_matrix_anx(test_population, test_parameters, expected_anx_result):
    anx_result = create_comorbidity_pop_matrix(test_population, "anx", test_parameters)
    assert np.allclose(anx_result[0], expected_anx_result)


@fixture
def expected_dpr_result():
    return np.array([18.0, 1.0, 11.038756528100167, 2.0, 0.0, 1.0, 1.0, 2002.0])


def test_create_comorbidity_pop_matrix_dpr(test_parameters, test_population, expected_dpr_result):
    dpr_result = create_comorbidity_pop_matrix(test_population, "dpr", test_parameters)
    assert np.allclose(dpr_result[0], expected_dpr_result)


@fixture
def expected_ckd_result():
    return np.array(
        [
            18.0,
            1.0,
            0.0,
            0.0,
            -2.200000000000003,
            30.6,
            4.985694456247702,
            1.0459082249794633,
            11.038756528100167,
            1.0,
            1.0,
            2.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            2002.0,
        ]
    )


def test_create_comorbidity_pop_matrix_ckd(test_parameters, test_population, expected_ckd_result):
    ckd_result = create_comorbidity_pop_matrix(test_population, "ckd", test_parameters)
    assert np.allclose(ckd_result[0], expected_ckd_result)


@fixture
def expected_lipid_result():
    return np.array(
        [
            18.0,
            1.0,
            0.0,
            0.0,
            -2.200000000000003,
            30.6,
            4.869224773242633,
            1.0857984339569167,
            11.038756528100167,
            1.0,
            1.0,
            1.0,
            2.0,
            0.0,
            1.0,
            1.0,
            1.0,
            2002.0,
        ]
    )


def test_create_comorbidity_pop_matrix_lipid(
    test_parameters, test_population, expected_lipid_result
):
    lipid_result = create_comorbidity_pop_matrix(test_population, "lipid", test_parameters)
    assert np.allclose(lipid_result[0], expected_lipid_result)


@fixture
def expected_dm_result():
    return np.array(
        [
            18.0,
            1.0,
            0.0,
            0.0,
            -2.200000000000003,
            30.6,
            5.309525247713419,
            1.1570813643292703,
            11.038756528100167,
            1.0,
            1.0,
            2.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            2002.0,
        ]
    )


def test_create_comorbidity_pop_matrix_dm(test_parameters, test_population, expected_dm_result):
    dm_result = create_comorbidity_pop_matrix(test_population, "dm", test_parameters)
    assert np.allclose(dm_result[0], expected_dm_result)


@fixture
def expected_ht_result():
    return np.array(
        [
            18.0,
            1.0,
            0.0,
            0.0,
            -2.200000000000003,
            30.6,
            6.216349417780263,
            1.3954920938426507,
            11.038756528100167,
            1.0,
            1.0,
            1.0,
            2.0,
            0.0,
            1.0,
            1.0,
            1.0,
            2002.0,
        ]
    )


def test_create_comorbidity_pop_matrix_ht(test_parameters, test_population, expected_ht_result):
    ht_result = create_comorbidity_pop_matrix(test_population, "ht", test_parameters)
    assert np.allclose(ht_result[0], expected_ht_result)


@fixture
def expected_other_result():
    """
    malig, esld, mi results
    """
    return np.array(
        [
            18.0,
            1.0,
            0.0,
            0.0,
            -2.200000000000003,
            30.6,
            5.045609471134623,
            1.0614023818440457,
            11.038756528100167,
            1.0,
            1.0,
            1.0,
            2.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            2002.0,
        ]
    )


def test_create_comorbidity_pop_matrix_other(
    test_parameters, test_population, expected_other_result
):
    other_result = create_comorbidity_pop_matrix(test_population, "malig", test_parameters)
    assert np.allclose(other_result[0], expected_other_result)
