"""
Test module for generation.py
"""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
from numpy.random import RandomState
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from pytest import fixture

from pearl.model import Parameters
from pearl.population.generation import (
    apply_bmi_intervention,
    calculate_post_art_bmi,
    calculate_pre_art_bmi,
    simulate_ages,
    simulate_new_dx,
)


@fixture
def random_state():
    return RandomState(seed=42)


@fixture
def test_simulate_ages_coeffs():
    coeffs = {
        "term": ["lambda1", "mu1", "mu2", "sigma1", "sigma2"],
        "estimate": [
            0.1428275968730224,
            25.90157466563675,
            44.15102821208629,
            3.36447304669819,
            8.995675326063536,
        ],
        "conf_high": [
            0.1696263422876576,
            26.46764195909201,
            44.7052307739297,
            3.8295935710199736,
            9.373568241855462,
        ],
        "conf_low": [
            0.1160288514583873,
            25.335507372181493,
            43.59682565024288,
            2.8993525223764065,
            8.617782410271609,
        ],
    }

    return pd.DataFrame(coeffs).set_index("term")


@fixture
def expected_ages():
    return np.array(
        [
            59.01796147725942,
            49.731344366913646,
            46.415830971908484,
            35.114459279477444,
            35.1135600789522,
        ]
    )


def test_simulate_ages(test_simulate_ages_coeffs, random_state, expected_ages):
    """
    It should return the same numbers when given a seeded random object.
    """
    result = simulate_ages(test_simulate_ages_coeffs, 5, random_state=random_state)

    assert np.allclose(result, expected_ages)


@fixture
def test_new_dx():
    new_dx = {
        "year": [2006, 2007, 2008, 2009, 2010, 2011, 2012],
        "lower": [
            8759.81022859518,
            8814.498826379442,
            8868.278247690176,
            8920.780812339448,
            8971.425854986937,
            9019.259320192376,
            9062.657221734276,
        ],
        "upper": [
            10095.003068659726,
            10019.678515177244,
            9945.263138168297,
            9872.124617820804,
            9800.843619475094,
            9732.374198571437,
            9668.34034133132,
        ],
    }

    return pd.DataFrame(new_dx).set_index("year")


@fixture
def test_linkage_to_care():
    linkage_to_care = {
        "year": [2006, 2007, 2008, 2009, 2010, 2011, 2012],
        "link_prob": [
            0.7581369863014515,
            0.7665993150685715,
            0.775061643835695,
            0.7835239726028185,
            0.791986301369942,
            0.8004486301370655,
            0.8089109589041854,
        ],
        "art_prob": [0.7, 0.7, 0.7, 0.7, 0.7, 0.85, 0.97],
    }

    return pd.DataFrame(linkage_to_care).set_index("year")


@fixture
def test_parameters(test_new_dx, test_linkage_to_care):
    parameters = Mock()
    parameters.new_dx = test_new_dx
    parameters.linkage_to_care = test_linkage_to_care
    return parameters


@fixture
def expected_n_initial_nonusers():
    return 9097


@fixture
def expected_new_agents():
    new_agents = {
        "year": [2010, 2011, 2012],
        "art_initiators": [5729, 7001, 8038],
        "art_delayed": [2455, 1235, 248],
    }

    return pd.DataFrame(new_agents).set_index("year")


def test_simulate_new_dx(
    test_parameters,
    random_state,
    expected_n_initial_nonusers,
    expected_new_agents,
):
    """
    It should return the same numbers when passed a seeded RandomState
    """
    initial_nonusers, new_agents = simulate_new_dx(test_parameters, random_state=random_state)

    assert expected_n_initial_nonusers == initial_nonusers
    assert_frame_equal(new_agents, expected_new_agents)


@fixture
def test_bmi_population():
    population = {
        "h1yy": [2010, 2011, 2010, 2012, 2010],
        "dm": [0, 0, 1, 0, 1],
        "pre_art_bmi": [28.4, 23.8, 27.3, 17.0, 32.2],
        "post_art_bmi": [30.6, 24.7, 31.0, 19.2, 35.6],
    }
    return pd.DataFrame(population)


@fixture
def test_bmi_intervention_parameters():
    parameters = Mock()
    parameters.bmi_intervention_start_year = 2009
    parameters.bmi_intervention_end_year = 2010
    parameters.bmi_intervention_coverage = 1.0
    parameters.bmi_intervention_effectiveness = 1.0
    return parameters


@fixture
def expected_bmi_intervention_output_scenario_one():
    bmi_output = [29.9, 24.7, 31.0, 19.2, 35.6]
    return pd.Series(bmi_output).rename("post_art_bmi")


def test_apply_bmi_intervention_scenario_one(
    test_bmi_population,
    test_bmi_intervention_parameters,
    random_state,
    expected_bmi_intervention_output_scenario_one,
):
    """
    It should return the same population when passed a seeded random_state.
    """
    test_bmi_intervention_parameters.bmi_intervention_scenario = 1
    bmi_output_scenario_one = apply_bmi_intervention(
        test_bmi_population, test_bmi_intervention_parameters, random_state=random_state
    )

    assert_series_equal(
        bmi_output_scenario_one["post_art_bmi"],
        expected_bmi_intervention_output_scenario_one,
    )


@fixture
def expected_bmi_intervention_output_scenario_two():
    bmi_output = [29.82, 24.70, 31.00, 19.20, 35.60]
    return pd.Series(bmi_output).rename("post_art_bmi")


def test_apply_bmi_intervention_scenario_two(
    test_bmi_population,
    test_bmi_intervention_parameters,
    random_state,
    expected_bmi_intervention_output_scenario_two,
):
    """
    It should return the same population when passed a seeded random_state.
    """
    test_bmi_intervention_parameters.bmi_intervention_scenario = 2
    bmi_output_scenario_two = apply_bmi_intervention(
        test_bmi_population, test_bmi_intervention_parameters, random_state=random_state
    )

    assert_series_equal(
        bmi_output_scenario_two["post_art_bmi"],
        expected_bmi_intervention_output_scenario_two,
    )


@fixture
def expected_bmi_intervention_output_scenario_three():
    bmi_output = [28.4, 24.7, 31.0, 19.2, 35.6]
    return pd.Series(bmi_output).rename("post_art_bmi")


def test_apply_bmi_intervention_scenario_three(
    test_bmi_population,
    test_bmi_intervention_parameters,
    random_state,
    expected_bmi_intervention_output_scenario_three,
):
    """
    It should return the same population when passed a seeded random_state.
    """
    test_bmi_intervention_parameters.bmi_intervention_scenario = 3
    bmi_output_scenario_three = apply_bmi_intervention(
        test_bmi_population, test_bmi_intervention_parameters, random_state=random_state
    )

    assert_series_equal(
        bmi_output_scenario_three["post_art_bmi"],
        expected_bmi_intervention_output_scenario_three,
    )


@fixture
def param_file_path():
    return Path("tests/pearl/assets/parameters.h5")


@fixture
def test_art_bmi_parameters(param_file_path):
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
def test_art_bmi_population():
    population = {
        "year": pd.Series([2002, 2004, 2006, 2011, 2013]),
        "init_age": pd.Series([19.0, 36.0, 50.0, 65.0, 85.0]),
        "pre_art_bmi": pd.Series([28.4, 23.8, 27.3, 17.0, 32.2]),
        "init_sqrtcd4n": pd.Series(
            [
                11.038756528100167,
                23.32942023071808,
                23.73959591750536,
                20.67559217085155,
                18.336389525364968,
            ]
        ),
        "last_init_sqrtcd4n": pd.Series(
            [
                7.785768284900243,
                13.921099280684002,
                6.670803203549613,
                3.962003815707643,
                11.188837871609948,
            ]
        ),
        "h1yy": pd.Series([2000, 2002, 2004, 2009, 2011]),
        "last_h1yy": pd.Series([2009, 2008, 2005, 2010, 2012]),
        "intercept": pd.Series([1.0, 1.0, 1.0, 1.0, 1.0]),
    }
    return pd.DataFrame(population)


@fixture
def expected_post_art_bmi_no_intervention():
    return np.array(
        [
            29.802222136230082,
            22.51843690121106,
            26.695163432294805,
            14.856891053958686,
            28.78478441725878,
        ]
    )


def test_calculate_post_art_bmi_no_intervention(
    test_art_bmi_population,
    test_art_bmi_parameters,
    random_state,
    expected_post_art_bmi_no_intervention,
):
    """
    It should return the same output, given the same input and a seeded RandomState with no
    intervention.
    """
    post_art_bmi = calculate_post_art_bmi(
        test_art_bmi_population,
        test_art_bmi_parameters,
        random_state=random_state,
        intervention=False,
    )
    np.allclose(post_art_bmi, expected_post_art_bmi_no_intervention)


@fixture
def expected_post_art_bmi_with_intervention():
    return np.array(
        [
            29.130797492808735,
            22.51837077835867,
            26.56599902564644,
            14.856891053955207,
            27.39921848297732,
        ]
    )


def test_calculate_post_art_bmi_with_intervention(
    test_art_bmi_population,
    test_art_bmi_parameters,
    random_state,
    expected_post_art_bmi_with_intervention,
):
    """
    It should return the same output, given the same input and a seeded RandomState with no
    intervention.
    """
    post_art_bmi = calculate_post_art_bmi(
        test_art_bmi_population,
        test_art_bmi_parameters,
        random_state=random_state,
        intervention=True,
    )
    np.allclose(post_art_bmi, expected_post_art_bmi_with_intervention)


@fixture
def expected_pre_art_bmi():
    return np.array(
        [
            28.91516451518173,
            27.361387946712178,
            26.029469342376085,
            20.654728481445687,
            19.85041131111226,
        ]
    )


def test_calculate_pre_art_bmi(
    test_art_bmi_population,
    test_art_bmi_parameters,
    random_state,
    expected_post_art_bmi_with_intervention,
):
    """
    It should return the same output, given the same input and a seeded RandomState with no
    intervention.
    """
    pre_art_bmi = calculate_pre_art_bmi(
        test_art_bmi_population, test_art_bmi_parameters, random_state=random_state
    )
    np.allclose(pre_art_bmi, expected_post_art_bmi_with_intervention)
