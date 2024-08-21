from pathlib import Path
import shutil

import dask
from dask import delayed
import pandas as pd
from pandas.testing import assert_frame_equal
from pytest import fixture

from pearl.model import Parameters, Pearl


@fixture
def config():
    config = {
        "new_dx": "base",
        "final_year": 2012,
        "mortality_model": "by_sex_race_risk",
        "mortality_threshold_flag": 1,
        "idu_threshold": "2x",
        "verbose": 0,
        "bmi_intervention": 1,
        "bmi_intervention_scenario": 1,
        "bmi_intervention_start_year": 2010,
        "bmi_intervention_end_year": 2011,
        "bmi_intervention_coverage": 1.0,
        "bmi_intervention_effectiveness": 1.0,
    }
    return config


@fixture
def param_file_path():
    return Path("tests/pearl/assets/parameters.h5")


@fixture
def output_folder():
    out_dir = Path("tests/pearl/end_to_end/output")
    if out_dir.is_dir():
        shutil.rmtree(out_dir)
    Path.mkdir(out_dir)
    return out_dir


@fixture
def parameter(param_file_path, output_folder, config):
    return Parameters(
        path=param_file_path,
        rerun_folder=None,
        output_folder=output_folder,
        group_name="msm_black_male",
        new_dx=config["new_dx"],
        final_year=config["final_year"],
        mortality_model=config["mortality_model"],
        mortality_threshold_flag=config["mortality_threshold_flag"],
        idu_threshold=config["idu_threshold"],
        verbose=config["verbose"],
        bmi_intervention=config["bmi_intervention"],
        bmi_intervention_scenario=config["bmi_intervention_scenario"],
        bmi_intervention_start_year=config["bmi_intervention_start_year"],
        bmi_intervention_end_year=config["bmi_intervention_end_year"],
        bmi_intervention_coverage=config["bmi_intervention_coverage"],
        bmi_intervention_effectiveness=config["bmi_intervention_effectiveness"],
        seed=42,
    )


@fixture
def expected_population():
    return pd.read_parquet(Path("tests/pearl/assets/pearl_test/population.parquet"))


def test_pearl_single_threaded(parameter, expected_population, output_folder):
    """
    Pearl should run identically when seeded in a single threaded environment.
    """
    Pearl(parameter, parameter.group_name, 1).run()

    try:
        result_population = pd.read_parquet(Path(output_folder / "population.parquet"))
        result_population.to_parquet(Path("tests/pearl/assets/pearl_test/population.parquet"))
        assert_frame_equal(result_population, expected_population)
    except Exception as e:
        raise e
    finally:
        shutil.rmtree(output_folder)


def test_pearl_multi_threaded(parameter, expected_population, output_folder):
    """
    Pearl should run identically when seeded in a multi-threaded environment.
    """

    @delayed
    def run(parameter):
        Pearl(parameter, parameter.group_name, 1).run()

    result = []
    for _ in range(3):
        result.append(run(parameter))

    dask.compute(result)

    try:
        result_population = pd.read_parquet(Path(output_folder / "population.parquet"))
        assert_frame_equal(result_population, expected_population)
    except Exception as e:
        raise e
    finally:
        shutil.rmtree(output_folder)
