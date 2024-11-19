# Imports
import cProfile
from datetime import datetime
import pstats
import random

from pearl.definitions import PROJECT_DIR
from pearl.model import Parameters, Pearl


def run(group_name_run, replication_run, seed):
    parameters = Parameters(
        path=param_file_path,
        output_folder=None,
        replication=replication_run,
        group_name=group_name_run,
        new_dx="base",
        final_year=2035,
        mortality_model="by_sex_race_risk",
        mortality_threshold_flag=1,
        idu_threshold="2x",
        bmi_intervention_scenario=0,
        bmi_intervention_start_year=2012,
        bmi_intervention_end_year=2017,
        bmi_intervention_coverage=1,
        bmi_intervention_effectiveness=1,
        seed=seed,
        sa_variables=[],
    )
    Pearl(parameters).run()


if __name__ == "__main__":
    with cProfile.Profile() as profile:
        # set a seed for the main thread
        random.seed(42)

        start_time = datetime.now()

        pearl_path = PROJECT_DIR
        param_file_path = pearl_path / "param_files/parameters.h5"

        group_names = [
            "msm_white_male",
        ]

        num_seeds = 1
        seeds = []
        seed = random.randint(1, 100000000)
        results = []
        for group_name_run in group_names:
            while seed in seeds:
                seed = random.randint(1, 100000000)
            results.append(run(group_name_run, 1, seed=seed))

    result = pstats.Stats(profile)
    result.sort_stats(pstats.SortKey.TIME)
    result.print_stats()
    result.dump_stats("results.prof")
