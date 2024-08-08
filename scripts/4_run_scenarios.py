# Imports
import argparse
import subprocess
from datetime import datetime

from pearl.definitions import PROJECT_DIR

# SETUP: list of config files to run:
config_files = ["test.yaml", "test.yaml", "test.yaml", "test.yaml"]
###############################################################################################
print("Starting 4_run_scenarios.py")
start_time = datetime.now()
pearl_path = PROJECT_DIR
date_string = datetime.today().strftime("%Y-%m-%d")

# CREATING PARAMETER FILE:
parser = argparse.ArgumentParser()
parser.add_argument("--createParam", action="store_true")
args = parser.parse_args()
if args.createParam:
    print("Creating the parameter file ...")
    command = ["python3", f"{pearl_path}/scripts/1_create_param_file.py"]
    output_and_error_log = "1_out.log"
    with open(output_and_error_log, "w") as log_file:
        process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
    # Wait for the process to finish
    process.communicate()
    # Check the exit code
    exit_code = process.returncode
    print("param file created with exit Code:", exit_code)

# RUNNING LIST OF SCENARIOS:
for f in config_files:
    print(f"reading config file: {f} ...")
    # paths
    config_file_path = pearl_path / "config" / f
    output_root_path = pearl_path / f"out/{config_file_path.stem}_{date_string}"
    ############
    print("Running simulations in parallel ...")
    # Specify the command to call another Python script
    command = [
        "python3",
        f"{pearl_path}/scripts/2_simulate.py",
        "--config",
        f"{config_file_path}",
        "--overwrite",
    ]
    # Use subprocess to run the command and redirect both output and error to the same file
    output_and_error_log = f"2_out_{config_file_path.stem}.log"
    with open(output_and_error_log, "w") as log_file:
        process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
    # Wait for the process to finish
    process.communicate()
    # Check the exit code
    exit_code = process.returncode
    print("simulations ran with exit Code:", exit_code)
    ############
    print("Combining outputs ...")
    # Specify the command to call another Python script
    command = [
        "python3",
        f"{pearl_path}/scripts/3_combine_parquet.py",
        "--in_dir",
        f"{output_root_path}/parquet_output",
    ]
    output_and_error_log = f"3_out_{config_file_path.stem}.log"
    with open(output_and_error_log, "w") as log_file:
        process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
    # Wait for the process to finish
    process.communicate()
    # Check the exit code
    exit_code = process.returncode
    print("outputs converted to hdf with exit Code:", exit_code)

end_time = datetime.now()
print(f"**** Elapsed Time: {end_time - start_time} ****")
