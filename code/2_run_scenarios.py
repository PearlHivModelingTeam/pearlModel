# Imports
import pearl
import yaml
import subprocess
from pathlib import Path
from datetime import datetime
import os

start_time = datetime.now()
pearl_path = Path('..')
date_string = datetime.today().strftime('%Y-%m-%d')

config_files = ['bmi_S1.yaml']
for f in config_files:
    print(f"reading config file: {f} ...")
    # paths
    config_file_path = pearl_path/'config'/f
    output_root_path = pearl_path/f'out/{config_file_path.stem}_{date_string}'
    ############
    print(f"Running simulations in parallel ...")
    # Specify the command to call another Python script
    command = ["python3", f"{pearl_path}/code/2_simulate_server.py", "--config", f"{config_file_path}","--overwrite"]
    # Use subprocess to run the command and redirect both output and error to the same file
    output_and_error_log = f"2_out_{f}.log"
    with open(output_and_error_log, "w") as log_file:
        process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
    # Wait for the process to finish
    process.communicate()
    # Check the exit code
    exit_code = process.returncode
    print("simulations ran with exit Code:", exit_code)
    ############
    print(f"Creating HDF outputs ...")
    # Specify the command to call another Python script
    command = ["python3", f"{pearl_path}/code/3_convert_csv_to_hdf.py","--dir", f"{output_root_path}"]
    output_and_error_log = f"3_out_{f}.log"
    with open(output_and_error_log, "w") as log_file:
        process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
    # Wait for the process to finish
    process.communicate()
    # Check the exit code
    exit_code = process.returncode
    print("outputs converted to hdf with exit Code:", exit_code)


end_time = datetime.now()
print(f'**** Elapsed Time: {end_time - start_time} ****')
