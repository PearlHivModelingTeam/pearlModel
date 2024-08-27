from datetime import datetime

output_dir = directory(f"out/test_{datetime.today().strftime('%Y-%m-%d')}")

rule all:
    input: output_dir 

rule create_params:
    output: "param_files/parameters.h5"
    script:
        "scripts/1_create_param_file.py"

rule simulate:
    input: "param_files/parameters.h5"
    output: output_dir 
    shell:
        "python scripts/2_simulate.py --config config/test.yaml"
