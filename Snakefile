rule all:
    input: "/out"
        

rule create_params:
    output: "/param_files/parameters.h5"
    script:
        "/scripts/1_create_param_file.py"

rule simulate:
    input: "/param_files/parameters.h5"
    output: "/out"
    shell:
        "python scripts/2_simulate.py --config config/test.yaml"
