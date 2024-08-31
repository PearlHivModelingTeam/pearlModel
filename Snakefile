from datetime import datetime

output_dir = directory(f"out/test_{datetime.today().strftime('%Y-%m-%d')}/parquet_output")

rule all:
    input: 
        directory(f"out/test_{datetime.today().strftime('%Y-%m-%d')}/combined/bmi_info.parquet"),
        directory(f"out/test_{datetime.today().strftime('%Y-%m-%d')}/combined/bmi_int_cascade.parquet"),
        directory(f"out/test_{datetime.today().strftime('%Y-%m-%d')}/combined/new_init_age.parquet")

rule create_params:
    output: "param_files/parameters.h5"
    script:
        "scripts/1_create_param_file.py"

rule simulate:
    input: "param_files/parameters.h5"
    output: output_dir
    shell:
        "python scripts/2_simulate.py --config config/test.yaml"

rule combine:
    input: output_dir
    output: 
        directory(f"out/test_{datetime.today().strftime('%Y-%m-%d')}/combined/new_init_age.parquet"),
        directory(f"out/test_{datetime.today().strftime('%Y-%m-%d')}/combined/bmi_int_cascade.parquet")
    shell:
        "python scripts/3_combine_parquet.py --in_dir {output_dir}"

rule aggregate:
    input: output_dir
    output: directory(f"out/test_{datetime.today().strftime('%Y-%m-%d')}/combined/bmi_info.parquet")
    shell:
        "python scripts/5_aggregate.py --in_dir {output_dir}"
