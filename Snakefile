# type: ignore

rule all:
    input: 
        directory("out/S0_1000/combined/bmi_info.parquet"),
        directory("out/S0_1000/combined/bmi_int_cascade.parquet"),
        directory("out/S0_1000/combined/new_init_age.parquet"),
        directory("out/S3_1000/combined/bmi_info.parquet"),
        directory("out/S3_1000/combined/bmi_int_cascade.parquet"),
        directory("out/S3_1000/combined/new_init_age.parquet"),

rule create_params:
    output: 
        "param_files/parameters.h5"
    script:
        "scripts/1_create_param_file.py"

rule simulate:
    input: 
        "param_files/parameters.h5",
    output: 
        directory("out/{config}/parquet_output")
    params:
        config_file = "config/{config}.yaml"
    shell:
        "python scripts/2_simulate.py --config {params.config_file} --overwrite"

rule combine:
    input: 
        directory("out/{config}/parquet_output")
    output: 
        directory("out/{config}/combined/new_init_age.parquet"),
        directory("out/{config}/combined/bmi_int_cascade.parquet")
    params:
        combine_dir = "out/{config}/parquet_output"
    shell:
        "python scripts/3_combine_parquet.py --in_dir {params.combine_dir}"

rule aggregate:
    input: 
        directory("out/{config}/parquet_output")
    output: 
        directory("out/{config}/combined/bmi_info.parquet")
    params:
        combine_dir = "out/{config}/parquet_output"
    shell:
        "python scripts/5_aggregate.py --in_dir {params.combine_dir}"
