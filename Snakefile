# type: ignore

output_dir = directory(f"out/{config}/parquet_output")

for config in ["S0_1000", "S3_1000"]:
    rule:
        input: 
            directory(f"out/{config}/combined/bmi_info.parquet"),
            directory(f"out/{config}/combined/bmi_int_cascade.parquet"),
            directory(f"out/{config}/combined/new_init_age.parquet")

    rule:
        output: "param_files/parameters.h5"
        script:
            "scripts/1_create_param_file.py"

    rule:
        input: "param_files/parameters.h5"
        output: output_dir
        shell:
            "python scripts/2_simulate.py --config config/{config}.yaml --overwrite"

    rule:
        input: output_dir
        output: 
            directory(f"out/{config}/combined/new_init_age.parquet"),
            directory(f"out/{config}/combined/bmi_int_cascade.parquet")
        shell:
            "python scripts/3_combine_parquet.py --in_dir {output_dir}"

    rule:
        input: output_dir
        output: directory(f"out/{config}/combined/bmi_info.parquet")
        shell:
            "python scripts/5_aggregate.py --in_dir {output_dir}"
