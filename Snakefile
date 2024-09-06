# type: ignore

rule all:
    input: 
        "out/final_table.csv",
        "out/fig2a.png",
        "out/fig2b.png",
        "out/fig2c.png",
        "out/figure2c_table.csv",
        "out/fig2d.png",
        "out/figure2d_table.csv",
        "out/fig3a.png",
        "out/figure3a_table.csv",
        "out/fig3b.png",
        "out/figure3b_table.csv",
        "out/fig3c.png",
        "out/figure3c_table.csv",
        "out/fig3d.png",
        "out/figure3d_table.csv",

rule create_params:
    output: 
        "param_files/parameters.h5"
    script:
        "scripts/1_create_param_file.py"

rule simulate:
    input: 
        "param_files/parameters.h5",
    output:
        directory("out/{config}/parquet_output"),
        "out/{config}/parquet_output/het_black_female/replication_00/new_init_age.parquet",
        "out/{config}/parquet_output/het_black_female/replication_00/bmi_int_cascade.parquet",
        "out/{config}/parquet_output/het_black_female/replication_00/bmi_int_dm_prev.parquet",
    params:
        config_file = "config/{config}.yaml"
    shell:
        "python scripts/2_simulate.py --config {params.config_file} --overwrite"

rule combine:
    input: 
        "out/{config}/parquet_output/het_black_female/replication_00/new_init_age.parquet",
        "out/{config}/parquet_output/het_black_female/replication_00/bmi_int_cascade.parquet",
        "out/{config}/parquet_output/het_black_female/replication_00/bmi_int_dm_prev.parquet",
        combine_dir = "out/{config}/parquet_output",
    output: 
        "out/{config}/combined/new_init_age.parquet",
        "out/{config}/combined/bmi_int_cascade.parquet",
        "out/{config}/combined/bmi_int_dm_prev.parquet",
    shell:
        "python scripts/3_combine_parquet.py --in_dir {input.combine_dir}"

rule aggregate:
    input: 
        combine_dir = "out/{config}/parquet_output"
    output: 
        directory("out/{config}/combined/bmi_info.parquet")
    shell:
        "python scripts/5_aggregate.py --in_dir {input.combine_dir}"

rule bmi_paper_outputs:
    input:
        "out/S0_1000/combined/bmi_info.parquet",
        "out/S0_1000/combined/bmi_int_cascade.parquet",
        "out/S0_1000/combined/new_init_age.parquet",
        "out/S0_1000/combined/bmi_int_dm_prev.parquet",
        "out/S3_1000/combined/bmi_info.parquet",
        "out/S3_1000/combined/bmi_int_cascade.parquet",
        "out/S3_1000/combined/new_init_age.parquet",
        "out/S3_1000/combined/bmi_int_dm_prev.parquet",
    output:
        "out/final_table.csv",
        "out/fig2a.png",
        "out/fig2b.png",
        "out/fig2c.png",
        "out/figure2c_table.csv",
        "out/fig2d.png",
        "out/figure2d_table.csv",
        "out/fig3a.png",
        "out/figure3a_table.csv",
        "out/fig3b.png",
        "out/figure3b_table.csv",
        "out/fig3c.png",
        "out/figure3c_table.csv",
        "out/fig3d.png",
        "out/figure3d_table.csv",
    params:
        out_dir = directory("out"),
        baseline = "out/S0_1000/combined",
        variable = "out/S3_1000/combined",
    shell:
        "python scripts/6_bmi_plots.py --baseline {params.baseline} --variable {params.variable} --out_dir {params.out_dir}"
