# type: ignore

num_replications = 1000
zero_ = str(0).zfill(len(str(num_replications)))


rule all:
    input: 
        f"out/S0_{num_replications}/final_table.csv",
        f"out/S0_{num_replications}/fig2a.png",
        f"out/S0_{num_replications}/fig2b.png",
        f"out/S0_{num_replications}/fig2c.png",
        f"out/S0_{num_replications}/figure2c_table.csv",
        f"out/S0_{num_replications}/fig2d.png",
        f"out/S0_{num_replications}/figure2d_table.csv",
        f"out/S0_{num_replications}/fig3a.png",
        f"out/S0_{num_replications}/figure3a_table.csv",
        f"out/S0_{num_replications}/fig3b.png",
        f"out/S0_{num_replications}/figure3b_table.csv",
        f"out/S0_{num_replications}/fig3c.png",
        f"out/S0_{num_replications}/figure3c_table.csv",
        f"out/S0_{num_replications}/fig3d.png",
        f"out/S0_{num_replications}/figure3d_table.csv",
        f"out/S0_{num_replications}/tornado.png",

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
        "out/{config}/parquet_output/het_black_female/replication_" + f"{zero_}/new_init_age.parquet",
        "out/{config}/parquet_output/het_black_female/replication_" + f"{zero_}/bmi_int_cascade.parquet",
        "out/{config}/parquet_output/het_black_female/replication_" + f"{zero_}/bmi_int_dm_prev.parquet",
    params:
        config_file = "config/{config}.yaml"
    shell:
        "python scripts/2_simulate.py --config {params.config_file} --overwrite"

rule combine:
    input: 
        "out/{config}/parquet_output/het_black_female/replication_" + f"{zero_}/new_init_age.parquet",
        "out/{config}/parquet_output/het_black_female/replication_" + f"{zero_}/bmi_int_cascade.parquet",
        "out/{config}/parquet_output/het_black_female/replication_" + f"{zero_}/bmi_int_dm_prev.parquet",
        combine_dir = "out/{config}/parquet_output",
    output: 
        directory("out/{config}/combined/new_init_age.parquet"),
        directory("out/{config}/combined/bmi_int_cascade.parquet"),
        directory("out/{config}/combined/bmi_int_dm_prev.parquet"),
        directory("out/{config}/combined/parameters.parquet"),
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
        f"out/S0_{num_replications}/combined/bmi_info.parquet",
        f"out/S0_{num_replications}/combined/bmi_int_cascade.parquet",
        f"out/S0_{num_replications}/combined/new_init_age.parquet",
        f"out/S0_{num_replications}/combined/bmi_int_dm_prev.parquet",
        f"out/S3_{num_replications}/combined/bmi_info.parquet",
        f"out/S3_{num_replications}/combined/bmi_int_cascade.parquet",
        f"out/S3_{num_replications}/combined/new_init_age.parquet",
        f"out/S3_{num_replications}/combined/bmi_int_dm_prev.parquet",

    output:
        f"out/S0_{num_replications}/final_table.csv",
        f"out/S0_{num_replications}/fig2a.png",
        f"out/S0_{num_replications}/fig2b.png",
        f"out/S0_{num_replications}/fig2c.png",
        f"out/S0_{num_replications}/figure2c_table.csv",
        f"out/S0_{num_replications}/fig2d.png",
        f"out/S0_{num_replications}/figure2d_table.csv",
        f"out/S0_{num_replications}/fig3a.png",
        f"out/S0_{num_replications}/figure3a_table.csv",
        f"out/S0_{num_replications}/fig3b.png",
        f"out/S0_{num_replications}/figure3b_table.csv",
        f"out/S0_{num_replications}/fig3c.png",
        f"out/S0_{num_replications}/figure3c_table.csv",
        f"out/S0_{num_replications}/fig3d.png",
        f"out/S0_{num_replications}/figure3d_table.csv",
    params:
        out_dir = f"out/S0_{num_replications}",
        baseline = f"out/S0_{num_replications}/combined",
        variable = f"out/S3_{num_replications}/combined",
    shell:
        "python scripts/6_bmi_plots.py --baseline {params.baseline} --variable {params.variable} --out_dir {params.out_dir}"

rule bmi_SA:
    input:
        f"out/S3_SA_{num_replications}/combined/bmi_int_cascade.parquet",
        f"out/S3_SA_{num_replications}/combined/new_init_age.parquet",
        f"out/S3_SA_{num_replications}/combined/bmi_int_dm_prev.parquet",
        f"out/S3_SA_{num_replications}/combined/parameters.parquet",
        
        f"out/S0_SA_{num_replications}/combined/bmi_int_cascade.parquet",
        f"out/S0_SA_{num_replications}/combined/new_init_age.parquet",
        f"out/S0_SA_{num_replications}/combined/bmi_int_dm_prev.parquet",
        f"out/S0_SA_{num_replications}/combined/parameters.parquet",
    output:
        f"out/S0_{num_replications}/tornado.png",
    params:
        out_dir = f"out/S0_{num_replications}",
        baseline = f"out/S0_SA_{num_replications}/combined",
        variable = f"out/S3_SA_{num_replications}/combined",
    shell:
        "python scripts/7_bmi_sa.py --baseline {params.baseline} --variable {params.variable} --out_dir {params.out_dir}"
