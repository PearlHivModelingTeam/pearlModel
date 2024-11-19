# declare imports
import argparse
from datetime import datetime
from pathlib import Path

import dask.dataframe as dd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pearl.post_processing.bmi import (
    add_summary,
    calc_overall_risk,
    calc_percentage,
    calc_percentage_and_add_summary,
    calc_risk_by_group,
    clean_control,
    create_summary_table,
    group_order,
    group_order_with_sub_total,
    group_title_dict,
    palette,
    rearrange_group_order,
    round_thousand,
    calc_dm_prop,
    add_sub_total,
    calc_overall_bmi_risk
)

if __name__ == "__main__":
    start_time = datetime.now()
    df_summary_dict = {}

    # Define the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline")
    parser.add_argument("--variable")
    parser.add_argument("--out_dir")
    args = parser.parse_args()

    baseline_dir = Path(args.baseline)
    variable_dir = Path(args.variable)
    out_dir = Path(args.out_dir)

    start_year = 2013
    end_year = 2017
    font_size = 9
    
    ##################################################################################################################################
    # Age Groups
    ##################################################################################################################################
    # we will look at the "bmi_int_dm_prev.h5" for S0
    bmi_int_dm_prev = dd.read_parquet(baseline_dir / "dm_final_output.parquet").reset_index()

    # Add Overall
    all_but_group = list(bmi_int_dm_prev.columns[1:])
    bmi_int_dm_prev_overall = bmi_int_dm_prev.groupby(all_but_group).sum().reset_index()
    bmi_int_dm_prev_overall["group"] = "overall"
    bmi_int_dm_prev = dd.concat([bmi_int_dm_prev, bmi_int_dm_prev_overall], ignore_index=True)

    # type the dataframe for space efficiency
    bmi_int_dm_prev = bmi_int_dm_prev.astype(
        {
            "group": "str",
            "replication": "int16",
            "bmiInt_scenario": np.int8,
            "h1yy": np.int16,
            "init_age_group": np.int8,
            "bmiInt_impacted": bool,
            "dm": bool,
            "t_dm": np.int16,
            "n": np.int16,
        }
    )

    # clean to control specifications
    control_bmi_int_dm_prev = clean_control(bmi_int_dm_prev, only_eligible=True, only_received = True)

    dm_risk_table = calc_overall_risk(control_bmi_int_dm_prev).compute()
    
    ##################################################################################################################################
    # we will look at the "bmi_int_dm_prev.h5" for S0
    bmi_int_dm_prev = dd.read_parquet(variable_dir / "dm_final_output.parquet").reset_index()

    # Add Overall
    all_but_group = list(bmi_int_dm_prev.columns[1:])
    bmi_int_dm_prev_overall = bmi_int_dm_prev.groupby(all_but_group).sum().reset_index()
    bmi_int_dm_prev_overall["group"] = "overall"
    bmi_int_dm_prev = dd.concat([bmi_int_dm_prev, bmi_int_dm_prev_overall], ignore_index=True)

    # type the dataframe for space efficiency
    bmi_int_dm_prev = bmi_int_dm_prev.astype(
        {
            "group": "str",
            "replication": "int16",
            "bmiInt_scenario": np.int8,
            "h1yy": np.int16,
            "init_age_group": np.int8,
            "bmiInt_impacted": bool,
            "dm": bool,
            "t_dm": np.int16,
            "n": np.int16,
        }
    )

    # clean to control specifications
    control_bmi_int_dm_prev = clean_control(bmi_int_dm_prev, only_eligible=True, only_received = True)

    # Figure 4A
    dm_risk_table_S1 = calc_overall_risk(control_bmi_int_dm_prev).compute()
    
    
    S0_sample = (
        dm_risk_table.groupby("init_age_group")
        .apply(lambda x: x.sample(20, replace=True))
        .reset_index(drop=True)
        )

    S1_sample = (
            dm_risk_table_S1.groupby("init_age_group")
            .apply(lambda x: x.sample(20, replace=True))
            .reset_index(drop=True)
        )

    S0_sample = S0_sample.sort_values(by = 'init_age_group').reset_index(drop = True)
    S1_sample = S1_sample.sort_values(by = 'init_age_group').reset_index(drop = True)

    risk_df = S0_sample.copy()
    risk_df['risk_S1'] =S1_sample['risk']

    risk_df['abs_reduction'] = risk_df['risk'] - risk_df['risk_S1']
    risk_df['rel_reduction'] = (risk_df['risk'] - risk_df['risk_S1'])/risk_df['risk'] * 100
    
    risk_df = S0_sample.copy()
    risk_df['risk_S1'] =S1_sample['risk']

    risk_df['abs_reduction'] = risk_df['risk'] - risk_df['risk_S1']
    risk_df['rel_reduction'] = (risk_df['risk'] - risk_df['risk_S1'])/risk_df['risk'] * 100
    
    # Age Abs Reduction
    ##########################################################################################
    bar_ax = sns.boxplot(
        x=risk_df["init_age_group"],
        y=risk_df["abs_reduction"],
        palette=palette,
        showfliers=False,
        hue=risk_df["init_age_group"],
    )
    
    bar_ax.set_ylabel(
        "Absolute percentage point risk reduction in incident diabetes\n diagnosis (per 1,000 pys) with vs. without the intervention",
        fontsize=font_size,
    )
    bar_ax.set_xlabel("Age Group at ART Initiation", fontsize = font_size, labelpad = 5)
    bar_ax.set_xticks(range(0, 7))
    bar_ax.set_xticklabels(["<20", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"])
    
    bar_ax.get_legend().remove()
    bar_fig = bar_ax.get_figure()
    bar_fig.savefig(out_dir / "bmi_age_group_risk/age_group_abs_reduction.png", bbox_inches="tight", dpi=1000)
    plt.show()
    plt.clf()
    
    df = (
        risk_df.groupby("init_age_group")[["abs_reduction"]]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
    )
    df.columns = ["group", 0.025, 0.5, 0.975]
    df["formatted"] = df.apply(
        lambda row: f"{row[0.50]:.1f} [{row[0.025]:.1f} - {row[0.975]:.1f}]", axis=1
    )
    df = rearrange_group_order(df)
    df.to_csv(out_dir / "bmi_age_group_risk/age_group_abs_risk_reduction_table.csv")
    
    # Age Rel Reduction
    ##########################################################################################
    bar_ax = sns.boxplot(
        x=risk_df["init_age_group"],
        y=risk_df["rel_reduction"],
        palette=palette,
        showfliers=False,
        hue=risk_df["init_age_group"],
    )
    bar_ax.set_ylabel(
        "Relative 5-year risk reduction in incident DM diagnoses \n with (vs. without) the intervention",
        fontsize=8.5,
    )
    bar_ax.set_xlabel("Age Group at ART Initiation")
    bar_ax.set_xticks(range(0, 7))
    bar_ax.set_xticklabels(["<20", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"])
    
    bar_ax.get_legend().remove()
    bar_fig = bar_ax.get_figure()
    bar_fig.savefig(out_dir / "bmi_age_group_risk/age_group_rel_reduction.png", bbox_inches="tight", dpi=1000)
    plt.show()
    plt.clf()
    
    df = (
        risk_df.groupby("init_age_group")[["rel_reduction"]]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
    )
    df.columns = ["group", 0.025, 0.5, 0.975]
    df["formatted"] = df.apply(
        lambda row: f"{row[0.50]:.1f} [{row[0.025]:.1f} - {row[0.975]:.1f}]", axis=1
    )
    df = rearrange_group_order(df)
    df.to_csv(out_dir / "bmi_age_group_risk/age_group_rel_risk_reduction_table.csv")
    

    ##################################################################################################################################
    # BMI Groups
    ##################################################################################################################################
    ##################################################################################################################################
    # we will look at the "bmi_int_dm_prev.h5" for S0
    bmi_int_dm_prev = dd.read_parquet(baseline_dir / "bmi_cat_final_output.parquet").reset_index()

    # Add Overall
    all_but_group = list(bmi_int_dm_prev.columns[1:])
    bmi_int_dm_prev_overall = bmi_int_dm_prev.groupby(all_but_group).sum().reset_index()
    bmi_int_dm_prev_overall["group"] = "overall"
    bmi_int_dm_prev = dd.concat([bmi_int_dm_prev, bmi_int_dm_prev_overall], ignore_index=True)

    # type the dataframe for space efficiency
    bmi_int_dm_prev = bmi_int_dm_prev.astype(
        {
            "group": "str",
            "replication": "int16",
            "bmiInt_scenario": np.int8,
            "h1yy": np.int16,
            "init_bmi_group": np.int8,
            "bmiInt_impacted": bool,
            "dm": bool,
            "t_dm": np.int16,
            "n": np.int16,
        }
    )

    # clean to control specifications
    control_bmi_int_dm_prev = clean_control(bmi_int_dm_prev, only_eligible=True, only_received = True)

    dm_risk_table = calc_overall_bmi_risk(control_bmi_int_dm_prev).compute()
    
    ##################################################################################################################################
    # we will look at the "bmi_int_dm_prev.h5" for S0
    bmi_int_dm_prev = dd.read_parquet(variable_dir / "bmi_cat_final_output.parquet").reset_index()

    # Add Overall
    all_but_group = list(bmi_int_dm_prev.columns[1:])
    bmi_int_dm_prev_overall = bmi_int_dm_prev.groupby(all_but_group).sum().reset_index()
    bmi_int_dm_prev_overall["group"] = "overall"
    bmi_int_dm_prev = dd.concat([bmi_int_dm_prev, bmi_int_dm_prev_overall], ignore_index=True)

    # type the dataframe for space efficiency
    bmi_int_dm_prev = bmi_int_dm_prev.astype(
        {
            "group": "str",
            "replication": "int16",
            "bmiInt_scenario": np.int8,
            "h1yy": np.int16,
            "init_bmi_group": np.int8,
            "bmiInt_impacted": bool,
            "dm": bool,
            "t_dm": np.int16,
            "n": np.int16,
        }
    )

    # clean to control specifications
    control_bmi_int_dm_prev = clean_control(bmi_int_dm_prev, only_eligible=True, only_received = True)

    dm_risk_table_S1 = calc_overall_bmi_risk(control_bmi_int_dm_prev).compute()
    
    S0_sample = (
        dm_risk_table.groupby("init_bmi_group")
        .apply(lambda x: x.sample(20, replace=True))
        .reset_index(drop=True)
        )

    S1_sample = (
            dm_risk_table_S1.groupby("init_bmi_group")
            .apply(lambda x: x.sample(20, replace=True))
            .reset_index(drop=True)
        )

    S0_sample = S0_sample.sort_values(by = 'init_bmi_group').reset_index(drop = True)
    S1_sample = S1_sample.sort_values(by = 'init_bmi_group').reset_index(drop = True)

    risk_df = S0_sample.copy()
    risk_df['risk_S1'] =S1_sample['risk']

    risk_df['abs_reduction'] = risk_df['risk'] - risk_df['risk_S1']
    risk_df['rel_reduction'] = (risk_df['risk'] - risk_df['risk_S1'])/risk_df['risk'] * 100
    
    pre_art_bmi_bins = [0, 18.5, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, float("inf")]
    # Create a label map
    bmi_group_map = {i: f"{pre_art_bmi_bins[i]}-{pre_art_bmi_bins[i+1]}" for i in range(len(pre_art_bmi_bins) - 1)}
    bmi_group_map[12] = '> 30'
    group_order = list(bmi_group_map.values())

    # bmi_group_map = {1:'[18.5-21.6]', 2:'[21.7-24.9]',3:'[25-27.4]',4:'[27.5-29.9]'}
    # group_order = ['[18.5-21.6]','[21.7-24.9]','[25-27.4]','[27.5-29.9]']
    risk_df["init_bmi_group"] = risk_df["init_bmi_group"].map(bmi_group_map)
    
    # BMI Group Abs Reduction
    #########################################################################################################################
    bar_ax = sns.boxplot(
        x=risk_df["init_bmi_group"],
        y=risk_df["abs_reduction"],
        palette=palette,
        showfliers=False,
        hue=risk_df["init_bmi_group"],
        order = group_order[1:-1],
        )
    bar_ax.set_ylabel(
        "Absolute percentage point risk reduction in incident diabetes\n diagnosis (per 1,000 pys) with vs. without the intervention",
        fontsize=8.5,
    )
    bar_ax.set_xlabel("BMI Group at ART Initiation")
    bar_ax.tick_params(axis="x", rotation=90)
    bar_fig = bar_ax.get_figure()
    bar_fig.savefig(out_dir / "bmi_age_group_risk/bmi_group_abs_reduction.png", bbox_inches="tight", dpi=1000)
    plt.show()
    plt.clf()
    
    df = (
        risk_df.groupby("init_bmi_group")[["abs_reduction"]]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
    )
    df.columns = ["group", 0.025, 0.5, 0.975]
    df["formatted"] = df.apply(
        lambda row: f"{row[0.50]:.1f} [{row[0.025]:.1f} - {row[0.975]:.1f}]", axis=1
    )
    df = rearrange_group_order(df)
    df.to_csv(out_dir / "bmi_age_group_risk/bmi_group_abs_risk_reduction_table.csv")
    
    # BMI Group Rel Reduction
    #########################################################################################################################
    bar_ax = sns.boxplot(
            x=risk_df["init_bmi_group"],
            y=risk_df["rel_reduction"],
            palette=palette,
            showfliers=False,
            hue=risk_df["init_bmi_group"],
            order = group_order[1:-1],
        )
    bar_ax.set_ylabel(
        "Relative 5-year risk reduction in incident DM diagnoses \n with (vs. without) the intervention",
        fontsize=8.5,
    )
    bar_ax.set_xlabel("BMI Group at ART Initiation")
    bar_ax.tick_params(axis="x", rotation=90)
    bar_fig = bar_ax.get_figure()
    bar_fig.savefig(out_dir / "bmi_age_group_risk/bmi_group_rel_reduction.png", bbox_inches="tight", dpi=1000)
    plt.show()
    plt.clf()
    
    df = (
        risk_df.groupby("init_bmi_group")[["rel_reduction"]]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
    )
    df.columns = ["group", 0.025, 0.5, 0.975]
    df["formatted"] = df.apply(
        lambda row: f"{row[0.50]:.1f} [{row[0.025]:.1f} - {row[0.975]:.1f}]", axis=1
    )
    df = rearrange_group_order(df)
    # to save output
    df.to_csv(out_dir / "bmi_age_group_risk/bmi_group_rel_risk_reduction_table.csv")
    