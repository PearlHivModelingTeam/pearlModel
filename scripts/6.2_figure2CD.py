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
    calc_overall_bmi_risk,
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

    ########################################################################################################
    # 2c
    bmi_int_dm_prev = dd.read_parquet(baseline_dir /'dm_final_output.parquet').reset_index()

    # Add Overall
    all_but_group = list(bmi_int_dm_prev.columns[1:])
    bmi_int_dm_prev_overall = bmi_int_dm_prev.groupby(all_but_group).sum().reset_index()
    bmi_int_dm_prev_overall['group'] = 'overall'
    bmi_int_dm_prev = dd.concat([bmi_int_dm_prev, bmi_int_dm_prev_overall], ignore_index=True)

    # type the dataframe for space efficiency
    bmi_int_dm_prev = bmi_int_dm_prev.astype({'group':'str', 'replication':'int16', 'bmiInt_scenario':np.int8, 'h1yy': np.int16, 'bmiInt_impacted':bool, 'dm': bool, 't_dm': np.int16, 'n': np.int16})

    # clean to control specifications
    control_bmi_int_dm_prev = clean_control(bmi_int_dm_prev, only_eligible=True, only_received = True).compute()

    group_dm_risk_table = calc_risk_by_group(control_bmi_int_dm_prev, 7).compute()

    group_dm_risk_table["group"] = group_dm_risk_table["group"].map(group_title_dict)

    group_risk_ax = sns.boxplot(
        x=group_dm_risk_table["group"],
        y=group_dm_risk_table["risk"],
        color="seagreen",
        showfliers=False,
        palette=palette,
        hue=group_dm_risk_table["group"],
        order=group_order[:-1],
        hue_order=group_order,
    )

    group_risk_ax.tick_params(axis="x", rotation=90)

    ########################################################
    # last_group = group_order[-1]
    # secondary_ax = diff_ax.twinx()  # Create secondary y-axis
    
    # # Plot the last group's data on the secondary axis
    # sns.boxplot(
    #     x=abs_sample_diff_plot[abs_sample_diff_plot["group"] == last_group]["group"],
    #     y=-abs_sample_diff_plot[abs_sample_diff_plot["group"] == last_group]["risk"],
    #     ax=secondary_ax,
    #     color="skyblue",
    #     showfliers=False,
    #     width=0.5,
    #     order=group_order,
    # )
    
    # # Make the secondary y-axis labels visible
    # secondary_ax.set_ylabel("* Overall", fontsize=8.5)
    
    # # Set y-labels to same font size for clarity
    # secondary_ax.tick_params(axis='y', labelsize=8.5)
    # diff_ax.tick_params(axis='y', labelsize=8.5)
    
    # diff_ax.axvline(x=len(group_order)-1.5, color="red", linestyle="--")
    
    overall_info = group_dm_risk_table[group_dm_risk_table['group'] == 'Overall']
    df = (
        overall_info.groupby("group")[["risk"]]
        .quantile([0.05, 0.5, 0.95])
        .unstack()
        .reset_index()
    )
    df.columns = ["group", 0.025, 0.5, 0.975]
    overall_info_text = f'* Overall Median: {df[0.5][0]:.1f}     * Overall 95% UR: {df[0.025][0]:.1f}-{df[0.975][0]:.1f}'
    ########################################################
    
    group_risk_ax.set_xlabel(overall_info_text)
    group_risk_ax.set_ylabel("Risk of incident diabetes diagnosis (per 1,000 pys)\n among subgroups of PLWH")
    group_risk_ax.set_ylim(0, 40)
    group_risk_fig = group_risk_ax.get_figure()
    group_risk_fig.savefig(out_dir / "fig2c.png", bbox_inches="tight")
    plt.show()
    plt.clf()

    # table 2c
    df = (
        group_dm_risk_table.groupby("group")[["risk"]]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
    )
    df.columns = ["group", 0.025, 0.5, 0.975]
    df["formatted"] = df.apply(
        lambda row: f"{row[0.50]:.1f} [{row[0.025]:.1f} - {row[0.975]:.1f}]",
        axis=1,
    )
    df = rearrange_group_order(df)
    df.to_csv(out_dir / "figure2c_table.csv")
    df_summary_dict['Control|7-year Risk of DM Diagnosis Post-ART Initiation'] = df['formatted']

    # 2D
    # group_dm_risk_table = calc_risk_by_group(control_bmi_int_dm_prev, 7).compute()

    # group_dm_risk_table["group"] = group_dm_risk_table["group"].map(group_title_dict)

    group_risk_ax = sns.boxplot(
        x=group_dm_risk_table["group"],
        y=group_dm_risk_table["dm_num"],
        color="seagreen",
        showfliers=False,
        palette=palette,
        hue=group_dm_risk_table["group"],
        order=group_order[:-1],
        hue_order=group_order,
    )

    group_risk_ax.tick_params(axis="x", rotation=90)

   ########################################################
    # last_group = group_order[-1]
    # secondary_ax = diff_ax.twinx()  # Create secondary y-axis
    
    # # Plot the last group's data on the secondary axis
    # sns.boxplot(
    #     x=abs_sample_diff_plot[abs_sample_diff_plot["group"] == last_group]["group"],
    #     y=-abs_sample_diff_plot[abs_sample_diff_plot["group"] == last_group]["risk"],
    #     ax=secondary_ax,
    #     color="skyblue",
    #     showfliers=False,
    #     width=0.5,
    #     order=group_order,
    # )
    
    # # Make the secondary y-axis labels visible
    # secondary_ax.set_ylabel("* Overall", fontsize=8.5)
    
    # # Set y-labels to same font size for clarity
    # secondary_ax.tick_params(axis='y', labelsize=8.5)
    # diff_ax.tick_params(axis='y', labelsize=8.5)
    
    # diff_ax.axvline(x=len(group_order)-1.5, color="red", linestyle="--")
    
    overall_info = group_dm_risk_table[group_dm_risk_table['group'] == 'Overall']
    df = (
        overall_info.groupby("group")[["dm_num"]]
        .quantile([0.05, 0.5, 0.95])
        .unstack()
        .reset_index()
    )
    df.columns = ["group", 0.025, 0.5, 0.975]
    overall_info_text = f'* Overall Median: {df[0.5][0]:.0f}     * Overall 95% UR: {df[0.025][0]:.0f}-{df[0.975][0]:.0f}'
    ########################################################
    
    group_risk_ax.set_xlabel(overall_info_text)
    group_risk_ax.set_ylabel("7-year number of incident diabetes diagnosis\n among subgroups of PLWH")
    group_risk_fig = group_risk_ax.get_figure()
    group_risk_fig.savefig(out_dir / "fig2d.png", bbox_inches="tight")
    plt.show()
    plt.clf()

    df = (
        group_dm_risk_table.groupby("group")[["dm_num"]]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
    )
    df.columns = ["group", 0.025, 0.5, 0.975]
    df["formatted"] = df.apply(
        lambda row: f"{row[0.50]:.1f} [{row[0.025]:.1f} - {row[0.975]:.1f}]", axis=1
    )
    df = rearrange_group_order(df)
    df.to_csv(out_dir / "figure2d_table.csv")
    df_summary_dict['Control|7-year Number of DM Diagnosis Post-ART Initiation'] = df['formatted']

    pd.DataFrame(df_summary_dict).to_csv(out_dir/'df_summary.csv', index = False)

    print("Figure 2 Finished.")