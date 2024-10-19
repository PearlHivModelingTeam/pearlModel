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
    add_sub_total
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

    ##################################################################################################################################
    # we will look at the "bmi_int_dm_prev.h5" for S0
    bmi_int_dm_prev = dd.read_parquet(baseline_dir / "dm_final_output.parquet").reset_index()

    # Now we will work on the remaining percentage columns
    bmi_int_cascade = dd.read_parquet(baseline_dir / "bmi_int_cascade.parquet").reset_index()

    # filter for only starting h1yy after 2013 and before 2017
    control_bmi_int_cascade = bmi_int_cascade.loc[
        (bmi_int_cascade["h1yy"] >= start_year) & (bmi_int_cascade["h1yy"] <= 2017)
    ].compute()

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

    # sum across replications, group, and years_after_h1yy
    control_bmi_int_dm_prev_agg = (
        control_bmi_int_dm_prev.groupby(
            ["group", "years_after_h1yy", "replication", "time_exposure_to_risk"]
        )["n"]
        .sum()
        .reset_index()
        .compute()
    )

    df = control_bmi_int_dm_prev_agg.groupby(['group', 'replication'])[['n']].sum().reset_index()
    df['group'] = df['group'].map(group_title_dict)
    df = df.groupby('group')[['n']].apply(lambda x: x.quantile([0.025,0.5,0.975])).unstack().reset_index()
    df.columns = ['group',0.025, 0.5, 0.975]
    df['formatted'] = df.apply(
        lambda row: '{:.0f} [{:.0f} - {:.0f}]'.format(round(row[0.50], -2), round(row[0.025], -2), round(row[0.975], -2)), axis=1
    )
    df = rearrange_group_order(df)
    df.to_csv(out_dir/'number_receiving_intervention_table.csv')
    df_summary_dict['group'] = df['group']
    df_summary_dict['Control|Number Receiving Intervention'] = df['formatted']

    # Figure 2A
    dm_risk_table = calc_overall_risk(control_bmi_int_dm_prev).compute()
    
    # Graph Overall DM Probability and Population across ART initiation Groups
    pop_ax = sns.barplot(
        x=dm_risk_table["init_age_group"],
        y=dm_risk_table["num"],
        estimator="median",
        color="steelblue",
        errorbar=("pi", 95),
    )
    
    pop_ax.tick_params(axis="x", rotation=90)
    
    rounded_vals = [round_thousand(x) for x in pop_ax.containers[0].datavalues]
    
    pop_ax.bar_label(pop_ax.containers[0], labels=rounded_vals, padding=5)
    
    pop_ax.set_ylabel("Population size under the control arm")
    pop_ax.set_xlabel("Age Group at ART Initiation")
    pop_ax.set_xticks(range(0, 7))
    pop_ax.set_xticklabels(["<20", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"])
    pop_ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
    
    ax2 = pop_ax.twinx()
    percentiles = (
        dm_risk_table.groupby("init_age_group")["risk"].quantile([0.025, 0.5, 0.975]).unstack()
    )
    ax2.plot(
        percentiles.index,
        percentiles.loc[:, 0.50],
        marker="o",
        linestyle="-",
        color="r",
        label="Median Risk",
    )
    ax2.fill_between(
        percentiles.index,
        percentiles.loc[:, 0.025],
        percentiles.loc[:, 0.975],
        color="lightcoral",
        alpha=0.5,
        label="95% CI",
    )
    ax2.set_ylabel("7-year Risk of DM Diagnosis Post-ART Initiation")
    
    pop_fig = pop_ax.get_figure()
    pop_fig.savefig(out_dir / "fig2a.png", bbox_inches="tight")
    # clear the plot
    plt.show()
    plt.clf()
    
    df = dm_risk_table.groupby(['init_age_group'])[['num', 'risk']].quantile([0.025,0.5,0.975]).unstack()
    final_df = pd.DataFrame()
    for col in ['num', 'risk']:
        col_df = df[col].reset_index()
        if col == 'risk':
            final_df[col] = col_df.apply(lambda row: '{:.1f} [{:.1f} - {:.1f}]'.format(row[0.5], row[0.025], row[0.975]), axis=1)
        else:
            final_df[col] = col_df.apply(lambda row: '{:.0f} [{:.0f} - {:.0f}]'.format(round(row[0.5],-2), round(row[0.025],-2), round(row[0.975],-2)), axis=1)
    
    final_df.to_csv(out_dir/'figure2a_table.csv', index = False)

    ## Fig2B
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
    
    pre_art_bmi_bins = [0, 18.5, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, float("inf")]
    # Create a label map
    bmi_group_map = {i: f"{pre_art_bmi_bins[i]}-{pre_art_bmi_bins[i+1]}" for i in range(len(pre_art_bmi_bins) - 1)}
    bmi_group_map[13] = '> 30'
    group_order = list(bmi_group_map.values())
    
    dm_risk_table["init_bmi_group"] = dm_risk_table["init_bmi_group"].map(bmi_group_map)
    
    # Graph Overall DM Probability and Population across BMI initiation Groups
    pop_ax = sns.barplot(
        x=dm_risk_table["init_bmi_group"],
        y=dm_risk_table["num"],
        estimator="median",
        color="steelblue",
        errorbar=("pi", 95),
        order = group_order
    )
    
    pop_ax.tick_params(axis="x", rotation=90)
    
    rounded_vals = [round_thousand(x) for x in pop_ax.containers[0].datavalues]
    
    pop_ax.bar_label(pop_ax.containers[0], labels=rounded_vals, padding=5)
    
    pop_ax.set_ylabel("Population size under the control arm")
    pop_ax.set_xlabel("BMI Group at ART Initiation")
    pop_ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
    
    ax2 = pop_ax.twinx()
    percentiles = (
        dm_risk_table.groupby("init_bmi_group")["risk"].quantile([0.025, 0.5, 0.975]).unstack()
    )
    ax2.plot(
        percentiles.index,
        percentiles.loc[:, 0.50],
        marker="o",
        linestyle="-",
        color="r",
        label="Median Risk",
    )
    ax2.fill_between(
        percentiles.index,
        percentiles.loc[:, 0.025],
        percentiles.loc[:, 0.975],
        color="lightcoral",
        alpha=0.5,
        label="95% CI",
    )
    
    ax2.set_ylabel("7-year Risk of DM Diagnosis Post-ART Initiation")
    
    pop_fig = pop_ax.get_figure()
    pop_fig.savefig(out_dir / "fig2b.png", bbox_inches="tight")
    # clear the plot
    plt.show()
    plt.clf()
    
    df = dm_risk_table.groupby(['init_bmi_group'])[['num', 'risk']].quantile([0.025,0.5,0.975]).unstack()
    final_df = pd.DataFrame()
    for col in ['num', 'risk']:
        col_df = df[col].reset_index()
        if col == 'risk':
            final_df[col] = col_df.apply(lambda row: '{:.1f} [{:.1f} - {:.1f}]'.format(row[0.5], row[0.025], row[0.975]), axis=1)
        else:
            final_df[col] = col_df.apply(lambda row: '{:.0f} [{:.0f} - {:.0f}]'.format(round(row[0.5],-2), round(row[0.025],-2), round(row[0.975],-2)), axis=1)
    
    final_df.to_csv(out_dir/'figure2b_table.csv', index = False)
    
    ######################################################################################################################
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
    
    #################################################################################################################################
    # 2D
    group_dm_risk_table = calc_risk_by_group(control_bmi_int_dm_prev_agg, 7).compute()

    group_dm_risk_table["group"] = group_dm_risk_table["group"].map(group_title_dict)

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
    group_risk_ax.set_ylabel("7-year number of incident diabetes diagnosis among subgroups of PLWH")
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

    ########################################################################################################
    # Suppliment Figure 2
    # calculate group prevalence
    group_prevalence = calc_percentage(control_bmi_int_cascade, "bmiInt_ineligible_dm")
    group_prevalence["dm_per_1000"] = (group_prevalence["n"] / 100) * 1000
    group_prevalence["group"] = group_prevalence["group"].map(group_title_dict)

    # Graph Median Prevalence of DM by group
    bar_ax = sns.barplot(
        x=group_prevalence["group"],
        y=group_prevalence["dm_per_1000"],
        estimator="median",
        palette=palette,
        hue=group_prevalence["group"],
        errorbar=("pi", 95),
        order=group_order,
        hue_order=group_order,
    )

    bar_ax.tick_params(axis="x", rotation=90)

    bar_ax.set_xlabel("")
    bar_ax.set_ylabel(
        "Prevalence of Preexisting DM Diagnosis at ART Initiation \n(per 1,000 persons)"
    )
    bar_ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))

    bar_fig = bar_ax.get_figure()
    bar_fig.savefig(out_dir / "figS2.png", bbox_inches="tight")
    plt.show()
    plt.clf()

    df = group_prevalence.groupby('group')[['dm_per_1000']].quantile([0.025,0.5,0.975]).unstack().reset_index()
    df.columns = ['group',0.025, 0.5, 0.975]
    df['formatted'] = df.apply(
        lambda row: '{:.0f} [{:.0f} - {:.0f}]'.format(row[0.50], row[0.025], row[0.975]), axis=1
    )
    df = rearrange_group_order(df)
    df.to_csv(out_dir/'figureS2_table.csv')
    df_summary_dict['Control|Prevalence of Preexisting DM Diagnosis at ART Initiation (per 1,000 persons)'] = df['formatted']
    ########################################################################################################