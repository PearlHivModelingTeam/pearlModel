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

    # Now we will work on the remaining percentage columns
    bmi_int_cascade = dd.read_parquet(baseline_dir / "bmi_int_cascade.parquet").reset_index()

    # filter for only starting h1yy after 2013 and before 2017
    control_bmi_int_cascade = bmi_int_cascade.loc[
        (bmi_int_cascade["h1yy"] >= start_year) & (bmi_int_cascade["h1yy"] <= 2017)
    ].compute()
    
    # Add subtotal for each risk category
    control_bmi_int_cascade = add_sub_total(control_bmi_int_cascade, groupby = ['bmiInt_scenario', 'h1yy', 'bmiInt_ineligible_dm',
       'bmiInt_ineligible_underweight', 'bmiInt_ineligible_obese',
       'bmiInt_eligible', 'bmiInt_received', 'bmi_increase_postART',
       'bmi_increase_postART_over5p', 'become_obese_postART',
       'bmiInt_impacted', 'replication'])

    # First lets get the "Number initiating ART (2013-2017)"
    new_init_age = dd.read_parquet(baseline_dir / "new_init_age.parquet").reset_index()

    # Add Overall
    new_init_age_overall = new_init_age.groupby(["replication", "year", "age"]).sum().reset_index()
    new_init_age_overall["group"] = "overall"

    new_init_age = dd.concat([new_init_age, new_init_age_overall])

    # filter for only years <= 2017 and years >= 2013
    control_new_init_age = new_init_age.loc[
        (new_init_age["year"] <= 2017) & (new_init_age["year"] >= 2013)
    ]

    # group by group and sum to get all initiating art in 2013-2017
    control_new_init_age_total_sum = control_new_init_age.groupby(['group', 'replication'])['n'].sum().reset_index().compute()
    
    # Add sub total
    control_new_init_age_total_sum = add_sub_total(control_new_init_age_total_sum)

    final_table = create_summary_table(control_new_init_age_total_sum, 'Population Size', percent=False)
    

    # We group by group, and age, but istead of taking the median, we sum over all 'n'
    control_new_init_age_simulation_sum = (
        control_new_init_age.groupby(["group", "age"])["n"].sum().reset_index().compute()
    )
    
    # Add subtotal
    control_new_init_age_simulation_sum = add_sub_total(control_new_init_age_simulation_sum, groupby = ['age'])
    
    # loop over each group and calculate the median age of initiation
    control_median_age_of_init_by_group = {}

    # Normally loopin in pandas is a bad idea, but this method was suggested by a number of communities for the weighted median problem, as it is not supported natively
    for group in control_new_init_age_simulation_sum["group"].unique():
        temp_df = control_new_init_age_simulation_sum[
            control_new_init_age_simulation_sum["group"] == group
        ]
        group_quantile = temp_df["age"].repeat(temp_df["n"]).quantile([0.025, 0.5, 0.975])
        control_median_age_of_init_by_group[group] = (
            f"{int(group_quantile[0.5])} [{int(group_quantile[0.025])} - {int(group_quantile[0.975])}]"
        )

    # convert to pandas dataframe
    control_median_age_of_init_by_group = pd.DataFrame(
        control_median_age_of_init_by_group.items(), columns=["group", "age_of_init"]
    )

    # join with our final table
    final_table = final_table.join(
        control_median_age_of_init_by_group.set_index("group"), on="group"
    )
    
    
    final_table = calc_percentage_and_add_summary(
        final_table, control_bmi_int_cascade, "bmiInt_ineligible_dm"
    )

    
    to_add = [
        "bmiInt_ineligible_dm",
        "bmiInt_ineligible_underweight",
        "bmiInt_ineligible_obese",
        "bmiInt_eligible",
    ]

    # Calculate percentages for to_add
    for column in to_add:
        final_table = calc_percentage_and_add_summary(final_table, control_bmi_int_cascade, column)

    # calculate those with normal bmi or over weight bmi by taking the complement of (underweight + overweight)
    under_plus_over = (
        calc_percentage(
            control_bmi_int_cascade, "bmiInt_ineligible_underweight", numerator=1, percentage=True
        )["n"]
        + calc_percentage(
            control_bmi_int_cascade, "bmiInt_ineligible_obese", numerator=1, percentage=True
        )["n"]
    )
    control_bmi_int_cascade_dm_eligibility_normal_ratio = 100 - under_plus_over

    control_bmi_int_cascade_dm_eligibility_normal_ratio = pd.DataFrame(
        control_bmi_int_cascade_dm_eligibility_normal_ratio
    )
    control_bmi_int_cascade_dm_eligibility_normal_ratio["group"] = calc_percentage(
        control_bmi_int_cascade, "bmiInt_ineligible_underweight", numerator=1, percentage=True
    )["group"]

    # merge into final table
    final_table = add_summary(
        final_table, control_bmi_int_cascade_dm_eligibility_normal_ratio, "normal_over_weight"
    )

    
    # calulate the number of people eligibly for intervention
    control_bmi_int_cascade_eligible_population = control_bmi_int_cascade.loc[
        control_bmi_int_cascade["bmiInt_eligible"] == 1
    ]

    # calculate the Number of ART initiators eligible for the intervention
    control_bmi_int_cascade_eligible_population_sum = (
        control_bmi_int_cascade_eligible_population.groupby(["group", "replication"])["n"]
        .sum()
        .reset_index()
    )

    # the above does not have the overall data, so we create it here
    control_bmi_int_cascade_eligible_population_sum_overall = (
        control_bmi_int_cascade_eligible_population_sum[~control_bmi_int_cascade_eligible_population_sum['group'].isin(['het_female','het_male','msm_male','idu_male','idu_female'])].groupby(["replication"])["n"]
        .sum()
        .reset_index()
    )
    control_bmi_int_cascade_eligible_population_sum_overall["group"] = "overall"

    # concat the overall with the subgroup data
    control_bmi_int_cascade_eligible_population_sum_complete = pd.concat(
        [
            control_bmi_int_cascade_eligible_population_sum,
            control_bmi_int_cascade_eligible_population_sum_overall,
        ]
    ).reset_index(drop=True)

    final_table = add_summary(
        final_table, control_bmi_int_cascade_eligible_population_sum_complete, "n", percent=False
    )

    final_table = final_table.rename(
        columns={"n": "Number of ART initiators eligible for the intervention"}
    )

    to_add_2 = ["bmi_increase_postART", "become_obese_postART"]

    # For the final 2 columns, we will condition our analysis on only those who are eligible for an intervention
    control_bmi_int_cascade_eligible = control_bmi_int_cascade.loc[
        control_bmi_int_cascade["bmiInt_eligible"] == 1
    ]

    for column in to_add_2:
        final_table = calc_percentage_and_add_summary(
            final_table, control_bmi_int_cascade_eligible, column
        )

    # map group to semantic values
    final_table["group"] = final_table["group"].map(group_title_dict)

    # clean up column names
    final_table = final_table.rename(
        columns={
            "group": "Group",
            "age_of_init": "Median age at ART initiation",
            "bmiInt_ineligible_dm": "Population percentage with pre-existing DM diagnosis at ART initiation",
            "bmiInt_ineligible_underweight": "Population percentage with BMI <18.5 at ART initiation ",
            "bmiInt_ineligible_obese": "Population percentage with BMI >=30 at ART initiation",
            "normal_over_weight": "Population percentage with 18.5<= BMI <30 at ART initiation",
            "bmiInt_eligible": "Population percentage of ART initiators eligible for the intervention",
            "bmi_increase_postART": "Percentage of eligible population experiencing weight gain over 2-years post ART initiation",
            "become_obese_postART": "Percentage of eligible population developing obese BMI (>=30) over 2-years post ART initiation",
        }
    )

    # rearrange columns for presentation
    final_table = final_table[final_table.columns[[0, 1, 2, 3, 4, 7, 5, 6, 8, 9, 10]]]

    final_table = final_table.set_index("Group").reindex(group_order_with_sub_total).reset_index()

    # save table to csv
    final_table.to_csv(out_dir / "table1.csv", index=False)
    print('Table 1 Finished.')


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

    pop_fig = pop_ax.get_figure()
    pop_fig.savefig(out_dir / "fig2a.png", bbox_inches="tight")
    # clear the plot
    plt.show()
    plt.clf()

    # Fig2B
    bar_ax = sns.barplot(
        x=dm_risk_table["init_age_group"],
        y=dm_risk_table["risk"],
        estimator="median",
        color="steelblue",
        errorbar=("pi", 95),
    )

    rounded_vals = [np.round(x,1) for x in bar_ax.containers[0].datavalues]

    bar_ax.bar_label(bar_ax.containers[0], labels=rounded_vals, padding=5)

    bar_ax.set_ylabel("7-year Risk of DM Diagnosis after ART Initiation")
    bar_ax.set_xlabel("Age Group at ART Initiation")
    bar_ax.set_xticks(range(0, 7))
    bar_ax.set_xticklabels(["<20", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"])
    bar_ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))

    pop_fig = bar_ax.get_figure()
    pop_fig.savefig(out_dir / "fig2b.png", bbox_inches="tight")
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

    final_df.to_csv(out_dir/'figure2a&b_table.csv', index = False)

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
    last_group = group_order[-1]
    secondary_ax = group_risk_ax.twinx()  # Create secondary y-axis

    # Plot the last group's data on the secondary axis
    sns.boxplot(
        x=group_dm_risk_table[group_dm_risk_table["group"] == last_group]["group"],
        y=group_dm_risk_table[group_dm_risk_table["group"] == last_group]["risk"],
        ax=secondary_ax,
        color="skyblue",
        showfliers=False,
        width=0.5,
        order=group_order,
    )

    # Make the secondary y-axis labels visible
    secondary_ax.set_ylabel("* Overall", fontsize=8.5)

    # Set y-labels to same font size for clarity
    secondary_ax.tick_params(axis='y', labelsize=8.5)
    group_risk_ax.tick_params(axis='y', labelsize=8.5)

    group_risk_ax.axvline(x=len(group_order)-1.5, color="red", linestyle="--")
    ########################################################

    group_risk_ax.set_xlabel("")
    group_risk_ax.set_ylabel("Risk of incident diabetes diagnosis (per 1,000 pys) among subgroups of PAH")
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
    last_group = group_order[-1]
    secondary_ax = group_risk_ax.twinx()  # Create secondary y-axis

    # Plot the last group's data on the secondary axis
    sns.boxplot(
        x=group_dm_risk_table[group_dm_risk_table["group"] == last_group]["group"],
        y=group_dm_risk_table[group_dm_risk_table["group"] == last_group]["dm_num"],
        ax=secondary_ax,
        color="skyblue",
        showfliers=False,
        width=0.5,
        order=group_order,
    )

    # Make the secondary y-axis labels visible
    secondary_ax.set_ylabel("* Overall", fontsize=8.5)

    # Set y-labels to same font size for clarity
    secondary_ax.tick_params(axis='y', labelsize=8.5)
    group_risk_ax.tick_params(axis='y', labelsize=8.5)

    group_risk_ax.axvline(x=len(group_order)-1.5, color="red", linestyle="--")
    ########################################################


    group_risk_ax.set_xlabel("")
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

    # Figure 3
    ##############################################################################################################################
    num_samples = 2000

    # we will look at the "bmi_int_dm_prev.h5" for S1
    bmi_int_dm_prev_s1 = dd.read_parquet(variable_dir / "dm_final_output.parquet").reset_index()

    bmi_int_dm_prev_s1 = bmi_int_dm_prev_s1.astype(
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

    # Add Overall
    all_but_group = list(bmi_int_dm_prev_s1.columns[1:])
    bmi_int_dm_prev_s1_overall = bmi_int_dm_prev_s1.groupby(all_but_group).sum().reset_index()
    bmi_int_dm_prev_s1_overall["group"] = "overall"
    bmi_int_dm_prev_s1 = dd.concat(
        [bmi_int_dm_prev_s1, bmi_int_dm_prev_s1_overall], ignore_index=True
    )

    # clean to control specifications
    control_bmi_int_dm_prev_s1 = clean_control(bmi_int_dm_prev_s1, only_eligible=True, only_received=True)

    # filter for only people eligible for intervention
    bmi_int_s1_eligible_risk = calc_risk_by_group(control_bmi_int_dm_prev_s1, 7)

    s1_sample = (
        bmi_int_s1_eligible_risk.groupby("group")
        .apply(lambda x: x.sample(num_samples, replace=True))
        .reset_index(drop=True)
        .compute()
    )

    del bmi_int_dm_prev_s1, bmi_int_s1_eligible_risk

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

    bmi_int_eligible_risk = calc_risk_by_group(control_bmi_int_dm_prev, 7)

    s0_sample = (
        bmi_int_eligible_risk.groupby("group")
        .apply(lambda x: x.sample(num_samples, replace=True))
        .reset_index(drop=True)
        .compute()
    )

    del bmi_int_dm_prev, bmi_int_eligible_risk

    s0_sample = s0_sample.sort_values(by="group").reset_index(drop=True)
    s1_sample = s1_sample.sort_values(by="group").reset_index(drop=True)
    #######################################################################################################
    # Fig 3A
    # absolute difference
    abs_sample_diff = s1_sample[["dm_num", "risk"]] - s0_sample[["dm_num", "risk"]]
    abs_sample_diff["group"] = s0_sample["group"]
    abs_sample_diff["num"] = s0_sample["num"]

    abs_sample_diff_plot = abs_sample_diff.copy()
    abs_sample_diff_plot["group"] = abs_sample_diff_plot["group"].map(group_title_dict)

    diff_ax = sns.boxplot(
        x=abs_sample_diff_plot["group"],
        y=-abs_sample_diff_plot["risk"],
        color="seagreen",
        showfliers=False,
        palette=palette,
        hue=abs_sample_diff_plot["group"],
        order=group_order[:-1],
        hue_order=group_order,
    )

    diff_ax.tick_params(axis="x", rotation=90)

    ########################################################
    last_group = group_order[-1]
    secondary_ax = diff_ax.twinx()  # Create secondary y-axis

    # Plot the last group's data on the secondary axis
    sns.boxplot(
        x=abs_sample_diff_plot[abs_sample_diff_plot["group"] == last_group]["group"],
        y=-abs_sample_diff_plot[abs_sample_diff_plot["group"] == last_group]["risk"],
        ax=secondary_ax,
        color="skyblue",
        showfliers=False,
        width=0.5,
        order=group_order,
    )

    # Make the secondary y-axis labels visible
    secondary_ax.set_ylabel("* Overall", fontsize=8.5)

    # Set y-labels to same font size for clarity
    secondary_ax.tick_params(axis='y', labelsize=8.5)
    diff_ax.tick_params(axis='y', labelsize=8.5)

    diff_ax.axvline(x=len(group_order)-1.5, color="red", linestyle="--")
    ########################################################

    diff_ax.set_xlabel("")
    diff_ax.set_ylabel(
        "Absolute percentage point risk reduction in incident diabetes\n diagnosis (per 1,000 pys) with vs. without the intervention",
        fontsize=8.5,
    )

    diff_ax.axhline(y=0, color="r", linestyle="-")

    diff_fig = diff_ax.get_figure()
    diff_fig.savefig(out_dir / "fig3a.png", bbox_inches="tight")
    plt.show()
    plt.clf()

    abs_sample_diff_plot["risk"] = -abs_sample_diff_plot["risk"]
    df = (
        abs_sample_diff_plot.groupby("group")[["risk"]]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
    )
    df.columns = ["group", 0.025, 0.5, 0.975]
    df["formatted"] = df.apply(
        lambda row: f"{row[0.50]:.1f} [{row[0.025]:.1f} - {row[0.975]:.1f}]", axis=1
    )
    df = rearrange_group_order(df)
    df.to_csv(out_dir / "figure3a_table.csv")

    # 3b
    # relative difference
    rel_sample_diff = -(s1_sample[["risk"]] - s0_sample[["risk"]]) / s0_sample[["risk"]]
    rel_sample_diff["group"] = s0_sample["group"]

    rel_sample_diff_plot = rel_sample_diff.copy()
    rel_sample_diff_plot["group"] = rel_sample_diff_plot["group"].map(group_title_dict)

    rel_ax = sns.boxplot(
        x=rel_sample_diff_plot["group"],
        y=rel_sample_diff_plot["risk"]*100,
        color="seagreen",
        showfliers=False,
        palette=palette,
        hue=rel_sample_diff_plot["group"],
        order=group_order,
        hue_order=group_order[:-1],
    )

    rel_ax.tick_params(axis="x", rotation=90)

    ########################################################
    #Plot Overall Group
    last_group = group_order[-1]
    secondary_ax = rel_ax.twinx()  # Create secondary y-axis

    # Plot the last group's data on the secondary axis
    sns.boxplot(
        x=rel_sample_diff_plot[rel_sample_diff_plot["group"] == last_group]["group"],
        y=-rel_sample_diff_plot[rel_sample_diff_plot["group"] == last_group]["risk"]*100,
        ax=secondary_ax,
        color="skyblue",
        showfliers=False,
        width=0.5,
        order=group_order,
    )

    # Make the secondary y-axis labels visible
    secondary_ax.set_ylabel("* Overall", fontsize=8.5)

    # Set y-labels to same font size for clarity
    secondary_ax.tick_params(axis='y', labelsize=8.5)
    rel_ax.tick_params(axis='y', labelsize=8.5)

    # add dashed vertical line to seperate overall group
    rel_ax.axvline(x=len(group_order)-1.5, color="red", linestyle="--")
    ########################################################

    rel_ax.set_xlabel("")
    rel_ax.set_ylabel(
        "%Relative risk reduction in incident DM diagnoses \n with (vs. without) the intervention",
        fontsize=8.5,
    )
    rel_ax.axhline(y=0, color="r", linestyle="-")
    rel_fig = rel_ax.get_figure()
    rel_fig.savefig(out_dir / "fig3b.png", bbox_inches="tight")
    plt.show()
    plt.clf()

    df = (
        rel_sample_diff_plot.groupby("group")[["risk"]]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
    )
    df.columns = ["group", 0.025, 0.5, 0.975]
    df["formatted"] = df.apply(
        lambda row: f"{row[0.50]:.3f} [{row[0.025]:.3f} - {row[0.975]:.3f}]", axis=1
    )
    df = rearrange_group_order(df)
    df.to_csv(out_dir / "figure3b_table.csv")

    #######################################################################################################
    # 3c
    abs_sample_diff_plot["dm_per_1000"] = abs_sample_diff_plot["risk"] * (-1000)
    abs_sample_diff_plot["NNT"] = -np.round(
        abs_sample_diff_plot["num"] / abs_sample_diff_plot["dm_num"], 0
    )

    dm_per_1000_ax = sns.boxplot(
        x=abs_sample_diff_plot["group"],
        y=abs_sample_diff_plot["NNT"],
        color="seagreen",
        showfliers=False,
        palette=palette,
        hue=abs_sample_diff_plot["group"],
        order=group_order[:-1],
        hue_order=group_order,
    )

    dm_per_1000_ax.tick_params(axis="x", rotation=90)

    ########################################################
    #Plot Overall Group
    last_group = group_order[-1]
    secondary_ax = dm_per_1000_ax.twinx()  # Create secondary y-axis

    # Plot the last group's data on the secondary axis
    sns.boxplot(
        x=abs_sample_diff_plot[abs_sample_diff_plot["group"] == last_group]["group"],
        y=abs_sample_diff_plot[abs_sample_diff_plot["group"] == last_group]["NNT"],
        ax=secondary_ax,
        color="skyblue",
        showfliers=False,
        width=0.5,
        order=group_order,
    )

    # Make the secondary y-axis labels visible
    secondary_ax.set_ylabel("* Overall", fontsize=8.5)

    # Set y-labels to same font size for clarity
    secondary_ax.tick_params(axis='y', labelsize=8.5)
    dm_per_1000_ax.tick_params(axis='y', labelsize=8.5)

    # add dashed vertical line to seperate overall group
    dm_per_1000_ax.axvline(x=len(group_order)-1.5, color="red", linestyle="--")
    ########################################################

    dm_per_1000_ax.set_xlabel("")
    dm_per_1000_ax.set_ylabel(
        "Number of people who must experience the intervention\n (NNT) to avert 1 incident DM diagnosis",
        fontsize=8,
    )
    dm_per_1000_ax.axhline(y=0, color="r", linestyle="-")
    dm_per_1000_fig = dm_per_1000_ax.get_figure()
    dm_per_1000_fig.savefig(out_dir / "fig3c.png", bbox_inches="tight")
    plt.show()
    plt.clf()

    df = (
        abs_sample_diff_plot.groupby("group")[["NNT"]]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
    )
    df.columns = ["group", 0.025, 0.5, 0.975]
    df["formatted"] = df.apply(
        lambda row: f"{row[0.50]:.0f} [{row[0.025]:.0f} - {row[0.975]:.0f}]", axis=1
    )
    df = rearrange_group_order(df)
    df.to_csv(out_dir / "figure3c_table.csv")

    #######################################################################################################
    # 3d
    abs_sample_diff_plot["dm_num_prevented"] = abs_sample_diff_plot["dm_num"] * -1
    dm_prevented_ax = sns.boxplot(
        x=abs_sample_diff_plot["group"],
        y=abs_sample_diff_plot["dm_num_prevented"],
        color="seagreen",
        showfliers=False,
        palette=palette,
        hue=abs_sample_diff_plot["group"],
        order=group_order[:-1],
        hue_order=group_order,
    )

    dm_prevented_ax.tick_params(axis="x", rotation=90)

    ########################################################
    #Plot Overall Group
    last_group = group_order[-1]
    secondary_ax = dm_prevented_ax.twinx()  # Create secondary y-axis

    # Plot the last group's data on the secondary axis
    sns.boxplot(
        x=abs_sample_diff_plot[abs_sample_diff_plot["group"] == last_group]["group"],
        y=abs_sample_diff_plot[abs_sample_diff_plot["group"] == last_group]["dm_num_prevented"],
        ax=secondary_ax,
        color="skyblue",
        showfliers=False,
        width=0.5,
        order=group_order,
    )

    # Make the secondary y-axis labels visible
    secondary_ax.set_ylabel("* Overall", fontsize=8.5)

    # Set y-labels to same font size for clarity
    secondary_ax.tick_params(axis='y', labelsize=8.5)
    dm_prevented_ax.tick_params(axis='y', labelsize=8.5)

    # add dashed vertical line to seperate overall group
    dm_prevented_ax.axvline(x=len(group_order)-1.5, color="red", linestyle="--")
    ########################################################

    dm_prevented_ax.set_xlabel("")
    dm_prevented_ax.set_ylabel(
        "Number of incident DM diagnoses averted\n with (vs. without) the intervention",
        fontsize=8.5,
    )
    dm_prevented_ax.axhline(y=0, color="r", linestyle="-")
    dm_prevented_fig = dm_prevented_ax.get_figure()
    dm_prevented_fig.savefig(out_dir / "fig3d.png", bbox_inches="tight")
    plt.show()
    plt.clf()

    df = (
        abs_sample_diff_plot.groupby("group")[["dm_num_prevented"]]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
    )
    df.columns = ["group", 0.025, 0.5, 0.975]
    df["formatted"] = df.apply(
        lambda row: f"{row[0.50]:.0f} [{row[0.025]:.0f} - {row[0.975]:.0f}]", axis=1
    )
    df = rearrange_group_order(df)
    df.to_csv(out_dir / "figure3d_table.csv")

    ############################################################################################################################
    # # Supplimental Figure 1
    # # we will look at the "bmi_int_dm_prev.h5" for S0
    # bmi_int_dm_prev = dd.read_parquet(baseline_dir /'dm_final_output.parquet').reset_index()

    # # Add Overall
    # all_but_group = list(bmi_int_dm_prev.columns[1:])
    # bmi_int_dm_prev_overall = bmi_int_dm_prev.groupby(all_but_group).sum().reset_index()
    # bmi_int_dm_prev_overall['group'] = 'overall'
    # bmi_int_dm_prev = dd.concat([bmi_int_dm_prev, bmi_int_dm_prev_overall], ignore_index=True)

    # # type the dataframe for space efficiency
    # bmi_int_dm_prev = bmi_int_dm_prev.astype({'group':'str', 'replication':'int16', 'bmiInt_scenario':np.int8, 'h1yy': np.int16, 'bmiInt_impacted':bool, 'dm': bool, 't_dm': np.int16, 'n': np.int16})

    # # clean to control specifications
    # control_bmi_int_dm_prev = clean_control(bmi_int_dm_prev, only_eligible=False)

    # # sum across replications, group, and years_after_h1yy
    # control_bmi_int_dm_prev_agg = control_bmi_int_dm_prev.groupby(['group', 'years_after_h1yy','replication'])['n'].sum().reset_index().compute()
    # death_df = control_bmi_int_dm_prev.groupby(['group', 'time_exposure_to_risk','replication'])['n'].sum().reset_index().compute()

    # dm_prop_df = calc_dm_prop(control_bmi_int_dm_prev_agg, death_df)
    # dm_prop_df['group'] = dm_prop_df['group'].map(group_title_dict)

    # # Plot Set Up
    # year_period = 7

    # plot_groups_reordered = ['Black HET Women', 'White HET Women', 'Hispanic HET Women',
    #                     'Black HET Men', 'White HET Men', 'Hispanic HET Men',
    #                     'Black WWID', 'White WWID', 'Hispanic WWID',
    #                     'Black MWID', 'White MWID', 'Hispanic MWID',
    #                     'Black MSM', 'White MSM', 'Hispanic MSM']

    # colors = ['r', 'b']

    # column_names = ['Black', 'White', 'Hispanic']

    # row_names = ['HET Women', 'HET Men', 'WWID', 'MWID', 'MSM']

    # # Plotting Codes
    # fig, axs = plt.subplots(5, 3, figsize=(16, 12))

    # plot_groups = np.sort(dm_prop_df.group.unique())
    # plot_groups = plot_groups[plot_groups != 'Overall']

    # for i, group in enumerate(plot_groups_reordered):

    #     group_df = dm_prop_df[dm_prop_df.group == group]
    #     real_group_df = real_DM_df[real_DM_df.group == group]
    #     ax = axs.flatten()[i]

    #     if i < 3:
    #         ax.set_title(column_names[i], fontweight='bold')

    #     if i % 3 == 0:
    #         k = i // 3
    #         ax.set_ylabel(row_names[k], fontweight='bold')

    #     # Plot PEARL Projection
    #     percentiles = group_df.groupby('years_after_h1yy')['proportion'].quantile([0.025, 0.5, 0.975]).unstack()
    #     # print(percentiles)
    #     ax.plot(percentiles.index[:year_period],
    #             percentiles.loc[1:year_period,0.50],
    #             marker='o',
    #             linestyle='-',
    #             color='r',
    #             label='PEARL Projection Median Probability')

    #     # ax.fill_between(percentiles.index[:year_period],
    #     #                 percentiles.loc[1:year_period, 0.025],
    #     #                 percentiles.loc[1:year_period,0.975],
    #     #                 color='r',
    #     #                 alpha=0.4,
    #     #                 label='95% Uncertainty Range')

    #     # Plot NA-ACCORD real Data
    #     ax.plot(real_group_df['t'], real_group_df['prop'], marker='o', linestyle='-', color='b', label='NA-ACCORD')
    #     ax.fill_between(real_group_df['t'], real_group_df['prop_lower_ci'], real_group_df['prop_upper_ci'], color='b', alpha=0.1, label='NA-ACCORD 95% Uncertainty Range')

    #     ax.set_xticks(range(1,year_period+1))

    # plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust rect to leave space for the labels
    # plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)  # Adjust the subplots to leave space for the labels

    # fig.supxlabel('Years Since ART Initiation', y = -0.02,fontsize=16)
    # fig.supylabel('Annual risk of new DM diagnosis', x=-0.02,fontsize=16)

    # fig.savefig(out_dir/"figS1.png",bbox_inches='tight')
