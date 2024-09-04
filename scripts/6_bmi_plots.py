# declare imports
import argparse
from datetime import datetime
import math
from pathlib import Path

import colorcet as cc
import dask.dataframe as dd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pearllib import NA_ACCORD_group_title_dict, group_title_dict
from scipy.stats import median_abs_deviation
import seaborn as sns

group_order = [
    "Black HET Women",
    "White HET Women",
    "Hispanic HET Women",
    "Black HET Men",
    "White HET Men",
    "Hispanic HET Men",
    "Black WWID",
    "White WWID",
    "Hispanic WWID",
    "Black MWID",
    "White MWID",
    "Hispanic MWID",
    "Black MSM",
    "White MSM",
    "Hispanic MSM",
    "Overall",
]

# define color pallete
palette = sns.color_palette(cc.glasbey_light, n_colors=16)


def calc_percentage(df, column_name, numerator=1, percentage=True):
    # group by group and column_name and sum over 'n'
    df_binary = (
        df.groupby(["group", "replication", column_name])["n"].sum().reset_index().compute()
    )

    # the above does not have the overall data, so we create it here
    overall = df_binary.groupby(["replication", column_name])["n"].sum().reset_index()
    overall["group"] = "overall"

    # concat the overall with the subgroup data
    df_binary_complete = pd.concat([df_binary, overall]).reset_index(drop=True)

    # calculate the ratio of column_name with numerator value over sum
    df_ratio = (
        df_binary_complete.loc[df_binary_complete[column_name] == numerator].reset_index()["n"]
        / df_binary_complete.groupby(["group", "replication"]).sum().reset_index()["n"]
    )
    df_ratio = pd.DataFrame(df_ratio)
    # add back the group column that is lost in the above calculations
    df_ratio["group"] = df_binary_complete.loc[
        df_binary_complete[column_name] == numerator
    ].reset_index()["group"]

    if percentage:
        df_ratio["n"] = df_ratio["n"] * 100

    return df_ratio


def round_thousand(x):
    return int(math.ceil(x / 100.0)) * 100 if x > 1000 else x


def create_summary_table(df, name, precision=0, percent=True):
    df_quantile = df.groupby("group")["n"].quantile([0.05, 0.5, 0.95]).unstack().reset_index()

    if precision == 0:
        df_quantile = df_quantile.apply(lambda x: x.astype(int) if is_numeric_dtype(x) else x)
    else:
        df_quantile = df_quantile.round(precision)

    df_quantile[0.05] = df_quantile[0.05].apply(round_thousand)
    df_quantile[0.5] = df_quantile[0.5].apply(round_thousand)
    df_quantile[0.95] = df_quantile[0.95].apply(round_thousand)

    f_string = "{}% [{} - {}%]" if percent else "{} [{} - {}]"

    df_quantile[name] = df_quantile.apply(
        lambda x: f_string.format(x[0.5], x[0.05], x[0.95]), axis=1
    )

    return df_quantile[["group", name]]


def add_summary(destination_df, source_df, name, percent=True):
    # create summary table for df we want to add
    summary_table = create_summary_table(source_df, name, percent=percent)

    # merge to destination and return
    return destination_df.merge(summary_table)


def calc_percentage_and_add_summary(destination_df, source_df, name):
    # calculate the percentage of ineligible_dm
    percentage_df = calc_percentage(source_df, name)

    # merge to destination and return
    return add_summary(destination_df, percentage_df, name)


def clean_control(df, only_eligible=True):
    # filter to only people who have initiated art from 2010 to 2017
    df_control = df[(df["h1yy"] <= 2017) & (df["h1yy"] >= 2010)]

    if only_eligible:
        # Filter for only eligible
        df_control = df_control[df_control["bmiInt_eligible"] is True]

    # Add column of t_dm_after_h1yy to keep trace of years after initiation
    df_control["years_after_h1yy"] = df_control["t_dm"] - df_control["h1yy"]

    return df_control


def calc_overall_risk(df, follow_up=7):
    # filter for only overall group
    df_overall = df[df["group"] == "overall"]

    # filter for only follow_up-year follow up with dm
    df_overall_follow_up = df_overall.loc[
        (df_overall["years_after_h1yy"] > 0) & (df_overall["years_after_h1yy"] <= follow_up)
    ]

    # group by replication and age group and sum
    df_overall_follow_up_sum = (
        df_overall_follow_up.groupby(["init_age_group", "replication"])["n"].sum().reset_index()
    )
    df_overall_follow_up_sum = df_overall_follow_up_sum.rename(columns={"n": "dm_num"})

    # now for the denominator
    # group by replication and age group and sum
    df_overall_follow_up_sum_total = (
        df_overall.groupby(["init_age_group", "replication"])["n"].sum().reset_index()
    )
    df_overall_follow_up_sum_total = df_overall_follow_up_sum_total.rename(columns={"n": "num"})

    # create risk table and calculate risk
    dm_risk_table = dd.merge(df_overall_follow_up_sum, df_overall_follow_up_sum_total, how="left")
    dm_risk_table["risk"] = dm_risk_table["dm_num"] / dm_risk_table["num"]

    return dm_risk_table


def calc_risk_by_group(df, years_follow_up):
    # filter for only x-year follow up with dm
    df_follow_up = df.loc[
        (df["years_after_h1yy"] > 0) & (df["years_after_h1yy"] <= years_follow_up)
    ]

    # group by replication and group and sum
    df_follow_up_sum = df_follow_up.groupby(["group", "replication"])["n"].sum().reset_index()
    df_follow_up_sum = df_follow_up_sum.rename(columns={"n": "dm_num"})

    # group by replication and group and sum
    df_all_sum = df.groupby(["group", "replication"])["n"].sum().reset_index()
    df_all_sum = df_all_sum.rename(columns={"n": "num"})

    # merge dataframes
    group_dm_risk_table = dd.merge(df_follow_up_sum, df_all_sum, how="left")

    # calculate risk
    group_dm_risk_table["risk"] = group_dm_risk_table["dm_num"] / group_dm_risk_table["num"]

    return group_dm_risk_table


def calc_dm_prop(df):
    dm_prop_df = pd.DataFrame()

    for i in range(df["replication"].max() + 1):
        for group in df.group.unique():
            # calcualte proportion of dm in each year after art initiation
            rep_df = df[(df.replication == i) & (df.group == group)].copy().reset_index(drop=True)

            # Get the total population
            rep_df["total_pop"] = rep_df["n"].sum()

            # Exclude people didn't develop DM duirng follow up years at all, with negative value of 'years_after_h1yy'
            rep_df = rep_df[rep_df["years_after_h1yy"] > 0].reset_index(drop=True)

            # Get the eligible population size who can develop DM
            rep_df["eligible_pop"] = (rep_df["total_pop"] - rep_df["n"].cumsum().shift(1)).fillna(
                rep_df.loc[0, "total_pop"]
            )

            # Calculate Proportion of each year and cumulative rate

            rep_df["proportion"] = rep_df["n"] / rep_df["eligible_pop"]

            dm_prop_df = pd.concat([dm_prop_df, rep_df], axis=0, ignore_index=True)

    return dm_prop_df


def plot_dm_prop(df_list, year_period=15):
    colors = [("r", "lightcoral"), ("b", "steelblue")]

    column_names = ["Black", "Hispanic", "White"]

    row_names = ["HET Men", "HET Women", "MSM", "MWID", "WWID"]

    fig, axs = plt.subplots(5, 3, figsize=(30, 20))

    plot_groups = np.sort(df_list[0].group.unique())
    plot_groups = plot_groups[plot_groups != "Overall"]

    for j, df in enumerate(df_list):
        for i, group in zip([0, 3, 6, 9, 12, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14], plot_groups):
            group_df = df[df.group == group]
            ax = axs.flatten()[i]

            if i < 3:
                ax.set_title(column_names[i], fontweight="bold")

            if i % 3 == 0:
                k = i // 3
                ax.set_ylabel(row_names[k], fontweight="bold")

            percentiles = (
                group_df.groupby("years_after_h1yy")["proportion"]
                .quantile([0.05, 0.5, 0.95])
                .unstack()
            )
            # print(percentiles)
            ax.plot(
                percentiles.index[: year_period - 1],
                percentiles.loc[2:year_period, 0.50],
                marker="o",
                linestyle="-",
                color=colors[j][0],
                label="Median Probability",
            )

            ax.fill_between(
                percentiles.index[: year_period - 1],
                percentiles.loc[2:year_period, 0.05],
                percentiles.loc[2:year_period, 0.95],
                color=colors[j][1],
                alpha=0.5,
                label="95% CI",
            )

            ax.set_xticks(range(2, year_period + 1))

            ax.set_ylim([0, 0.04])

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust rect to leave space for the labels
    plt.subplots_adjust(
        left=0.05, bottom=0.05, right=0.95, top=0.95
    )  # Adjust the subplots to leave space for the labels

    fig.supxlabel("Years Since ART Initiation", y=-0.02, fontsize=16)
    fig.supylabel("Risk of New DM Diagnosis", x=-0.02, fontsize=16)

    return fig


def rearrange_group_order(df):
    group_order = [
        "Black HET Women",
        "White HET Women",
        "Hispanic HET Women",
        "Black HET Men",
        "White HET Men",
        "Hispanic HET Men",
        "Black WWID",
        "White WWID",
        "Hispanic WWID",
        "Black MWID",
        "White MWID",
        "Hispanic MWID",
        "Black MSM",
        "White MSM",
        "Hispanic MSM",
        "Overall",
    ]

    # Reorder the rows based on the group_order
    df["group"] = pd.Categorical(df["group"], categories=group_order, ordered=True)
    df_sorted = df.sort_values("group").reset_index(drop=True)
    return df_sorted


if __name__ == "__main__":
    start_time = datetime.now()

    # Define the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline")
    parser.add_argument("--variable")
    args = parser.parse_args()

    baseline_dir = Path(args.baseline)
    variable_dir = Path(args.variable)

    # Now we will work on the remaining percentage columns
    bmi_int_cascade = dd.read_parquet(baseline_dir / "bmi_int_cascade.h5").reset_index()

    # filter for only starting h1yy after 2010 and before 2017
    control_bmi_int_cascade = bmi_int_cascade.loc[
        (bmi_int_cascade["h1yy"] >= 2010) & (bmi_int_cascade["h1yy"] <= 2017)
    ]

    # First lets get the "Number initiating ART (2010-2017)"
    new_init_age = dd.read_parquet(baseline_dir / "new_init_age.h5").reset_index()

    # Add Overall
    new_init_age_overall = new_init_age.groupby(["replication", "year", "age"]).sum().reset_index()
    new_init_age_overall["group"] = "overall"

    new_init_age = dd.concat([new_init_age, new_init_age_overall])

    # filter for only years <= 2017 and years >= 2010
    control_new_init_age = new_init_age.loc[
        (new_init_age["year"] <= 2017) & (new_init_age["year"] >= 2010)
    ]

    # group by group and sum to get all initiating art in 2010-2017
    control_new_init_age_total_sum = (
        control_new_init_age.groupby(["group", "replication"])["n"].sum().reset_index().compute()
    )

    final_table = create_summary_table(
        control_new_init_age_total_sum, "Population Size", percent=False
    )

    # We group by group, and age, but istead of taking the median, we sum over all 'n'
    control_new_init_age_simulation_sum = (
        control_new_init_age.groupby(["group", "age"])["n"].sum().reset_index().compute()
    )

    # loop over each group and calculate the median age of initiation
    control_median_age_of_init_by_group = {}

    # Normally loopin in pandas is a bad idea, but this method was suggested by a number of communities for the weighted median problem, as it is not supported natively
    for group in control_new_init_age_simulation_sum["group"].unique():
        temp_df = control_new_init_age_simulation_sum[
            control_new_init_age_simulation_sum["group"] == group
        ]
        group_quantile = temp_df["age"].repeat(temp_df["n"]).quantile([0.05, 0.5, 0.95])
        control_median_age_of_init_by_group[group] = (
            f"{int(group_quantile[0.5])} [{int(group_quantile[0.05])} - {int(group_quantile[0.95])}]"
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
        .compute()
    )

    # the above does not have the overall data, so we create it here
    control_bmi_int_cascade_eligible_population_sum_overall = (
        control_bmi_int_cascade_eligible_population_sum.groupby(["replication"])["n"]
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

    final_table = final_table.set_index("Group").reindex(group_order).reset_index()

    # save table to csv
    final_table.to_csv("../outputs/final_table.csv", index=False)
