import math
from typing import Any, List

import colorcet as cc
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import seaborn as sns

NA_ACCORD_group_title_dict = {
    "MSM_White_Men": "White MSM",
    "MSM_Black_Men": "Black MSM",
    "MSM_Hisp_Men": "Hispanic MSM",
    "IDU_White_Men": "White MWID",
    "IDU_Black_Men": "Black MWID",
    "IDU_Hisp_Men": "Hispanic MWID",
    "IDU_White_Women": "White WWID",
    "IDU_Black_Women": "Black WWID",
    "IDU_Hisp_Women": "Hispanic WWID",
    "HET_White_Men": "White HET Men",
    "HET_Black_Men": "Black HET Men",
    "HET_Hisp_Men": "Hispanic HET Men",
    "HET_White_Women": "White HET Women",
    "HET_Black_Women": "Black HET Women",
    "HET_Hisp_Women": "Hispanic HET Women",
    "OVERALL": "Overall",
}

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

group_order_with_sub_total = [
    "Black HET Women",
    "White HET Women",
    "Hispanic HET Women",
    "All HET Women",
    "Black HET Men",
    "White HET Men",
    "Hispanic HET Men",
    "All HET Men",
    "Black WWID",
    "White WWID",
    "Hispanic WWID",
    "All WWID",
    "Black MWID",
    "White MWID",
    "Hispanic MWID",
    "All MWID",
    "Black MSM",
    "White MSM",
    "Hispanic MSM",
    "All MSM",
    "Overall"
]

all_group_names_ov = [
    "msm_white_male",
    "msm_black_male",
    "msm_hisp_male",
    "idu_white_male",
    "idu_black_male",
    "idu_hisp_male",
    "idu_white_female",
    "idu_black_female",
    "idu_hisp_female",
    "het_white_male",
    "het_black_male",
    "het_hisp_male",
    "het_white_female",
    "het_black_female",
    "het_hisp_female",
    "overall",
]

all_group_titles_ov = [
    "White MSM",
    "Black MSM",
    "Hispanic MSM",
    "White MWID",
    "Black MWID",
    "Hispanic MWID",
    "White WWID",
    "Black WWID",
    "Hispanic WWID",
    "White HET Men",
    "Black HET Men",
    "Hispanic HET Men",
    "White HET Women",
    "Black HET Women",
    "Hispanic HET Women",
    "Overall",
]

# group_title_dict = dict(zip(all_group_names_ov, all_group_titles_ov))

group_title_dict = {'msm_white_male': 'White MSM',
    'msm_black_male': 'Black MSM',
    'msm_hisp_male': 'Hispanic MSM',
    'msm_male': 'All MSM',
    'idu_white_male': 'White MWID',
    'idu_black_male': 'Black MWID',
    'idu_hisp_male': 'Hispanic MWID',
    'idu_male': 'All MWID',
    'idu_white_female': 'White WWID',
    'idu_black_female': 'Black WWID',
    'idu_hisp_female': 'Hispanic WWID',
    'idu_female': 'All WWID',
    'het_white_male': 'White HET Men',
    'het_black_male': 'Black HET Men',
    'het_hisp_male': 'Hispanic HET Men',
    'het_male': 'All HET Men',
    'het_white_female': 'White HET Women',
    'het_black_female': 'Black HET Women',
    'het_hisp_female': 'Hispanic HET Women',
    'het_female': 'All HET Women',
    'overall': 'Overall'}

# define color pallete
palette = sns.color_palette(cc.glasbey_light, n_colors=16)


def calc_percentage(
    df: pd.DataFrame, column_name: str, numerator: int = 1, percentage: bool = True
) -> pd.DataFrame:
    # group by group and column_name and sum over 'n'
    df_binary = (
        df.groupby(["group", "replication", column_name])["n"].sum().reset_index()
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


def round_thousand(x: float) -> float:
    return int(math.ceil(x / 100.0)) * 100 if x > 1000 else x


def create_summary_table(
    df: pd.DataFrame, name: str, precision: float = 0, percent: bool = True
) -> pd.DataFrame:
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


def add_summary(
    destination_df: pd.DataFrame, source_df: pd.DataFrame, name: str, percent: bool = True
) -> pd.DataFrame:
    # create summary table for df we want to add
    summary_table = create_summary_table(source_df, name, percent=percent)

    # merge to destination and return
    return destination_df.merge(summary_table)


def calc_percentage_and_add_summary(
    destination_df: pd.DataFrame, source_df: pd.DataFrame, name: str
) -> pd.DataFrame:
    # calculate the percentage of ineligible_dm
    percentage_df = calc_percentage(source_df, name)

    # merge to destination and return
    return add_summary(destination_df, percentage_df, name)


def clean_control(
    df: pd.DataFrame, only_eligible: bool = False, only_received: bool = False
) -> pd.DataFrame:
    # filter to only people who have initiated art from 2010 to 2017
    df_control = df[(df["h1yy"] <= 2017) & (df["h1yy"] >= 2010)]

    if only_eligible:
        # Filter for only eligible
        df_control = df_control[df_control["bmiInt_eligible"] == 1]

    if only_received:
        df_control = df_control[df_control["bmiInt_received"] == 1]

    # Add column of t_dm_after_h1yy to keep trace of years after initiation
    df_control["years_after_h1yy"] = df_control["t_dm"] - df_control["h1yy"]

    # Add column to keep trace of years after initiation till death
    df_control["time_exposure_to_risk"] = df_control["year_died"] - df_control["h1yy"]

    return df_control


def calc_overall_risk(df: pd.DataFrame, years_follow_up: int = 7) -> pd.DataFrame:
    # filter for only overall group
    df_overall = df[df["group"] == "overall"]

    # filter for only follow_up-year follow up with dm
    df_overall_follow_up = df_overall.loc[
        (df_overall["years_after_h1yy"] > 0) & (df_overall["years_after_h1yy"] <= years_follow_up)
    ]

    # group by replication and age group and sum
    df_overall_follow_up_sum = (
        df_overall_follow_up.groupby(["init_age_group", "replication"])["n"].sum().reset_index()
    )
    df_overall_follow_up_sum = df_overall_follow_up_sum.rename(columns={"n": "dm_num"})

    # now for the denominator
    # First adjust people died in the same year of art initiation
    df_overall["time_exposure_to_risk"] = df_overall["time_exposure_to_risk"].where(
        df_overall["time_exposure_to_risk"] == 0, 1
    )

    # Second adjust people survive from simulation
    df_overall["time_exposure_to_risk"] = df_overall["time_exposure_to_risk"].where(
        df_overall["time_exposure_to_risk"] < 0, years_follow_up
    )

    # Third adjust people survived through follow-up period
    df_overall["time_exposure_to_risk"] = df_overall["time_exposure_to_risk"].where(
        df_overall["time_exposure_to_risk"] > years_follow_up, years_follow_up
    )

    # Calculate person-time variable
    df_overall["person-time-contributed"] = df_overall["n"] * df_overall["time_exposure_to_risk"]

    # group by replication and age group and sum
    df_overall_follow_up_sum_total = (
        df_overall.groupby(["init_age_group", "replication"])[["person-time-contributed", "n"]]
        .sum()
        .reset_index()
    )
    df_overall_follow_up_sum_total = df_overall_follow_up_sum_total.rename(columns={"n": "num"})

    # create risk table and calculate risk
    dm_risk_table = dd.merge(  # type: ignore [attr-defined]
        df_overall_follow_up_sum, df_overall_follow_up_sum_total, how="left"
    )
    dm_risk_table["risk"] = dm_risk_table["dm_num"] / dm_risk_table["person-time-contributed"]

    dm_risk_table["risk"] = dm_risk_table["risk"] * 1000

    return dm_risk_table


def calc_risk_by_group(df: pd.DataFrame, years_follow_up: int) -> pd.DataFrame:
    # filter for only x-year follow up with dm
    df_follow_up = df.loc[
        (df["years_after_h1yy"] > 0) & (df["years_after_h1yy"] <= years_follow_up)
    ]

    # group by replication and group and sum
    df_follow_up_sum = df_follow_up.groupby(["group", "replication"])["n"].sum().reset_index()
    df_follow_up_sum = df_follow_up_sum.rename(columns={"n": "dm_num"})

    # now for the denominator
    # First adjust people died in the same year of art initiation
    df["time_exposure_to_risk"] = df["time_exposure_to_risk"].where(
        df["time_exposure_to_risk"] == 0, 1
    )

    # Second adjust people survive from simulation
    df["time_exposure_to_risk"] = df["time_exposure_to_risk"].where(
        df["time_exposure_to_risk"] < 0, years_follow_up
    )

    # Third adjust people survived through follow-up period
    df["time_exposure_to_risk"] = df["time_exposure_to_risk"].where(
        df["time_exposure_to_risk"] > years_follow_up, years_follow_up
    )

    # Calculate person-time variable
    df["person-time-contributed"] = df["n"] * df["time_exposure_to_risk"]

    # group by replication and group and sum
    df_all_sum = (
        df.groupby(["group", "replication"])[["person-time-contributed", "n"]].sum().reset_index()
    )
    df_all_sum = df_all_sum.rename(columns={"n": "num"})

    # merge dataframes
    group_dm_risk_table = dd.merge(df_follow_up_sum, df_all_sum, how="left")  # type: ignore [attr-defined]

    # calculate risk
    group_dm_risk_table["risk"] = (
        group_dm_risk_table["dm_num"] / group_dm_risk_table["person-time-contributed"]
    )

    group_dm_risk_table["risk"] = group_dm_risk_table["risk"] * 1000

    # return group_dm_risk_table
    return group_dm_risk_table


def calc_dm_prop(df: pd.DataFrame, death_df: pd.DataFrame) -> pd.DataFrame:
    dm_prop_df = pd.DataFrame()

    for i in range(df["replication"].max() + 1):
        for group in df.group.unique():
            # calcualte proportion of dm in each year after art initiation
            rep_df = df[(df.replication == i) & (df.group == group)].copy().reset_index(drop=True)
            rep_death_df = (
                death_df[(death_df.replication == i) & (death_df.group == group)]
                .copy()
                .reset_index(drop=True)
            )

            # Get the total population
            rep_df["total_pop"] = rep_df["n"].sum()

            # Exclude people didn't develop DM duirng follow up years at all, with negative value
            # of 'years_after_h1yy'
            rep_df = rep_df[rep_df["years_after_h1yy"] > 0].reset_index(drop=True)

            # Exclude people survived till simulation period
            rep_death_df = rep_death_df[rep_death_df["time_exposure_to_risk"] > 0].reset_index(
                drop=True
            )

            # Get the eligible population size who can develop DM
            rep_df["eligible_pop"] = (rep_df["total_pop"] - rep_df["n"].cumsum().shift(1)).fillna(
                rep_df.loc[0, "total_pop"]
            )
            rep_df["eligible_pop"] = rep_df["total_pop"] - rep_death_df["n"]

            # Calculate Proportion of each year and cumulative rate

            rep_df["proportion"] = rep_df["n"] / rep_df["eligible_pop"]

            dm_prop_df = pd.concat([dm_prop_df, rep_df], axis=0, ignore_index=True)

    return dm_prop_df


def plot_dm_prop(df_list: List[pd.DataFrame], year_period: int = 15) -> Any:
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

    # Adjust rect to leave space for the labels

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.subplots_adjust(
        left=0.05, bottom=0.05, right=0.95, top=0.95
    )  # Adjust the subplots to leave space for the labels

    fig.supxlabel("Years Since ART Initiation", y=-0.02, fontsize=16)
    fig.supylabel("Risk of New DM Diagnosis", x=-0.02, fontsize=16)

    return fig


def rearrange_group_order(df: pd.DataFrame) -> pd.DataFrame:
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

def add_sub_total(df, groupby=None):
    if 'index' in df.columns:
        df = df.drop(columns = ['index'])
    
    if groupby is None:
        groupby = ['replication']
    
    for risk_group in ['het_female','het_male','msm_male','idu_male','idu_female']:
        
        df_tmp = df[df.group.isin([f"{risk_group.split('_')[0]}_black_{risk_group.split('_')[1]}",f"{risk_group.split('_')[0]}_hisp_{risk_group.split('_')[1]}", f"{risk_group.split('_')[0]}_white_{risk_group.split('_')[1]}"] ) ].groupby(groupby)[['n']].sum().reset_index()
        df_tmp['group'] = risk_group
        df_tmp = df_tmp[df.columns]
        df = pd.concat([df, df_tmp], axis = 0).reset_index(drop=True)

    return df
