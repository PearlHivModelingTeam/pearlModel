import argparse
from datetime import datetime
from pathlib import Path

from dask import dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pearl.post_processing.bmi import (
    add_overall,
    calc_risk_by_group,
    clean_control,
    group_title_dict,
)

target_columns = [
    "dm_prevalence_prev",
    "dm_prevalence",
    "dm_incidence",
    "pre_art_bmi",
    "post_art_bmi",
    "art_initiators",
]

variable_name_map = {
    "dm_prevalence_prev": "DM Prevalence at 2009",
    "dm_prevalence": "DM Prevalence 2009-end",
    "dm_incidence": "DM Incidence",
    "pre_art_bmi": "Pre ART BMI",
    "post_art_bmi": "Post ART BMI",
    "art_initiators": "# ART Initiators",
}


def add_overall_to_params(params):
    group = params["group"].unique()[1]
    overall = params[params["group"] == group]
    overall["group"] = "overall"
    params = pd.concat([params, overall])
    return params.reset_index(drop=True)


def calc_tornado_vals(
    baseline_risk_df,
    variable_risk_df,
    param_df_baseline,
    param_df_variable,
    col_name,
    num_samples=1000,
    lesser=0.1,
    greater=0.9,
):
    sub_param_baseline = param_df_baseline[["replication", "group", col_name]]
    sub_param_variable = param_df_variable[["replication", "group", col_name]]

    baseline_merged = sub_param_baseline.merge(
        baseline_risk_df, on=["replication", "group"], how="left"
    ).fillna(0)
    variable_merged = sub_param_variable.merge(
        variable_risk_df, on=["replication", "group"], how="left"
    ).fillna(0)

    quantile_val_baseline = (
        baseline_merged.groupby("group")[col_name]
        .quantile([lesser, greater])
        .unstack()
        .reset_index()
    )
    quantile_val_variable = (
        variable_merged.groupby("group")[col_name]
        .quantile([lesser, greater])
        .unstack()
        .reset_index()
    )

    difference_tornado_df = []
    relative_tornado_df = []
    baseline_tornado_df = []
    variable_tornado_df = []
    for group in quantile_val_baseline["group"].unique():
        group_df_baseline = baseline_merged[baseline_merged["group"] == group]
        group_df_variable = variable_merged[variable_merged["group"] == group]

        lesser_val_baseline = quantile_val_baseline[quantile_val_baseline["group"] == group][
            lesser
        ].values[0]
        greater_val_baseline = quantile_val_baseline[quantile_val_baseline["group"] == group][
            greater
        ].values[0]
        lesser_val_variable = quantile_val_variable[quantile_val_variable["group"] == group][
            lesser
        ].values[0]
        greater_val_variable = quantile_val_variable[quantile_val_variable["group"] == group][
            greater
        ].values[0]

        lesser_group_df_baseline = group_df_baseline[
            group_df_baseline[col_name] <= lesser_val_baseline
        ]
        greater_group_df_baseline = group_df_baseline[
            group_df_baseline[col_name] >= greater_val_baseline
        ]
        lesser_group_df_variable = group_df_variable[
            group_df_variable[col_name] <= lesser_val_variable
        ]
        greater_group_df_variable = group_df_variable[
            group_df_variable[col_name] >= greater_val_variable
        ]

        lesser_group_df_baseline_sample = lesser_group_df_baseline.sample(
            num_samples, replace=True
        ).reset_index()
        greater_group_df_baseline_sample = greater_group_df_baseline.sample(
            num_samples, replace=True
        ).reset_index()
        lesser_group_df_variable_sample = lesser_group_df_variable.sample(
            num_samples, replace=True
        ).reset_index()
        greater_group_df_variable_sample = greater_group_df_variable.sample(
            num_samples, replace=True
        ).reset_index()

        baseline_group_tornado_df = {
            "group": group,
            "variable": col_name,
            lesser: (lesser_group_df_baseline_sample["risk"]).quantile([lesser, 0.5, greater]),
            greater: (greater_group_df_baseline_sample["risk"]).quantile([lesser, 0.5, greater]),
            "lesser_count": lesser_group_df_baseline["risk"].count(),
            "greater_count": greater_group_df_baseline["risk"].count(),
        }
        baseline_tornado_df.append(baseline_group_tornado_df)
        variable_group_tornado_df = {
            "group": group,
            "variable": col_name,
            lesser: (lesser_group_df_variable_sample["risk"]).quantile([lesser, 0.5, greater]),
            greater: (greater_group_df_variable_sample["risk"]).quantile([lesser, 0.5, greater]),
            "lesser_count": lesser_group_df_variable["risk"].count(),
            "greater_count": greater_group_df_variable["risk"].count(),
        }
        variable_tornado_df.append(variable_group_tornado_df)
        difference_group_tornado_df = {
            "group": group,
            "variable": col_name,
            lesser: (
                lesser_group_df_variable_sample["risk"] - lesser_group_df_baseline_sample["risk"]
            ).quantile([lesser, 0.5, greater]),
            greater: (
                greater_group_df_variable_sample["risk"] - greater_group_df_baseline_sample["risk"]
            ).quantile([lesser, 0.5, greater]),
        }
        difference_tornado_df.append(difference_group_tornado_df)

        relative_group_tornado_df = {
            "group": group,
            "variable": col_name,
            lesser: (
                (lesser_group_df_variable_sample["risk"] - lesser_group_df_baseline_sample["risk"])
                / lesser_group_df_baseline_sample["risk"]
            ).quantile([lesser, 0.5, greater]),
            greater: (
                (
                    greater_group_df_variable_sample["risk"]
                    - greater_group_df_baseline_sample["risk"]
                )
                / lesser_group_df_baseline_sample["risk"]
            ).quantile([lesser, 0.5, greater]),
        }
        relative_tornado_df.append(relative_group_tornado_df)

    baseline_tornado_df = pd.DataFrame(baseline_tornado_df)
    variable_tornado_df = pd.DataFrame(variable_tornado_df)
    difference_tornado_df = pd.DataFrame(difference_tornado_df)
    relative_tornado_df = pd.DataFrame(relative_tornado_df)
    return baseline_tornado_df, variable_tornado_df, difference_tornado_df, relative_tornado_df


def overall_tornado_plot(tornado_vals, baseline_vals):
    # create an axis
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    group = "overall"

    group_vals = tornado_vals[tornado_vals["group"] == group].reset_index(drop=True)

    # plot parameters
    y_tick_label = group_vals["variable"]
    ys = range(len(y_tick_label))[::-1]

    color_lesser = "#0d47a1"
    color_greater = "#e2711d"
    color_line = "#2ECC71"

    # Data to be visualized
    lesser_value = group_vals.iloc[:, 2].values
    greater_value = group_vals.iloc[:, 3].values
    base = baseline_vals[baseline_vals["group"] == group]["risk"].values[0]

    # Draw bars
    for y, value, value2 in zip(ys, lesser_value, greater_value):
        # draw horizontal line markers for the uncertainty around lesser and greater
        # lesser
        ax.plot(
            (value.iloc[0], value.iloc[2]),
            (y, y),
            linewidth=2,
            color=color_lesser,
            marker="|",
            markersize=15,
        )
        ax.plot(value.iloc[1], y, "o", color=color_lesser, markersize=15)

        # greater
        ax.plot(
            (value2.iloc[0], value2.iloc[2]),
            (y, y),
            linewidth=2,
            color=color_greater,
            marker="|",
            markersize=15,
        )
        ax.plot(value2.iloc[1], y, "o", color=color_greater, markersize=15)

    # Add vertical line for median value
    ax.axvline(base, color=color_line, linewidth=2, label="Median value")

    # add vertical line for the 90% and 110% of baseline
    ax.axvline(base * 0.9, color="red", linewidth=2, label="Median value")
    ax.axvline(base * 1.1, color="red", linewidth=2, label="Median value")

    # Modify the graph
    ax.set_ylim([-1.5, 6])
    ax.set_yticks(ys)
    ax.set_yticklabels(y_tick_label)
    ax.set_title(group_title_dict[group], fontsize=24, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=16)

    return fig


def multi_tornado_plot(tornado_vals, baseline_vals):
    # create an axis
    fig, axs = plt.subplots(5, 3, figsize=(50, 20))

    # get the groups for plotting and sort them
    plot_groups = np.sort(tornado_vals.group.unique())

    # remove overall
    plot_groups = [item for item in plot_groups if item != "overall"]

    plot_groups_reordered = [
        "het_black_female",
        "het_white_female",
        "het_hisp_female",
        "het_black_male",
        "het_white_male",
        "het_hisp_male",
        "idu_black_female",
        "idu_white_female",
        "idu_hisp_female",
        "idu_black_male",
        "idu_white_male",
        "idu_hisp_male",
        "msm_black_male",
        "msm_white_male",
        "msm_hisp_male",
    ]

    column_names = ["Black", "White", "Hispanic"]

    row_names = ["HET Women", "HET Men", "WWID", "MWID", "MSM"]

    for i, group in enumerate(plot_groups_reordered):
        group_vals = tornado_vals[tornado_vals["group"] == group].reset_index(drop=True)
        ax = axs.flatten()[i]

        if i < 3:
            ax.set_title(column_names[i], fontweight="bold", fontsize=24)

        if i % 3 == 0:
            k = i // 3
            ax.set_ylabel(row_names[k], fontweight="bold", fontsize=24)

        # plot parameters
        y_tick_label = group_vals["variable"]
        ys = range(len(y_tick_label))[::-1]

        color_lesser = "#0d47a1"
        color_greater = "#e2711d"
        color_line = "#2ECC71"

        # Data to be visualized
        lesser_value = group_vals.iloc[:, 2].values
        greater_value = group_vals.iloc[:, 3].values
        base = baseline_vals[baseline_vals["group"] == group]["risk"].values[0]

        # Draw bars
        for y, value, value2 in zip(ys, lesser_value, greater_value):
            # draw horizontal line markers for the uncertainty around lesser and greater
            # lesser
            ax.plot(
                (value.iloc[0], value.iloc[2]),
                (y, y),
                linewidth=2,
                color=color_lesser,
                marker="|",
                markersize=15,
            )
            ax.plot(value.iloc[1], y, "o", color=color_lesser, markersize=15)

            # greater
            ax.plot(
                (value2.iloc[0], value2.iloc[2]),
                (y, y),
                linewidth=2,
                color=color_greater,
                marker="|",
                markersize=15,
            )
            ax.plot(value2.iloc[1], y, "o", color=color_greater, markersize=15)

        # Add vertical line for median value
        ax.axvline(base, color=color_line, linewidth=2, label="Median value")

        # add vertical line for the 90% and 110% of baseline
        ax.axvline(base * 0.9, color="red", linewidth=2, label="Median value")
        ax.axvline(base * 1.1, color="red", linewidth=2, label="Median value")

        # Modify the graph
        ax.set_ylim([-1.5, 6])
        ax.set_yticks(ys)
        ax.set_yticklabels(y_tick_label)
        ax.tick_params(axis="both", which="major", labelsize=12)

    return fig


if __name__ == "__main__":
    start_time = datetime.now()

    # Define the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline")
    parser.add_argument("--variable")
    parser.add_argument("--baseline_sa")
    parser.add_argument("--variable_sa")
    parser.add_argument("--out_dir")
    parser.add_argument("--num_samples", default=10000, type=int)
    parser.add_argument("--dm", default=False, type=bool)
    args = parser.parse_args()

    baseline_dir = Path(args.baseline)
    variable_dir = Path(args.variable)
    baseline_dir_sa = Path(args.baseline_sa)
    variable_dir_sa = Path(args.variable_sa)
    dm_run = args.dm

    out_dir = Path(args.out_dir)
    num_samples = args.num_samples

    param_df_baseline = pd.read_parquet(baseline_dir / "parameters.parquet").reset_index(drop=True)
    param_df_variable = pd.read_parquet(variable_dir / "parameters.parquet").reset_index(drop=True)
    param_df_baseline_sa = pd.read_parquet(baseline_dir_sa / "parameters.parquet").reset_index(
        drop=True
    )
    param_df_variable_sa = pd.read_parquet(variable_dir_sa / "parameters.parquet").reset_index(
        drop=True
    )

    # add overall to params
    param_df_baseline = add_overall_to_params(param_df_baseline)
    param_df_variable = add_overall_to_params(param_df_variable)
    param_df_baseline_sa = add_overall_to_params(param_df_baseline_sa)
    param_df_variable_sa = add_overall_to_params(param_df_variable_sa)

    # load the baseline data
    bmi_int_dm_prev_baseline = dd.read_parquet(baseline_dir / "dm_final_output.parquet")
    bmi_int_dm_prev_baseline_sa = dd.read_parquet(baseline_dir_sa / "dm_final_output.parquet")

    # add overall
    bmi_int_dm_prev_baseline = add_overall(bmi_int_dm_prev_baseline)
    bmi_int_dm_prev_baseline_sa = add_overall(bmi_int_dm_prev_baseline_sa)

    # clean to control specifications
    control_bmi_int_dm_prev_baseline = clean_control(
        bmi_int_dm_prev_baseline, only_eligible=True, only_received=True
    )
    control_bmi_int_dm_prev_baseline_sa = clean_control(
        bmi_int_dm_prev_baseline_sa, only_eligible=True, only_received=True
    )
    del bmi_int_dm_prev_baseline

    # filter for only people eligible for intervention
    bmi_int_eligible_risk_baseline = calc_risk_by_group(
        control_bmi_int_dm_prev_baseline, 7
    ).compute()
    bmi_int_eligible_dm_baseline = bmi_int_eligible_risk_baseline["dm_num"]

    bmi_int_eligible_risk_baseline_sa = calc_risk_by_group(
        control_bmi_int_dm_prev_baseline_sa, 7
    ).compute()
    bmi_int_eligible_dm_baseline_sa = bmi_int_eligible_risk_baseline_sa["dm_num"]

    if not dm_run:
        # risk
        baseline_risk_median = (
            bmi_int_eligible_risk_baseline.groupby("group")["risk"].median().reset_index()
        )
        baseline_risk_sa_median = (
            bmi_int_eligible_risk_baseline_sa.groupby("group")["risk"].median().reset_index()
        )
    else:
        # dm
        baseline_dm_median = (
            bmi_int_eligible_dm_baseline.groupby("group")["dm_num"].median().reset_index()
        )
        baseline_dm_sa_median = (
            bmi_int_eligible_dm_baseline_sa.groupby("group")["dm_num"].median().reset_index()
        )

    # load the variable data
    bmi_int_dm_prev_variable = dd.read_parquet(variable_dir / "dm_final_output.parquet")
    bmi_int_dm_prev_variable_sa = dd.read_parquet(variable_dir_sa / "dm_final_output.parquet")

    # add overall
    bmi_int_dm_prev_variable = add_overall(bmi_int_dm_prev_variable)
    bmi_int_dm_prev_variable_sa = add_overall(bmi_int_dm_prev_variable_sa)

    # clean to control specifications
    control_bmi_int_dm_prev_variable = clean_control(
        bmi_int_dm_prev_variable, only_eligible=True, only_received=True
    )
    control_bmi_int_dm_prev_variable_sa = clean_control(
        bmi_int_dm_prev_variable_sa, only_eligible=True, only_received=True
    )

    # filter for only people eligible for intervention
    bmi_int_eligible_risk_variable = calc_risk_by_group(
        control_bmi_int_dm_prev_variable, 7
    ).compute()
    bmi_int_eligible_dm_variable = bmi_int_eligible_risk_variable["dm_num"]

    bmi_int_eligible_risk_variable_sa = calc_risk_by_group(
        control_bmi_int_dm_prev_variable_sa, 7
    ).compute()
    bmi_int_eligible_dm_variable_sa = bmi_int_eligible_risk_variable_sa["dm_num"]

    if not dm_run:
        # risk
        variable_risk_median = (
            bmi_int_eligible_risk_variable.groupby("group")["risk"].median().reset_index()
        )
        variable_risk_sa_median = (
            bmi_int_eligible_risk_variable_sa.groupby("group")["risk"].median().reset_index()
        )

        # risk
        # gather samples from each non SA run
        baseline_risk = (
            bmi_int_eligible_risk_baseline.groupby("group")
            .sample(num_samples, replace=True)
            .reset_index()
        )
        variable_risk = (
            bmi_int_eligible_risk_variable.groupby("group")
            .sample(num_samples, replace=True)
            .reset_index()
        )
        # baseline risk difference
        # absolute difference
        risk_df = pd.DataFrame(variable_risk["risk"] - baseline_risk["risk"])

        # relative difference risk
        risk_df_relative = pd.DataFrame(
            (variable_risk["risk"] - baseline_risk["risk"]) / baseline_risk["risk"]
        )

        # add back the group column that is lost
        risk_df["group"] = baseline_risk["group"]
        risk_df_relative["group"] = baseline_risk["group"]

        # take the median across groups
        baseline_risk_difference_df = risk_df.groupby("group").median().reset_index()
        baseline_risk_difference_df_relative = (
            risk_df_relative.groupby("group").median().reset_index()
        )

        # gather samples from each SA run
        baseline_risk_sa = (
            bmi_int_eligible_risk_baseline_sa.groupby("group")
            .sample(num_samples, replace=True)
            .reset_index()
        )
        variable_risk_sa = (
            bmi_int_eligible_risk_variable_sa.groupby("group")
            .sample(num_samples, replace=True)
            .reset_index()
        )

        # SA
        # absolute difference
        risk_df_sa = pd.DataFrame(variable_risk_sa["risk"] - baseline_risk_sa["risk"])

        # relative difference
        risk_df_sa_relative = pd.DataFrame(
            (variable_risk_sa["risk"] - baseline_risk_sa["risk"]) / baseline_risk_sa["risk"]
        )

        # add back the group column that is lost
        risk_df_sa["group"] = baseline_risk_sa["group"]
        risk_df_sa_relative["group"] = baseline_risk_sa["group"]

        # take the median across groups
        baseline_risk_difference_df_sa = risk_df_sa.groupby("group").median().reset_index()
        baseline_risk_difference_df_sa_relative = (
            risk_df_sa_relative.groupby("group").median().reset_index()
        )

        baseline_tornado_vals = []
        variable_tornado_vals = []
        difference_tornado_vals = []
        relative_tornado_vals = []
        for col in target_columns:
            baseline_vals, variable_vals, difference_vals, relative_difference_vals = (
                calc_tornado_vals(
                    bmi_int_eligible_risk_baseline_sa,
                    bmi_int_eligible_risk_variable_sa,
                    param_df_baseline_sa,
                    param_df_variable_sa,
                    col,
                    lesser=0.20,
                    greater=0.80,
                )
            )

            baseline_tornado_vals.append(baseline_vals)
            variable_tornado_vals.append(variable_vals)
            difference_tornado_vals.append(difference_vals)
            relative_tornado_vals.append(relative_difference_vals)

        baseline_tornado_vals = pd.concat(baseline_tornado_vals).reset_index(drop=True)
        variable_tornado_vals = pd.concat(variable_tornado_vals).reset_index(drop=True)
        difference_tornado_vals = pd.concat(difference_tornado_vals).reset_index(drop=True)
        relative_tornado_vals = pd.concat(relative_tornado_vals).reset_index(drop=True)

        # rename variable to semantic labels
        baseline_tornado_vals["variable"] = baseline_tornado_vals["variable"].map(
            variable_name_map
        )
        variable_tornado_vals["variable"] = variable_tornado_vals["variable"].map(
            variable_name_map
        )
        difference_tornado_vals["variable"] = difference_tornado_vals["variable"].map(
            variable_name_map
        )
        relative_tornado_vals["variable"] = relative_tornado_vals["variable"].map(
            variable_name_map
        )
        baseline_tornado = multi_tornado_plot(baseline_tornado_vals, baseline_risk_sa_median)
        baseline_tornado.savefig(out_dir / "baseline_tornado.png", bbox_inches="tight")
        plt.clf()

        variable_tornado = multi_tornado_plot(variable_tornado_vals, variable_risk_sa_median)
        variable_tornado.savefig(out_dir / "variable_tornado.png", bbox_inches="tight")
        plt.clf()

        difference_tornado = multi_tornado_plot(
            difference_tornado_vals, baseline_risk_difference_df_sa
        )
        difference_tornado.savefig(out_dir / "tornado_absolute.png", bbox_inches="tight")
        plt.clf()

        relative_tornado = multi_tornado_plot(
            relative_tornado_vals, baseline_risk_difference_df_sa_relative
        )
        relative_tornado.savefig(out_dir / "tornado_relative.png", bbox_inches="tight")
        plt.clf()

        baseline_overall = overall_tornado_plot(baseline_tornado_vals, baseline_risk_sa_median)
        baseline_overall.savefig(out_dir / "overall_baseline.png", bbox_inches="tight")
        plt.clf()

        variable_overall = overall_tornado_plot(variable_tornado_vals, variable_risk_sa_median)
        variable_overall.savefig(out_dir / "overall_variable.png", bbox_inches="tight")
        plt.clf()

        overall_absolute = overall_tornado_plot(
            difference_tornado_vals, baseline_risk_difference_df_sa
        )
        overall_absolute.savefig(out_dir / "overall_aboslute.png", bbox_inches="tight")
        plt.clf()

        overall_relative = overall_tornado_plot(
            relative_tornado_vals, baseline_risk_difference_df_sa_relative
        )
        overall_relative.savefig(out_dir / "overall_relative.png", bbox_inches="tight")
        plt.clf()

    else:
        # dm
        variable_dm_median = (
            bmi_int_eligible_dm_variable.groupby("group")["dm_num"].median().reset_index()
        )

        variable_dm_sa_median = (
            bmi_int_eligible_dm_variable_sa.groupby("group")["dm_num"].median().reset_index()
        )

        baseline_dm = (
            bmi_int_eligible_dm_baseline.groupby("group")
            .sample(num_samples, replace=True)
            .reset_index()
        )
        variable_dm = (
            bmi_int_eligible_dm_variable.groupby("group")
            .sample(num_samples, replace=True)
            .reset_index()
        )

        dm_df = pd.DataFrame(variable_dm["dm_num"] - baseline_dm["dm_num"])

        dm_df_relative = pd.DataFrame(
            (variable_dm["dm_num"] - baseline_dm["dm_num"]) / baseline_dm["dm_num"]
        )

        dm_df = baseline_dm["group"]
        dm_df_relative = baseline_dm["group"]

        baseline_dm_difference_df = dm_df.groupby("group").median().reset_index()
        baseline_dm_difference_df_relative = dm_df_relative.groupby("group").median().reset_index()

        baseline_dm_sa = (
            bmi_int_eligible_dm_baseline_sa.groupby("group")
            .sample(num_samples, replace=True)
            .reset_index()
        )
        variable_dm_sa = (
            bmi_int_eligible_dm_variable_sa.groupby("group")
            .sample(num_samples, replace=True)
            .reset_index()
        )

        dm_df_sa = pd.DataFrame(variable_dm_sa["risk"] - baseline_dm_sa["risk"])

        dm_df_sa_relative = pd.DataFrame(
            (variable_dm_sa["risk"] - baseline_dm_sa["risk"]) / baseline_dm_sa["risk"]
        )

        dm_df_sa["group"] = baseline_dm_sa["group"]
        dm_df_sa_relative = baseline_dm_sa["group"]

        # take the median across groups
        baseline_dm_difference_df_sa = dm_df_sa.groupby("group").median().reset_index()
        baseline_dm_difference_df_sa_relative = (
            dm_df_sa_relative.groupby("group").median().reset_index()
        )

        dm_baseline_tornado_vals = []
        dm_variable_tornado_vals = []
        dm_difference_tornado_vals = []
        dm_relative_tornado_vals = []
        for col in target_columns:
            baseline_vals, variable_vals, difference_vals, relative_difference_vals = (
                calc_tornado_vals(
                    bmi_int_eligible_dm_baseline_sa,
                    bmi_int_eligible_dm_variable_sa,
                    param_df_baseline_sa,
                    param_df_variable_sa,
                    col,
                    lesser=0.20,
                    greater=0.80,
                )
            )

            dm_baseline_tornado_vals.append(baseline_vals)
            dm_variable_tornado_vals.append(variable_vals)
            dm_difference_tornado_vals.append(difference_vals)
            dm_relative_tornado_vals.append(relative_difference_vals)

        dm_baseline_tornado_vals = pd.concat(dm_baseline_tornado_vals).reset_index(drop=True)
        dm_variable_tornado_vals = pd.concat(dm_variable_tornado_vals).reset_index(drop=True)
        dm_difference_tornado_vals = pd.concat(dm_difference_tornado_vals).reset_index(drop=True)
        dm_relative_tornado_vals = pd.concat(dm_relative_tornado_vals).reset_index(drop=True)

        dm_baseline_tornado_vals["variable"] = dm_baseline_tornado_vals["variable"].map(
            variable_name_map
        )
        dm_variable_tornado_vals["variable"] = dm_variable_tornado_vals["variable"].map(
            variable_name_map
        )
        dm_difference_tornado_vals["variable"] = dm_difference_tornado_vals["variable"].map(
            variable_name_map
        )
        dm_relative_tornado_vals["variable"] = dm_relative_tornado_vals["variable"].map(
            variable_name_map
        )

        dm_baseline_tornado = multi_tornado_plot(dm_baseline_tornado_vals, baseline_dm_sa_median)
        dm_baseline_tornado.savefig(out_dir / "baseline_dm_tornado.png", bbox_inches="tight")
        plt.clf()

        dm_variable_tornado = multi_tornado_plot(dm_variable_tornado_vals, variable_dm_sa_median)
        dm_variable_tornado.savefig(out_dir / "variable_dm_tornado.png", bbox_inches="tight")
        plt.clf()

        dm_difference_tornado = multi_tornado_plot(
            dm_difference_tornado_vals, baseline_dm_difference_df_sa
        )
        dm_difference_tornado.savefig(out_dir / "tornado_dm_absolute.png", bbox_inches="tight")
        plt.clf()

        dm_relative_tornado = multi_tornado_plot(
            dm_relative_tornado_vals, baseline_dm_difference_df_sa_relative
        )
        dm_relative_tornado.savefig(out_dir / "tornado_dm_relative.png", bbox_inches="tight")
        plt.clf()

        dm_baseline_overall = overall_tornado_plot(dm_baseline_tornado_vals, baseline_dm_sa_median)
        dm_baseline_overall.savefig(out_dir / "overall_dm_baseline.png", bbox_inches="tight")
        plt.clf()

        dm_variable_overall = overall_tornado_plot(dm_variable_tornado_vals, variable_dm_sa_median)
        dm_variable_overall.savefig(out_dir / "overall_dm_variable.png", bbox_inches="tight")
        plt.clf()

        dm_overall_absolute = overall_tornado_plot(
            dm_difference_tornado_vals, baseline_dm_difference_df_sa
        )
        dm_overall_absolute.savefig(out_dir / "overall_dm_aboslute.png", bbox_inches="tight")
        plt.clf()

        dm_overall_relative = overall_tornado_plot(
            dm_relative_tornado_vals, baseline_dm_difference_df_sa_relative
        )
        dm_overall_relative.savefig(out_dir / "overall_relative.png", bbox_inches="tight")
        plt.clf()
