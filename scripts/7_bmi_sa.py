import argparse
from datetime import datetime
from pathlib import Path

from dask import dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pearl.post_processing.bmi import calc_risk_by_group, clean_control


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
            lesser: (lesser_group_df_baseline_sample["risk"]).median(),
            greater: (greater_group_df_baseline_sample["risk"]).median(),
            "lesser_count": lesser_group_df_baseline["risk"].count(),
            "greater_count": greater_group_df_baseline["risk"].count(),
        }
        baseline_tornado_df.append(baseline_group_tornado_df)
        variable_group_tornado_df = {
            "group": group,
            "variable": col_name,
            lesser: (lesser_group_df_variable_sample["risk"]).median(),
            greater: (greater_group_df_variable_sample["risk"]).median(),
            "lesser_count": lesser_group_df_variable["risk"].count(),
            "greater_count": greater_group_df_variable["risk"].count(),
        }
        variable_tornado_df.append(variable_group_tornado_df)
        difference_group_tornado_df = {
            "group": group,
            "variable": col_name,
            lesser: (
                lesser_group_df_variable_sample["risk"] - lesser_group_df_baseline_sample["risk"]
            ).median(),
            greater: (
                greater_group_df_variable_sample["risk"] - greater_group_df_baseline_sample["risk"]
            ).median(),
        }
        difference_tornado_df.append(difference_group_tornado_df)

    baseline_tornado_df = pd.DataFrame(baseline_tornado_df)
    variable_tornado_df = pd.DataFrame(variable_tornado_df)
    difference_tornado_df = pd.DataFrame(difference_tornado_df)
    return baseline_tornado_df, variable_tornado_df, difference_tornado_df


def tornado_plot(tornado_vals, baseline_vals):
    # create an axis
    fig, axs = plt.subplots(5, 3, figsize=(40, 20))

    # get the groups for plotting and sort them
    plot_groups = np.sort(tornado_vals.group.unique())

    for i, group in enumerate(plot_groups):
        group_vals = tornado_vals[tornado_vals["group"] == group].reset_index(drop=True)
        ax = axs.flatten()[i]

        # plot parameters
        y_tick_label = group_vals["variable"]
        ys = range(len(y_tick_label))[::-1]
        height = 0.8

        color_lesser = "#0d47a1"
        color_greater = "#e2711d"
        color_line = "#2ECC71"

        # Data to be visualized
        lesser_value = group_vals.iloc[:, 2].values
        greater_value = group_vals.iloc[:, 3].values
        base = baseline_vals[baseline_vals["group"] == group]["risk"].values[0]

        # Draw bars
        for y, value, value2 in zip(ys, lesser_value, greater_value):
            if abs(base - value) < abs(base - value2):
                ax.broken_barh(
                    [(base, value2 - base)],
                    (y - height / 2, height),
                    facecolors=[color_greater, color_greater],
                )
                ax.broken_barh(
                    [(base, value - base)],
                    (y - height / 2, height),
                    facecolors=[color_lesser, color_lesser],
                )
            else:
                ax.broken_barh(
                    [(base, value - base)],
                    (y - height / 2, height),
                    facecolors=[color_lesser, color_lesser],
                )
                ax.broken_barh(
                    [(base, value2 - base)],
                    (y - height / 2, height),
                    facecolors=[color_greater, color_greater],
                )

        # Add vertical line for median value
        ax.axvline(base, color=color_line, linewidth=1.5, label="Median value")

        # Modify the graph
        ax.set_ylim([-1.5, 6])
        ax.set_yticks(ys)
        ax.set_yticklabels(y_tick_label)
        ax.set_title(group, fontsize=12, fontweight="bold")

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
    args = parser.parse_args()

    baseline_dir = Path(args.baseline)
    variable_dir = Path(args.variable)
    baseline_dir_sa = Path(args.baseline_sa)
    variable_dir_sa = Path(args.variable_sa)

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

    # load the baseline data
    bmi_int_dm_prev_baseline = dd.read_parquet(baseline_dir / "dm_final_output.parquet")
    bmi_int_dm_prev_baseline_sa = dd.read_parquet(baseline_dir_sa / "dm_final_output.parquet")

    # clean to control specifications
    control_bmi_int_dm_prev_baseline = clean_control(
        bmi_int_dm_prev_baseline, only_eligible=True, only_received=True
    )
    control_bmi_int_dm_prev_baseline_sa = clean_control(
        bmi_int_dm_prev_baseline_sa, only_eligible=True, only_received=True
    )

    # filter for only people eligible for intervention
    bmi_int_eligible_risk_baseline = calc_risk_by_group(
        control_bmi_int_dm_prev_baseline, 7
    ).compute()
    bmi_int_eligible_risk_baseline_sa = calc_risk_by_group(
        control_bmi_int_dm_prev_baseline_sa, 7
    ).compute()

    baseline_risk_median = (
        bmi_int_eligible_risk_baseline.groupby("group")["risk"].median().reset_index()
    )
    baseline_risk_sa_median = (
        bmi_int_eligible_risk_baseline_sa.groupby("group")["risk"].median().reset_index()
    )

    # load the variable data
    bmi_int_dm_prev_variable = dd.read_parquet(variable_dir / "dm_final_output.parquet")
    bmi_int_dm_prev_variable_sa = dd.read_parquet(variable_dir_sa / "dm_final_output.parquet")

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
    bmi_int_eligible_risk_variable_sa = calc_risk_by_group(
        control_bmi_int_dm_prev_variable_sa, 7
    ).compute()

    variable_risk_median = (
        bmi_int_eligible_risk_variable.groupby("group")["risk"].median().reset_index()
    )

    variable_risk_sa_median = (
        bmi_int_eligible_risk_variable_sa.groupby("group")["risk"].median().reset_index()
    )

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

    # take the difference between the samples
    risk_df = pd.DataFrame(variable_risk["risk"] - baseline_risk["risk"])

    # add back the group column that is lost
    risk_df["group"] = baseline_risk["group"]

    # take the median across groups
    baseline_risk_difference_df = risk_df.groupby("group").median().reset_index()

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

    # take the difference between the samples
    risk_df_sa = pd.DataFrame(variable_risk_sa["risk"] - baseline_risk_sa["risk"])

    # add back the group column that is lost
    risk_df_sa["group"] = baseline_risk_sa["group"]

    # take the median across groups
    baseline_risk_difference_df_sa = risk_df_sa.groupby("group").median().reset_index()

    target_columns = [
        "dm_prevalence",
        "dm_prevalence_prev",
        "dm_incidence",
        "pre_art_bmi",
        "post_art_bmi",
        "art_initiators",
    ]

    variable_name_map = {
        "prev_users_dict_dm": "DM Prevalence at 2009",
        "prev_inits_dict_dm": "DM Prevalence 2009-end",
        "dm_incidence": "DM Incidence",
        "pre_art_bmi": "Pre ART BMI",
        "post_art_bmi": "Post ART BMI",
        "art_initiators": "# ART Initiators",
    }

    baseline_tornado_vals = []
    variable_tornado_vals = []
    difference_tornado_vals = []
    for col in target_columns:
        baseline_vals, variable_vals, difference_vals = calc_tornado_vals(
            bmi_int_eligible_risk_baseline_sa,
            bmi_int_eligible_risk_variable_sa,
            param_df_baseline_sa,
            param_df_variable_sa,
            col,
            lesser=0.10,
            greater=0.90,
        )
        baseline_tornado_vals.append(baseline_vals)
        variable_tornado_vals.append(variable_vals)
        difference_tornado_vals.append(difference_vals)
    baseline_tornado_vals = pd.concat(baseline_tornado_vals).reset_index(drop=True)
    variable_tornado_vals = pd.concat(variable_tornado_vals).reset_index(drop=True)
    difference_tornado_vals = pd.concat(difference_tornado_vals).reset_index(drop=True)

    # rename variable to semantic labels
    baseline_tornado_vals["variable"] = baseline_tornado_vals["variable"].map(variable_name_map)
    variable_tornado_vals["variable"] = variable_tornado_vals["variable"].map(variable_name_map)
    difference_tornado_vals["variable"] = difference_tornado_vals["variable"].map(
        variable_name_map
    )

    baseline_tornado = tornado_plot(baseline_tornado_vals, baseline_risk_sa_median)

    baseline_tornado.savefig(out_dir / "baseline_tornado.png")
    plt.clf()

    variable_tornado = tornado_plot(variable_tornado_vals, variable_risk_sa_median)

    variable_tornado.savefig(out_dir / "variable_tornado.png")
    plt.clf()

    difference_tornado = tornado_plot(difference_tornado_vals, baseline_risk_difference_df_sa)

    difference_tornado.savefig(out_dir / "tornado.png")
    plt.clf()
