from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.analysis.common import (
    OUTPUT_VISUALIZATIONS_DIR,
    BAR_COLORMAP,
    SEQUENTIAL_HEATMAP_COLORMAP,
    PLOT_FIGSIZE_ACTIVITY_TYPES_STACKED,
    _apply_label_truncation,
    LABEL_MAX_CHARS,
    create_and_save_figure,
    logger, MAX_DAYS_HEATMAP, compute_barh_fig_height, safe_filename, ensure_dir, PIE_COLORMAP,
)



def _save_category_bar_plot(
    series: pd.Series,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    figsize: tuple[int, int] = (12, 6),
) -> None:
    """
    Save a standard categorical bar chart.
    """
    if series is None or series.empty:
        return

    def plot() -> None:
        series.plot(kind="bar", colormap=BAR_COLORMAP)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45, ha="right")

    create_and_save_figure(
        plot,
        os.path.join(OUTPUT_VISUALIZATIONS_DIR, filename),
        figsize=figsize,
    )


def _save_daily_bar_plot(
    series: pd.Series,
    *,
    title: str,
    ylabel: str,
    filename: str,
    figsize: tuple[int, int] = (14, 7),
) -> None:
    """
    Save a date-indexed bar chart with readable date labels.
    """
    if series is None or series.empty:
        return

    series = series.sort_index()
    date_labels = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in series.index]

    def plot() -> None:
        x_pos = range(len(series))
        plt.bar(x_pos, series.values, width=0.6)
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel(ylabel)

        ax = plt.gca()
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(date_labels)
        ax.tick_params(axis="x", rotation=90)
        plt.margins(x=0.01)

    create_and_save_figure(
        plot,
        os.path.join(OUTPUT_VISUALIZATIONS_DIR, filename),
        figsize=figsize,
    )


def save_activity_types_distribution_plot(activity_counts: pd.Series) -> None:
    _save_category_bar_plot(
        activity_counts,
        title="Distribution of Activity Types",
        xlabel="Activity Type",
        ylabel="Count",
        filename="activity_types_distribution.png",
        figsize=(12, 6),
    )


def save_activities_over_time_plot(daily_activities: pd.Series) -> None:
    _save_daily_bar_plot(
        daily_activities,
        title="Number of Activities per Day",
        ylabel="Number of Activities",
        filename="activities_over_time.png",
        figsize=(14, 7),
    )


def save_points_by_activity_type_plot(points_by_type: pd.Series) -> None:
    _save_category_bar_plot(
        points_by_type,
        title="Total Points by Activity Type",
        xlabel="Activity Type",
        ylabel="Total Points",
        filename="points_by_activity_type.png",
        figsize=(12, 6),
    )


def save_points_by_player_plot(points_by_user: pd.Series) -> None:
    _save_category_bar_plot(
        points_by_user,
        title="Total Rewarded Points by Player",
        xlabel="Player ID",
        ylabel="Total Rewarded Points",
        filename="points_by_player.png",
        figsize=(12, 6),
    )


def save_rewards_by_activity_type_plot(rewards_by_type: pd.Series) -> None:
    _save_category_bar_plot(
        rewards_by_type,
        title="Average Rewarded Points by Activity Type",
        xlabel="Activity Type",
        ylabel="Average Rewarded Points",
        filename="rewards_by_activity_type.png",
        figsize=(12, 6),
    )

def save_points_over_time_plot(daily_points: pd.Series) -> None:
    _save_daily_bar_plot(
        daily_points,
        title="Total Rewarded Points per Day",
        ylabel="Total Rewarded Points",
        filename="points_over_time.png",
        figsize=(14, 7),
    )


def save_activity_heatmap_by_time_plot(activity_heatmap_data: pd.DataFrame) -> None:
    if activity_heatmap_data is None or activity_heatmap_data.empty:
        return

    if activity_heatmap_data.values.sum() <= 0:
        return

    def plot() -> None:
        sns.heatmap(
            activity_heatmap_data,
            cmap=SEQUENTIAL_HEATMAP_COLORMAP,
            annot=True,
            fmt=".0f",
            linewidths=0.5,
        )
        plt.title("Activity Heatmap by Day of Week and Hour of Day")
        plt.xlabel("Hour of Day")
        plt.ylabel("Day of Week")

    create_and_save_figure(
        plot,
        os.path.join(OUTPUT_VISUALIZATIONS_DIR, "activity_heatmap_by_time.png"),
        figsize=(15, 8),
    )


def save_activity_type_by_player_heatmap(heat: pd.DataFrame) -> None:
    if heat is None or heat.empty:
        return

    if heat.values.sum() <= 0:
        return

    player_count = len(heat.index)
    annot_fontsize = 7
    height = min(20, 8 + max(0, (player_count - 10) / 5 * 0.5))

    def plot() -> None:
        sns.heatmap(
            heat,
            cmap=SEQUENTIAL_HEATMAP_COLORMAP,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            annot_kws={"fontsize": annot_fontsize},
        )
        plt.title(f"Activity Type Distribution by Player ({player_count} Players)")
        plt.xlabel("Activity Type")
        plt.ylabel("Player ID")

    create_and_save_figure(
        plot,
        os.path.join(OUTPUT_VISUALIZATIONS_DIR, "activity_type_by_player.png"),
        figsize=(14, height),
    )

def save_player_activity_distribution_plot(user_activity: pd.Series) -> None:
    _save_category_bar_plot(
        user_activity,
        title="Number of Activities by Player",
        xlabel="Player ID",
        ylabel="Number of Activities",
        filename="player_activity_distribution.png",
        figsize=(12, 6),
    )


def save_usage_by_day_of_week_plot(activities_by_day: pd.Series) -> None:
    if activities_by_day is None or activities_by_day.empty:
        return

    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    counts = [int(activities_by_day.get(day, 0)) for day in day_order]

    def plot() -> None:
        x = range(len(day_order))
        plt.bar(x, counts)
        plt.title("Usage by Day of Week")
        plt.xlabel("Day of Week")
        plt.ylabel("Number of Activities")
        plt.xticks(list(x), day_order, rotation=45, ha="right")
        plt.margins(x=0.01)

    create_and_save_figure(
        plot,
        os.path.join(OUTPUT_VISUALIZATIONS_DIR, "usage_by_day_of_week.png"),
        figsize=(10, 6),
    )


def build_independent_colormap(n_colors: int):
    import numpy as np
    from matplotlib.colors import ListedColormap, hsv_to_rgb

    palettes = [
        "tab20", "tab20b", "tab20c", "tab10",
        "Set3", "Accent", "Dark2", "Set2", "Set1", "Pastel1", "Pastel2",
    ]

    colors_list = []
    for name in palettes:
        cmap = plt.get_cmap(name)
        if hasattr(cmap, "colors"):
            colors_list.extend(list(cmap.colors))
        else:
            colors_list.extend([cmap(i) for i in np.linspace(0, 1, 20)])

    unique = []
    seen = set()
    for rgba in colors_list:
        key = tuple(round(float(x), 5) for x in rgba)
        if key not in seen:
            seen.add(key)
            unique.append(rgba)

    if n_colors <= len(unique):
        selected = unique[:n_colors]
    else:
        selected = list(unique)
        extra = n_colors - len(unique)
        for i in range(extra):
            h = (i / max(1, extra))
            rgb = hsv_to_rgb([h, 0.75, 0.9])
            selected.append((rgb[0], rgb[1], rgb[2], 1.0))

    return ListedColormap(selected, name="independent")


def save_wave_comparisons_plot(activities: pd.DataFrame) -> None:
    if activities is None or activities.empty or "wave" not in activities.columns:
        return

    wave_activity_counts = activities.groupby("wave").size()
    wave_points = activities.groupby("wave")["points"].mean()

    if wave_activity_counts.empty and wave_points.empty:
        return

    def plot() -> None:
        fig, axes = plt.subplots(2, 1, figsize=(12, 12))

        wave_activity_counts.plot(kind="bar", ax=axes[0], colormap=BAR_COLORMAP)
        axes[0].set_title("Number of Activities per Wave")
        axes[0].set_xlabel("Wave")
        axes[0].set_ylabel("Number of Activities")

        wave_points.plot(kind="bar", ax=axes[1], colormap=BAR_COLORMAP)
        axes[1].set_title("Average Points per Wave")
        axes[1].set_xlabel("Wave")
        axes[1].set_ylabel("Average Points")

        plt.tight_layout()

    create_and_save_figure(
        plot,
        os.path.join(OUTPUT_VISUALIZATIONS_DIR, "wave_comparisons.png"),
        figsize=(12, 12),
    )

def save_wave_comparisons_by_activity_type_plot(activities: pd.DataFrame) -> None:
    if activities is None or activities.empty:
        return
    if "wave" not in activities.columns or "type" not in activities.columns:
        return

    type_wave_counts = pd.crosstab(activities["wave"], activities["type"])
    if type_wave_counts.empty or type_wave_counts.shape[1] == 0:
        return

    cmap = build_independent_colormap(type_wave_counts.shape[1])

    def plot() -> None:
        type_wave_counts.plot(kind="bar", figsize=(12, 6), colormap=cmap)
        plt.title("Activities per Wave by Activity Types")
        plt.xlabel("Wave")
        plt.ylabel("Number of Activities")
        plt.legend(title="Activity Type")
        plt.tight_layout()

    create_and_save_figure(
        plot,
        os.path.join(OUTPUT_VISUALIZATIONS_DIR, "wave_comparisons_by_activity_type.png"),
        figsize=(12, 8),
    )


def save_wave_points_by_activity_type_plot(activities: pd.DataFrame) -> None:
    if activities is None or activities.empty:
        return
    if "wave" not in activities.columns or "type" not in activities.columns or "points" not in activities.columns:
        return

    wave_type_points = activities.groupby(["wave", "type"])["points"].mean().unstack()
    if wave_type_points.empty or wave_type_points.shape[1] == 0:
        return

    cmap = build_independent_colormap(wave_type_points.shape[1])

    def plot() -> None:
        wave_type_points.plot(kind="bar", figsize=(12, 6), colormap=cmap)
        plt.title("Average Points per Wave by Activity Types")
        plt.xlabel("Wave")
        plt.ylabel("Average Points")
        plt.legend(title="Activity Type")
        plt.tight_layout()

    create_and_save_figure(
        plot,
        os.path.join(OUTPUT_VISUALIZATIONS_DIR, "wave_points_by_activity_type.png"),
        figsize=(12, 8),
    )

def save_wave_comparisons_by_player_plot(activities: pd.DataFrame) -> None:
    if activities is None or activities.empty:
        return
    if "wave" not in activities.columns or "pid" not in activities.columns:
        return

    dfp = activities.dropna(subset=["wave", "pid"]).copy()
    if dfp.empty:
        return

    dfp["pid_str"] = dfp["pid"].astype(str)

    player_volume = dfp.groupby("pid_str").size().sort_values(ascending=False)
    if player_volume.empty:
        return

    players = player_volume.index.tolist()

    wave_player_counts = (
        dfp.groupby(["wave", "pid_str"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=players, fill_value=0)
        .sort_index()
    )

    if "points" not in dfp.columns:
        dfp["points"] = 0

    wave_player_points = (
        dfp.groupby(["wave", "pid_str"])["points"]
        .mean()
        .unstack()
        .reindex(columns=players)
        .sort_index()
        .fillna(0.0)
    )

    if wave_player_counts.empty or wave_player_counts.values.sum() <= 0:
        return

    def plot() -> None:
        fig, axes = plt.subplots(2, 1, figsize=(18, 12), sharex=True)

        sns.heatmap(
            wave_player_counts,
            ax=axes[0],
            cmap=SEQUENTIAL_HEATMAP_COLORMAP,
            annot=True,
            fmt="d",
            linewidths=0.2,
        )
        axes[0].set_title(f"Activities per Wave by Player ({len(players)} Players)")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Wave")

        sns.heatmap(
            wave_player_points,
            ax=axes[1],
            cmap=SEQUENTIAL_HEATMAP_COLORMAP,
            annot=True,
            fmt=".1f",
            linewidths=0.2,
        )
        axes[1].set_title("Average Points per Wave by Player")
        axes[1].set_xlabel("Player ID")
        axes[1].set_ylabel("Wave")

        plt.tight_layout()

    create_and_save_figure(
        plot,
        os.path.join(OUTPUT_VISUALIZATIONS_DIR, "wave_comparisons_by_player.png"),
        figsize=(18, 12),
    )


def save_activity_types_stacked_by_date_plot(activities: pd.DataFrame) -> None:
    if activities is None or activities.empty:
        return
    if "date" not in activities.columns or "type" not in activities.columns:
        return

    activity_types_by_date = pd.crosstab(index=activities["date"], columns=activities["type"])
    if activity_types_by_date.empty:
        return

    type_counts = activities["type"].value_counts()
    top_count = min(5, len(type_counts))
    if top_count <= 0:
        return

    top_types = type_counts.nlargest(top_count).index.tolist()

    filtered = activity_types_by_date[top_types].copy()
    filtered.index = pd.to_datetime(filtered.index, errors="coerce")
    filtered = filtered[filtered.index.notna()].sort_index()

    if filtered.empty:
        return

    full_range = pd.date_range(filtered.index.min(), filtered.index.max(), freq="D")
    filtered = filtered.reindex(full_range).fillna(0)

    date_labels = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in filtered.index]
    cmap = build_independent_colormap(filtered.shape[1])

    def plot() -> None:
        plt.figure(figsize=PLOT_FIGSIZE_ACTIVITY_TYPES_STACKED)
        filtered.plot(kind="bar", stacked=True, colormap=cmap, width=1, ax=plt.gca())
        plt.title("Activity Types Distribution by Date (Top 5 Types)")
        plt.xlabel("Date")
        plt.ylabel("Number of Activities")
        plt.legend(title="Activity Type")

        ax = plt.gca()
        ax.set_xticks(range(len(filtered)))
        ax.set_xticklabels(date_labels)
        ax.tick_params(axis="x", rotation=90)
        plt.tight_layout()

    out_path = os.path.join(OUTPUT_VISUALIZATIONS_DIR, "activity_types_stacked_by_date.png")
    plot()
    fig = plt.gcf()
    _apply_label_truncation(fig, LABEL_MAX_CHARS)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure -> {out_path}")


def save_player_engagement_heatmap_plot(activities: pd.DataFrame) -> None:
    if activities is None or activities.empty:
        return
    if "pid" not in activities.columns or "date" not in activities.columns:
        return

    user_day = pd.crosstab(index=activities["pid"], columns=activities["date"]).fillna(0)
    if user_day.empty:
        return

    cols_sorted = sorted(list(user_day.columns))
    if len(cols_sorted) > MAX_DAYS_HEATMAP:
        cols_sorted = cols_sorted[-MAX_DAYS_HEATMAP:]

    user_order = user_day.sum(axis=1).sort_values(ascending=False).index
    heat = user_day.loc[user_order, cols_sorted].copy()

    try:
        heat.columns = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in heat.columns]
    except Exception:
        heat.columns = [str(d) for d in heat.columns]

    if heat.empty or heat.values.sum() <= 0:
        return

    def plot() -> None:
        sns.heatmap(
            heat,
            cmap=SEQUENTIAL_HEATMAP_COLORMAP,
            annot=True,
            fmt="d",
            annot_kws={"fontsize": 5},
            linewidths=0.2,
        )
        plt.title(f"Player Engagement Heatmap by Day ({len(user_order)} Players)")
        plt.xlabel("Date")
        plt.ylabel("Player ID")

    create_and_save_figure(
        plot,
        os.path.join(OUTPUT_VISUALIZATIONS_DIR, "player_engagement_heatmap.png"),
        figsize=(16, 10),
    )


def save_dropout_rates_distribution_plot(user_dropout: pd.DataFrame) -> None:
    if user_dropout is None or user_dropout.empty:
        return
    if "dropout_days" not in user_dropout.columns:
        return

    series = user_dropout["dropout_days"].dropna()
    if series.empty:
        return

    def plot() -> None:
        sns.histplot(series, kde=True, bins=20, color="tab:blue")
        plt.title("Distribution of User Dropout (First to Last Activity)")
        plt.xlabel("Days")
        plt.ylabel("Number of Users")

    create_and_save_figure(
        plot,
        os.path.join(OUTPUT_VISUALIZATIONS_DIR, "dropout_rates_distribution.png"),
        figsize=(10, 6),
    )


def save_joining_rates_distribution_plot(user_dropout: pd.DataFrame) -> None:
    if user_dropout is None or user_dropout.empty:
        return
    if "joining_days" not in user_dropout.columns:
        return

    series = user_dropout["joining_days"].dropna()
    if series.empty:
        return

    def plot() -> None:
        sns.histplot(series, kde=True, bins=20, color="tab:red")
        plt.title("Distribution of User Joining Rates")
        plt.xlabel("Days Between Campaign Start and First Activity")
        plt.ylabel("Number of Users")

    create_and_save_figure(
        plot,
        os.path.join(OUTPUT_VISUALIZATIONS_DIR, "joining_rates_distribution.png"),
        figsize=(10, 6),
    )


def save_combined_dropout_joining_rates_plot(user_dropout: pd.DataFrame) -> None:
    if user_dropout is None or user_dropout.empty:
        return
    required = {"dropout_days", "joining_days"}
    if not required.issubset(user_dropout.columns):
        return

    dropout_series = user_dropout["dropout_days"].dropna()
    joining_series = user_dropout["joining_days"].dropna()
    if dropout_series.empty or joining_series.empty:
        return

    def plot() -> None:
        plt.figure(figsize=(12, 7))
        sns.kdeplot(dropout_series, label="Dropout", fill=True, alpha=0.3, color="blue")
        sns.kdeplot(joining_series, label="Joining", fill=True, alpha=0.3, color="red")
        plt.title("Distribution of Dropout vs Joining Rates")
        plt.xlabel("Days")
        plt.ylabel("Density")
        plt.legend()

    create_and_save_figure(
        plot,
        os.path.join(OUTPUT_VISUALIZATIONS_DIR, "combined_dropout_joining_rates.png"),
        figsize=(12, 7),
    )



def save_combined_dropout_joining_boxplots_plot(user_dropout: pd.DataFrame) -> None:
    if user_dropout is None or user_dropout.empty:
        return
    required = {"dropout_days", "joining_days"}
    if not required.issubset(user_dropout.columns):
        return

    df_long = pd.concat(
        [
            pd.DataFrame({"days": user_dropout["dropout_days"], "metric": "Dropout"}),
            pd.DataFrame({"days": user_dropout["joining_days"], "metric": "Joining"}),
        ],
        ignore_index=True,
    ).dropna(subset=["days"])

    if df_long.empty:
        return

    def plot() -> None:
        ax = sns.boxplot(
            x="metric",
            y="days",
            data=df_long,
            hue="metric",
            palette=["blue", "red"],
            dodge=False,
        )
        if ax.legend_:
            ax.legend_.remove()
        plt.title("Dropout vs Joining (Boxplots)")
        plt.xlabel("")
        plt.ylabel("Days")

    create_and_save_figure(
        plot,
        os.path.join(OUTPUT_VISUALIZATIONS_DIR, "combined_dropout_joining_boxplots.png"),
        figsize=(12, 7),
    )


def save_churn_rate_over_time_plot(churn_df: pd.DataFrame) -> None:
    if churn_df is None or churn_df.empty:
        return
    if "churn_rate" not in churn_df.columns:
        return

    def plot() -> None:
        plt.plot(churn_df.index, churn_df["churn_rate"] * 100.0, color="tab:purple", linewidth=2)
        plt.title("Churn Rate Over Time (No activity in last 30 days)")
        plt.xlabel("Date")
        plt.ylabel("Churn Rate (%)")
        plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)

    create_and_save_figure(
        plot,
        os.path.join(OUTPUT_VISUALIZATIONS_DIR, "churn_rate_over_time.png"),
        figsize=(12, 6),
    )


def save_churn_counts_over_time_plot(churn_df: pd.DataFrame) -> None:
    if churn_df is None or churn_df.empty:
        return
    required = {"joined_count", "churned_count"}
    if not required.issubset(churn_df.columns):
        return

    def plot() -> None:
        plt.plot(churn_df.index, churn_df["joined_count"], linewidth=2, linestyle="--")
        plt.plot(churn_df.index, churn_df["churned_count"], linewidth=2)

        plt.title("Churn Counts Over Time (Joined vs Churned; 30-day inactivity)")
        plt.xlabel("Date")
        plt.ylabel("Number of Users")
        plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)
        plt.legend(
            [
                "Joined (cumulative by first activity)",
                "Churned (inactive last 30 days)",
            ]
        )

    create_and_save_figure(
        plot,
        os.path.join(OUTPUT_VISUALIZATIONS_DIR, "churn_counts_over_time.png"),
        figsize=(12, 6),
    )


def save_reward_challenge_rule_plots(rewards_df: pd.DataFrame) -> None:
    if rewards_df is None or rewards_df.empty:
        return

    required = {"challenge_name", "rule_name", "reward_points", "type"}
    if not required.issubset(rewards_df.columns):
        return

    # Counts by challenge (Top 30)
    challenge_counts = rewards_df["challenge_name"].value_counts().head(30)
    if not challenge_counts.empty:
        def plot_rewards_by_challenge_count() -> None:
            challenge_counts.plot(kind="bar", colormap=BAR_COLORMAP)
            plt.title("Count of Rewarded Activities by Challenge (Top 30)")
            plt.xlabel("Challenge Name")
            plt.ylabel("Count of Rewards")
            plt.xticks(rotation=45, ha="right")

        create_and_save_figure(
            plot_rewards_by_challenge_count,
            os.path.join(OUTPUT_VISUALIZATIONS_DIR, "rewards_count_by_challenge.png"),
            figsize=(14, 7),
        )

    # Total points by challenge
    points_by_challenge = (
        rewards_df.groupby("challenge_name")["reward_points"]
        .sum()
        .sort_values(ascending=False)
    )
    if not points_by_challenge.empty:
        def plot_points_by_challenge() -> None:
            points_by_challenge.plot(kind="bar", colormap=BAR_COLORMAP)
            plt.title("Total Rewarded Points by Challenge")
            plt.xlabel("Challenge Name")
            plt.ylabel("Total Points")
            plt.xticks(rotation=45, ha="right")

        create_and_save_figure(
            plot_points_by_challenge,
            os.path.join(OUTPUT_VISUALIZATIONS_DIR, "points_by_challenge.png"),
            figsize=(14, 7),
        )

    # Counts by rule (Top 30)
    rule_counts = rewards_df["rule_name"].value_counts().head(30)
    if not rule_counts.empty:
        def plot_rewards_by_rule_count() -> None:
            rule_counts.plot(kind="bar", colormap=BAR_COLORMAP)
            plt.title("Count of Rewarded Activities by Rule (Top 30)")
            plt.xlabel("Rule Name")
            plt.ylabel("Count of Rewards")
            plt.xticks(rotation=45, ha="right")

        create_and_save_figure(
            plot_rewards_by_rule_count,
            os.path.join(OUTPUT_VISUALIZATIONS_DIR, "rewards_count_by_rule.png"),
            figsize=(14, 7),
        )

    # Total points by rule
    points_by_rule = (
        rewards_df.groupby("rule_name")["reward_points"]
        .sum()
        .sort_values(ascending=False)
    )
    if not points_by_rule.empty:
        def plot_points_by_rule() -> None:
            points_by_rule.plot(kind="bar", colormap=BAR_COLORMAP)
            plt.title("Total Rewarded Points by Rule")
            plt.xlabel("Rule Name")
            plt.ylabel("Total Points")
            plt.xticks(rotation=45, ha="right")

        create_and_save_figure(
            plot_points_by_rule,
            os.path.join(OUTPUT_VISUALIZATIONS_DIR, "points_by_rule.png"),
            figsize=(14, 7),
        )

    # Per-type: points by challenge
    try:
        out_dir = ensure_dir(os.path.join(OUTPUT_VISUALIZATIONS_DIR, "by_type"))
        for activity_type in rewards_df["type"].dropna().unique().tolist():
            sub = rewards_df[rewards_df["type"] == activity_type]
            series = (
                sub.groupby("challenge_name")["reward_points"]
                .sum()
                .sort_values(ascending=False)
            )
            if series.empty:
                continue

            fname = os.path.join(
                out_dir,
                f"points_by_challenge_type_{safe_filename(activity_type)}.png",
            )
            fig_h = compute_barh_fig_height(len(series))

            def plot_points_by_challenge_for_type(
                series=series,
                activity_type=activity_type,
            ) -> None:
                series.plot(kind="barh", colormap=BAR_COLORMAP)
                plt.title(f"Total Rewarded Points by Challenge — Activity Type: {activity_type}")
                plt.xlabel("Total Points")
                plt.ylabel("Challenge Name")
                try:
                    plt.gca().invert_yaxis()
                except Exception:
                    pass

            create_and_save_figure(
                plot_points_by_challenge_for_type,
                fname,
                figsize=(14, fig_h),
            )
    except Exception as e:
        logger.error(f"Error creating per-type challenge analyses: {e}")

    # Per-challenge: points by rule
    try:
        out_dir = ensure_dir(os.path.join(OUTPUT_VISUALIZATIONS_DIR, "by_challenge"))
        for challenge_name in rewards_df["challenge_name"].dropna().unique().tolist():
            sub = rewards_df[rewards_df["challenge_name"] == challenge_name]
            series = (
                sub.groupby("rule_name")["reward_points"]
                .sum()
                .sort_values(ascending=False)
            )
            if series.empty:
                continue

            fname = os.path.join(
                out_dir,
                f"points_by_rule_challenge_{safe_filename(challenge_name)}.png",
            )
            fig_h = compute_barh_fig_height(len(series))

            def plot_points_by_rule_for_challenge(
                series=series,
                challenge_name=challenge_name,
            ) -> None:
                series.plot(kind="barh", colormap=BAR_COLORMAP)
                plt.title(f"Total Rewarded Points by Rule — Challenge: {challenge_name}")
                plt.xlabel("Total Points")
                plt.ylabel("Rule Name")
                try:
                    plt.gca().invert_yaxis()
                except Exception:
                    pass

            create_and_save_figure(
                plot_points_by_rule_for_challenge,
                fname,
                figsize=(14, fig_h),
            )
    except Exception as e:
        logger.error(f"Error creating per-challenge rule analyses: {e}")


def save_activity_completion_plot(activity_counts: pd.Series) -> None:
    if activity_counts is None or activity_counts.empty:
        return

    def plot() -> None:
        plt.bar(range(len(activity_counts)), activity_counts.values)
        plt.xlabel("Activity Type")
        plt.ylabel("Count")
        plt.title("Activity Completion by Type")
        plt.xticks(range(len(activity_counts)), activity_counts.index, rotation=45, ha="right")
        plt.tight_layout()

    create_and_save_figure(
        plot,
        os.path.join(OUTPUT_VISUALIZATIONS_DIR, "activity_completion.png"),
        figsize=(14, 8),
    )


def save_activity_type_by_hour_heatmap_plot(activity_hour_counts: pd.DataFrame) -> None:
    if activity_hour_counts is None or activity_hour_counts.empty:
        return

    def plot() -> None:
        sns.heatmap(
            activity_hour_counts,
            cmap=SEQUENTIAL_HEATMAP_COLORMAP,
            annot=True,
            fmt="d",
            linewidths=0.2,
        )
        plt.title("Activity Types by Hour of Day")
        plt.xlabel("Activity Type")
        plt.ylabel("Hour of Day")

    create_and_save_figure(
        plot,
        os.path.join(OUTPUT_VISUALIZATIONS_DIR, "activity_type_by_hour.png"),
        figsize=(14, 8),
    )


def save_activity_type_by_day_heatmap_plot(activity_day_counts: pd.DataFrame) -> None:
    if activity_day_counts is None or activity_day_counts.empty:
        return

    def plot() -> None:
        sns.heatmap(
            activity_day_counts,
            cmap=SEQUENTIAL_HEATMAP_COLORMAP,
            annot=True,
            fmt="d",
            linewidths=0.2,
        )
        plt.title("Activity Types by Day of Week")
        plt.xlabel("Activity Type")
        plt.ylabel("Day of Week")

    create_and_save_figure(
        plot,
        os.path.join(OUTPUT_VISUALIZATIONS_DIR, "activity_type_by_day.png"),
        figsize=(14, 8),
    )


def save_active_players_per_day_plot(active_users_per_day: pd.Series) -> None:
    if active_users_per_day is None or active_users_per_day.empty:
        return

    series = active_users_per_day.sort_index()
    date_labels = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in series.index]

    def plot() -> None:
        x_pos = range(len(series))
        plt.bar(x_pos, series.values, width=0.6)
        plt.title("Number of Active Players per Day")
        plt.xlabel("Date")
        plt.ylabel("Number of Active Players")
        ax = plt.gca()
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(date_labels)
        ax.tick_params(axis="x", rotation=90)
        plt.margins(x=0.01)

    create_and_save_figure(
        plot,
        os.path.join(OUTPUT_VISUALIZATIONS_DIR, "active_players_per_day.png"),
        figsize=(12, 6),
    )


def save_tasks_by_provider_plot(provider_counts: pd.Series) -> None:
    if provider_counts is None or provider_counts.empty:
        return

    def plot() -> None:
        provider_counts.plot(kind="bar", colormap=BAR_COLORMAP)
        plt.title("Tasks Performed by Users per Data Provider")
        plt.xlabel("Data Provider")
        plt.ylabel("Task Completions (reward entries)")
        plt.xticks(rotation=45, ha="right")

    create_and_save_figure(
        plot,
        os.path.join(OUTPUT_VISUALIZATIONS_DIR, "tasks_by_provider.png"),
        figsize=(12, 6),
    )


def save_tasks_completed_per_day_plot(per_day: pd.Series) -> None:
    if per_day is None or per_day.empty:
        return

    series = per_day.sort_index()
    date_labels = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in series.index]

    def plot() -> None:
        x_pos = range(len(series))
        plt.bar(x_pos, series.values, width=0.6)
        plt.title("Tasks Completed per Day")
        plt.xlabel("Date")
        plt.ylabel("Tasks Completed (unique user-task)")
        ax = plt.gca()
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(date_labels)
        ax.tick_params(axis="x", rotation=90)
        plt.margins(x=0.01)

    create_and_save_figure(
        plot,
        os.path.join(OUTPUT_VISUALIZATIONS_DIR, "tasks_completed_per_day.png"),
        figsize=(12, 6),
    )


def save_tasks_completed_per_player_plot(per_user: pd.Series) -> None:
    if per_user is None or per_user.empty:
        return

    plot_data = per_user.head(50) if len(per_user) > 50 else per_user
    title = "Tasks Completed per Player (Top 50)" if len(per_user) > 50 else "Tasks Completed per Player"

    def plot() -> None:
        plot_data.plot(kind="bar", colormap=BAR_COLORMAP)
        plt.title(title)
        plt.xlabel("Player ID")
        plt.ylabel("Tasks Completed (unique tasks)")
        plt.xticks(rotation=45, ha="right")

    create_and_save_figure(
        plot,
        os.path.join(OUTPUT_VISUALIZATIONS_DIR, "tasks_completed_per_player.png"),
        figsize=(12, 6),
    )


def save_geofence_hourly_activity_plot(filtered: pd.DataFrame) -> None:
    if filtered is None or filtered.empty:
        return
    if "hour_of_day" not in filtered.columns:
        return

    hourly_counts = (
        filtered["hour_of_day"]
        .value_counts()
        .sort_index()
        .reindex(range(24), fill_value=0)
    )

    def plot() -> None:
        sns.barplot(x=list(hourly_counts.index), y=list(hourly_counts.values))
        plt.title("Geofence Activity by Hour of Day")
        plt.xlabel("Hour of Day")
        plt.ylabel("Number of Records")
        plt.xticks(range(0, 24))
        plt.xlim(-0.5, 23.5)
        plt.grid(True, alpha=0.3)

    create_and_save_figure(
        plot,
        os.path.join(OUTPUT_VISUALIZATIONS_DIR, "geofence_hourly_activity.png"),
        figsize=(12, 6),
    )


def save_geofence_speed_by_hour_plot(filtered: pd.DataFrame) -> None:
    if filtered is None or filtered.empty:
        return
    required = {"hour_of_day", "SPEED"}
    if not required.issubset(filtered.columns):
        return

    plot_data = filtered.dropna(subset=["hour_of_day", "SPEED"])
    if plot_data.empty:
        return

    def plot() -> None:
        sns.boxplot(data=plot_data, x="hour_of_day", y="SPEED", order=list(range(24)))
        plt.title("Speed by Hour of Day")
        plt.xlabel("Hour of Day")
        plt.ylabel("Speed")
        plt.xticks(range(0, 24))
        plt.xlim(-0.5, 23.5)
        plt.grid(True, alpha=0.3)

    create_and_save_figure(
        plot,
        os.path.join(OUTPUT_VISUALIZATIONS_DIR, "geofence_speed_by_hour.png"),
        figsize=(12, 6),
    )


def save_geofence_movement_trajectory_plot(trajectory_data: pd.DataFrame) -> None:
    if trajectory_data is None or trajectory_data.empty:
        return
    required = {"LONGITUDE", "LATITUDE", "SPEED"}
    if not required.issubset(trajectory_data.columns):
        return

    plot_data = trajectory_data.dropna(subset=["LONGITUDE", "LATITUDE"])
    if plot_data.empty:
        return

    def plot() -> None:
        plt.plot(plot_data["LONGITUDE"], plot_data["LATITUDE"], "b-", alpha=0.7)
        sc = plt.scatter(
            plot_data["LONGITUDE"],
            plot_data["LATITUDE"],
            c=plot_data["SPEED"],
            cmap="viridis",
            alpha=0.8,
        )
        plt.colorbar(sc, label="Speed")
        plt.title("Movement Trajectory (Color indicates Speed)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True, alpha=0.3)

    create_and_save_figure(
        plot,
        os.path.join(OUTPUT_VISUALIZATIONS_DIR, "geofence_movement_trajectory.png"),
        figsize=(12, 8),
    )


def save_geofence_3d_visualization_plot(filtered: pd.DataFrame) -> None:
    if filtered is None or filtered.empty:
        return
    required = {"LONGITUDE", "LATITUDE", "ALTITUDE", "SPEED"}
    if not required.issubset(filtered.columns):
        return

    plot_data = filtered.dropna(subset=["LONGITUDE", "LATITUDE", "ALTITUDE", "SPEED"])
    if plot_data.empty:
        return

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    def plot() -> None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            plot_data["LONGITUDE"],
            plot_data["LATITUDE"],
            plot_data["ALTITUDE"],
            c=plot_data["SPEED"],
            cmap="viridis",
            alpha=0.6,
        )
        plt.colorbar(scatter, label="Speed")
        ax.set_title("3D Geofence Visualization")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_zlabel("Altitude")

    create_and_save_figure(
        plot,
        os.path.join(OUTPUT_VISUALIZATIONS_DIR, "geofence_3d_visualization.png"),
        figsize=(12, 10),
    )

def save_active_passive_pie_chart(
    active_users_count: int,
    passive_users_count: int,
) -> None:
    total_users_considered = active_users_count + passive_users_count
    if total_users_considered <= 0:
        return

    labels = [f"Active ({active_users_count})", f"Passive ({passive_users_count})"]
    sizes = [active_users_count, passive_users_count]

    def plot() -> None:
        try:
            cmap = plt.get_cmap(PIE_COLORMAP)
            colors = [cmap(0), cmap(3)]
        except Exception:
            colors = None

        wedges, *_ = plt.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%" if sum(sizes) > 0 else None,
            colors=colors,
            startangle=90,
            wedgeprops={"edgecolor": "white"},
        )
        plt.title("Active vs Passive Players (by rewards)")
        plt.axis("equal")

        legend_labels = [
            "Active: players with any rewarded activity",
            "Passive: enrolled players with no rewards recorded",
        ]
        plt.legend(
            wedges,
            legend_labels,
            title="Legend",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=1,
            frameon=True,
        )

    create_and_save_figure(
        plot,
        os.path.join(OUTPUT_VISUALIZATIONS_DIR, "player_active_vs_passive_pie.png"),
        figsize=(8, 6),
    )


def save_steps_trend_plot(total: pd.DataFrame) -> None:
    if total is None or total.empty:
        return
    required = {"date", "steps"}
    if not required.issubset(total.columns):
        return

    plot_data = total.copy()
    plot_data["date"] = pd.to_datetime(plot_data["date"], errors="coerce")
    plot_data = plot_data.dropna(subset=["date"])

    if plot_data.empty:
        return

    ticks = pd.to_datetime(plot_data["date"])
    labels = [d.strftime("%Y-%m-%d") for d in ticks]

    def plot() -> None:
        ax = plt.gca()
        ax.plot(plot_data["date"], plot_data["steps"], marker="o")
        ax.set_title("Steps Trend (All Users)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Steps")
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=90, ha="center")
        if len(ticks) > 0:
            ax.set_xlim(ticks.iloc[0], ticks.iloc[-1])

    create_and_save_figure(
        plot,
        os.path.join(OUTPUT_VISUALIZATIONS_DIR, "steps_trend.png"),
        figsize=PLOT_FIGSIZE_ACTIVITY_TYPES_STACKED,
    )