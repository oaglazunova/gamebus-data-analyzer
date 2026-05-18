from __future__ import annotations

import re
import json
from typing import Tuple, Dict, Optional, Union, List
import pandas as pd
import logging

from src.analysis.common import OUTPUT_VISUALIZATIONS_DIR, MAX_TYPES_HEATMAP, ensure_output_dirs, logger, WEEKDAY_ORDER
from src.analysis.loaders import extract_detailed_rewards, load_excel_files, load_json_files
from src.analysis.reporting import generate_descriptive_stats, generate_descriptive_summary_text, create_complete_report
from src.analysis.activity_metrics import (
    APP_SCOPES,
    build_activities_frame,
    normalize_activity_columns,
    assign_campaign_wave,
    compute_campaign_metrics,
    compute_dropout_metrics,
    compute_joining_metrics,
    get_enrolled_user_ids,
    compute_reward_based_active_user_ids,
    compute_active_passive_by_types,
    compute_scope_usage_metrics,
    filter_activities_by_types,
    resolve_analysis_user_ids,
    filter_tabular_data_to_user_ids,
    filter_json_data_to_user_ids,
    ensure_user_email_mapping_exists,
    COMBINED_ACTIVE_TYPES,
    DROPOUT_INACTIVITY_DAYS,
    compute_real_dropout_and_weekly_retention,
)
from src.analysis.activity_plots import (
    save_activity_types_distribution_plot,
    save_activities_over_time_plot,
    save_points_by_activity_type_plot,
    save_points_by_player_plot,
    save_rewards_by_activity_type_plot,
    save_points_over_time_plot,
    save_activity_heatmap_by_time_plot,
    save_activity_type_by_player_heatmap,
    save_player_activity_distribution_plot,
    save_usage_by_day_of_week_plot,
    save_wave_comparisons_plot,
    save_wave_comparisons_by_activity_type_plot,
    save_wave_points_by_activity_type_plot,
    save_wave_comparisons_by_player_plot,
    save_activity_types_stacked_by_date_plot,
    save_player_engagement_heatmap_plot, save_dropout_rates_distribution_plot, save_joining_rates_distribution_plot,
    save_combined_dropout_joining_rates_plot, save_combined_dropout_joining_boxplots_plot,
    save_churn_counts_over_time_plot, save_churn_rate_over_time_plot, save_reward_challenge_rule_plots,
    save_activity_completion_plot, save_activity_type_by_hour_heatmap_plot, save_activity_type_by_day_heatmap_plot,
    save_active_players_per_day_plot, save_tasks_by_provider_plot, save_tasks_completed_per_day_plot,
    save_tasks_completed_per_player_plot, save_geofence_hourly_activity_plot, save_geofence_speed_by_hour_plot,
    save_geofence_movement_trajectory_plot, save_geofence_3d_visualization_plot, save_active_passive_pie_chart,
    save_steps_trend_plot, save_usage_by_hour_buckets_plot,
    save_usage_by_hour_plot,
    save_active_user_comparison_plot,
    save_real_dropout_status_plot,
    save_weekly_retention_plot,
)
from src.utils.logging import setup_logging, console_info, console_error
from src.analysis.geofence_analysis import analyze_geofence_data

# -----------------------------------------------------------------------------
# Activities analysis (core)
# -----------------------------------------------------------------------------
def analyze_activities(
    csv_data: Dict[str, pd.DataFrame],
) -> Optional[Tuple[pd.DataFrame, int, Optional[Dict], Optional[Dict], Optional[Dict]]]:
    try:
        activities = build_activities_frame(csv_data)
        activities = normalize_activity_columns(activities)
        activities["wave"] = assign_campaign_wave(activities, csv_data)
    except Exception as e:
        logger.error(f"Error processing activities data: {e}")
        return None

    # ------------------------------------------------------------------
    # Enrolled users base + reward-based active users (legacy comparison)
    # ------------------------------------------------------------------
    enrolled_user_ids: set = set()
    reward_based_active_user_ids: set = set()
    reward_based_passive_user_ids: List = []
    reward_based_active_users_count = 0
    reward_based_passive_users_count = 0

    # Per-scope metrics (GameBus / Nutrida / Combined)
    app_scope_metrics: Dict[str, Dict] = {}

    try:
        enrolled_user_ids = get_enrolled_user_ids(csv_data, activities)
    except Exception as e:
        logger.error(f"Error resolving enrolled participant cohort: {e}")
        return None

    try:
        reward_based_active_user_ids = compute_reward_based_active_user_ids(activities)
        reward_based_passive_user_ids = sorted(list(enrolled_user_ids - reward_based_active_user_ids))
        reward_based_active_users_count = len(reward_based_active_user_ids)
        reward_based_passive_users_count = len(reward_based_passive_user_ids)

        # Keep the old reward-based engagement label for backward compatibility
        reward_engagement_map = {
            pid: ("Active" if pid in reward_based_active_user_ids else "Passive")
            for pid in activities["pid"].dropna().unique()
        }
        activities["engagement_by_rewards"] = activities["pid"].map(reward_engagement_map)

    except Exception as e:
        logger.error(f"Error computing reward-based user metrics: {e}")

    # ------------------------------------------------------------------
    # Descriptor-based scope analysis
    # ------------------------------------------------------------------
    def _definition_note(scope_key: str) -> str:
        if scope_key == "gamebus":
            return (
                "GameBus active = >=1 of: DRINKING_DIARY, GENERAL_ACTIVITY, "
                "NUTRITION_DIARY, PHYSICAL_ACTIVITY, WALK, BIKE. "
                "Passive = enrolled but none of these."
            )
        if scope_key == "nutrida":
            return (
                "Nutrida active = >=1 of: PLAN_MEAL, NUTRITION_DIARY_VID. "
                "Passive = enrolled but none of these."
            )
        return (
            "Combined active = >=1 of the GameBus or Nutrida qualifying activities. "
            "Passive = enrolled but none in either app."
        )

    try:
        descriptor_active_counts_for_comparison: Dict[str, int] = {}

        for scope_key, scope_cfg in APP_SCOPES.items():
            scope_label = scope_cfg["label"]

            try:
                # Active/passive by descriptor list (new main method)
                active_passive = compute_active_passive_by_types(
                    activities=activities,
                    enrolled_user_ids=enrolled_user_ids,
                    active_types=scope_cfg["active_types"],
                )

                # Scope usage metrics
                usage_metrics = compute_scope_usage_metrics(
                    activities=activities,
                    enrolled_user_ids=enrolled_user_ids,
                    metric_types=scope_cfg["metric_types"],
                    usage_time_types=scope_cfg["usage_time_types"],
                )

                descriptor_active_counts_for_comparison[f"{scope_label} (descriptor-based)"] = (
                    active_passive["active_users_count"]
                )

                # Scope-specific plots
                try:
                    save_active_passive_pie_chart(
                        active_passive["active_users_count"],
                        active_passive["passive_users_count"],
                        scope_label=scope_label,
                        definition_note=_definition_note(scope_key),
                    )
                except Exception as e:
                    logger.error(f"Error plotting active/passive pie for {scope_label}: {e}")

                try:
                    save_usage_by_day_of_week_plot(
                        usage_metrics["usage_by_day_of_week"],
                        scope_label=scope_label,
                    )
                except Exception as e:
                    logger.error(f"Error plotting usage by day of week for {scope_label}: {e}")

                try:
                    save_usage_by_hour_buckets_plot(
                        usage_metrics["usage_by_hour_buckets"],
                        scope_label=scope_label,
                    )
                except Exception as e:
                    logger.error(f"Error plotting usage by hour buckets for {scope_label}: {e}")

                try:
                    save_usage_by_hour_plot(
                        usage_metrics["usage_by_hour"],
                        scope_label=scope_label,
                    )
                except Exception as e:
                    logger.error(f"Error plotting usage by hour for {scope_label}: {e}")

                try:
                    save_active_players_per_day_plot(
                        usage_metrics["active_players_per_day"],
                        scope_label=scope_label,
                    )
                except Exception as e:
                    logger.error(f"Error plotting active players per day for {scope_label}: {e}")

                # Heatmap: day-of-week x hour (timing analysis uses usage_time_types only)
                try:
                    usage_df = activities[activities["type"].isin(scope_cfg["usage_time_types"])].copy()
                    if not usage_df.empty:
                        usage_df["day_of_week"] = usage_df["createdAt"].dt.day_name()

                        activity_heatmap_data = pd.crosstab(
                            index=usage_df["day_of_week"],
                            columns=usage_df["hour"],
                        ).fillna(0)

                        activity_heatmap_data = activity_heatmap_data.reindex(WEEKDAY_ORDER).fillna(0)
                        activity_heatmap_data = activity_heatmap_data.reindex(columns=range(24), fill_value=0)

                        save_activity_heatmap_by_time_plot(
                            activity_heatmap_data,
                            scope_label=scope_label,
                        )
                except Exception as e:
                    logger.error(f"Error creating activity heatmap by time for {scope_label}: {e}")

                # Store for report
                app_scope_metrics[scope_key] = {
                    "label": scope_label,
                    "active_users_count": active_passive["active_users_count"],
                    "passive_users_count": active_passive["passive_users_count"],
                    "active_user_ids": sorted(list(active_passive["active_user_ids"])),
                    "passive_user_ids": sorted(list(active_passive["passive_user_ids"])),

                    "avg_active_days_per_participant": usage_metrics["avg_active_days_per_participant"],
                    "median_active_days_per_participant": usage_metrics["median_active_days_per_participant"],
                    "std_active_days_per_participant": usage_metrics["std_active_days_per_participant"],
                    "min_active_days_per_participant": usage_metrics["min_active_days_per_participant"],
                    "max_active_days_per_participant": usage_metrics["max_active_days_per_participant"],

                    "avg_active_players_per_day": usage_metrics["avg_active_players_per_day"],
                    "median_active_players_per_day": usage_metrics["median_active_players_per_day"],
                    "std_active_players_per_day": usage_metrics["std_active_players_per_day"],
                    "min_active_players_per_day": usage_metrics["min_active_players_per_day"],
                    "max_active_players_per_day": usage_metrics["max_active_players_per_day"],

                    "avg_time_to_first_inactivity_days": usage_metrics["avg_time_to_first_inactivity_days"],
                    "median_time_to_first_inactivity_days": usage_metrics["median_time_to_first_inactivity_days"],
                    "std_time_to_first_inactivity_days": usage_metrics["std_time_to_first_inactivity_days"],
                    "min_time_to_first_inactivity_days": usage_metrics["min_time_to_first_inactivity_days"],
                    "max_time_to_first_inactivity_days": usage_metrics["max_time_to_first_inactivity_days"],

                    "peak_activity_hour": usage_metrics["peak_activity_hour"],

                    # Raw series for reporting if needed
                    "active_players_per_day_series": usage_metrics["active_players_per_day"],
                    "usage_by_day_of_week_series": usage_metrics["usage_by_day_of_week"],
                    "usage_by_hour_series": usage_metrics["usage_by_hour"],
                    "usage_by_hour_buckets_series": usage_metrics["usage_by_hour_buckets"],
                }

            except Exception as e:
                logger.error(f"Error computing scope metrics for {scope_label}: {e}")

        # Comparison plot: descriptor-based vs reward-based
        try:
            comparison_counts = dict(descriptor_active_counts_for_comparison)
            comparison_counts["Reward-based (legacy)"] = reward_based_active_users_count
            save_active_user_comparison_plot(comparison_counts)
        except Exception as e:
            logger.error(f"Error plotting descriptor-vs-reward active user comparison: {e}")

    except Exception as e:
        logger.error(f"Error computing descriptor-based scope analysis: {e}")

    # Descriptive stats (drop 'properties' if present)
    try:
        activities_for_stats = activities.copy()
        if "properties" in activities_for_stats.columns:
            activities_for_stats = activities_for_stats.drop(columns=["properties"])
        generate_descriptive_stats(activities_for_stats, "Activities Data", None)
        del activities_for_stats
    except Exception as e:
        logger.error(f"Error generating descriptive statistics: {e}")

    # Basic plots: distribution, daily counts, points by type
    try:
        activity_counts = activities["type"].value_counts()
        save_activity_types_distribution_plot(activity_counts)
    except Exception as e:
        logger.error(f"Error creating activity types distribution visualization: {e}")

    # Wave comparisons
    try:
        save_wave_comparisons_plot(activities)
        save_wave_comparisons_by_activity_type_plot(activities)
        save_wave_points_by_activity_type_plot(activities)
        save_wave_comparisons_by_player_plot(activities)
    except Exception as e:
        logger.error(f"Error creating wave comparisons: {e}")

    # Daily activities time series
    try:
        daily_activities = activities.groupby("date").size()
        save_activities_over_time_plot(daily_activities)
    except Exception as e:
        logger.error(f"Error creating activities over time visualization: {e}")

    # Points by activity type (total) -- unchanged
    try:
        points_by_type = activities.groupby("type")["points"].sum().sort_values(ascending=False)
        save_points_by_activity_type_plot(points_by_type)
    except Exception as e:
        logger.error(f"Error creating points by activity type visualization: {e}")

    # Points by player -- unchanged
    try:
        points_by_user = activities.groupby("pid")["points"].sum().sort_values(ascending=False)
        save_points_by_player_plot(points_by_user)
    except Exception as e:
        logger.error(f"Error creating points by user visualization: {e}")

    # Rewards by activity type (mean) -- unchanged
    try:
        rewards_by_type = activities.groupby("type")["points"].mean().sort_values(ascending=False)
        save_rewards_by_activity_type_plot(rewards_by_type)
    except Exception as e:
        logger.error(f"Error creating rewards by activity type visualization: {e}")

    # Points over time -- unchanged
    try:
        daily_points = activities.groupby("date")["points"].sum()
        save_points_over_time_plot(daily_points)
    except Exception as e:
        logger.error(f"Error creating points over time visualization: {e}")

    # Reward-level analysis by challenge and rule (explode detailed_rewards)
    try:
        rewards_df = activities[["pid", "date", "type", "detailed_rewards"]].copy()
        rewards_df = rewards_df.explode("detailed_rewards")
        rewards_df = rewards_df[rewards_df["detailed_rewards"].apply(lambda x: isinstance(x, dict))]

        if not rewards_df.empty:
            def _extract_reward_points(r):
                try:
                    return int(r.get("points", 0)) if isinstance(r, dict) else 0
                except Exception:
                    return 0

            def _extract_challenge(r):
                name, cid = None, None
                if isinstance(r, dict):
                    ch = r.get("challenge")
                    if isinstance(ch, dict):
                        name = ch.get("name") or ch.get("label")
                        cid = ch.get("xid") or ch.get("id")
                    if name is None:
                        for key in ("challengeName", "name", "label"):
                            if key in r and pd.notna(r.get(key)):
                                name = r.get(key)
                                break
                    if cid is None:
                        for key in ("challengeId", "challenge_id", "cid", "id", "xid"):
                            if key in r and pd.notna(r.get(key)):
                                cid = r.get(key)
                                break
                if isinstance(cid, list) and cid:
                    cid = cid[0]
                return name, cid

            def _extract_rule(r):
                if isinstance(r, dict):
                    val = r.get("rule")
                    if isinstance(val, (str, int, float)):
                        return str(val)
                return None

            rewards_df["reward_points"] = rewards_df["detailed_rewards"].apply(_extract_reward_points)
            rewards_df[["challenge_name", "challenge_id"]] = rewards_df["detailed_rewards"].apply(
                lambda r: pd.Series(_extract_challenge(r))
            )
            rewards_df["rule_name"] = rewards_df["detailed_rewards"].apply(_extract_rule)

            rewards_df["challenge_name"] = rewards_df["challenge_name"].fillna("Unknown")
            rewards_df["rule_name"] = rewards_df["rule_name"].fillna("Unknown")

            save_reward_challenge_rule_plots(rewards_df)

    except Exception as e:
        logger.error(f"Error creating challenge/rule analyses: {e}")

    # User activity distribution
    try:
        user_activity = activities.groupby("pid").size().sort_values(ascending=False)
        save_player_activity_distribution_plot(user_activity)
    except Exception as e:
        logger.error(f"Error creating user activity distribution visualization: {e}")

    # Heatmap: activity type distribution by player (guarded)
    try:
        if activities["pid"].nunique() >= 2 and activities["type"].nunique() >= 2:
            user_type_counts = pd.crosstab(activities["pid"], activities["type"])
            if not user_type_counts.empty:
                row_sums = user_type_counts.sum(axis=1)
                user_type_counts = user_type_counts[row_sums > 0]
                user_type_counts_norm = user_type_counts.div(user_type_counts.sum(axis=1), axis=0)

                heat = user_type_counts_norm.copy()
                if heat.shape[1] > MAX_TYPES_HEATMAP:
                    top_types = user_type_counts.sum().nlargest(MAX_TYPES_HEATMAP).index
                    heat = heat[top_types]

                save_activity_type_by_player_heatmap(heat)
    except Exception as e:
        logger.error(f"Error creating activity type by user heatmap: {e}")

    # Campaign metrics
    campaign_metrics: Optional[Dict] = None
    unique_users_count = 0

    try:
        campaign_metrics = compute_campaign_metrics(activities, csv_data)
        unique_users_count = int(campaign_metrics.get("unique_users", 0))

        campaign_metrics["enrolled_users_count"] = len(enrolled_user_ids)
        campaign_metrics["app_scope_metrics"] = app_scope_metrics

        # Backward compatibility: top-level active/passive now means Combined descriptor-based
        combined_scope = app_scope_metrics.get("combined", {})
        campaign_metrics["active_users_count"] = int(combined_scope.get("active_users_count", 0))
        campaign_metrics["passive_users_count"] = int(combined_scope.get("passive_users_count", 0))
        campaign_metrics["passive_user_ids"] = combined_scope.get("passive_user_ids", [])

        # Legacy comparison (reward-based)
        campaign_metrics["reward_based_active_users_count"] = reward_based_active_users_count
        campaign_metrics["reward_based_passive_users_count"] = reward_based_passive_users_count
        campaign_metrics["reward_based_passive_user_ids"] = reward_based_passive_user_ids

        # Real dropout and weekly retention.
        #
        # This uses Combined qualifying activity types, not DAY_AGGREGATE.
        # That means it measures engagement dropout/retention rather than passive
        # sensor-sync continuity.
        try:
            retention_result = compute_real_dropout_and_weekly_retention(
                activities=activities,
                csv_data=csv_data,
                enrolled_user_ids=enrolled_user_ids,
                active_types=COMBINED_ACTIVE_TYPES,
                inactivity_days=DROPOUT_INACTIVITY_DAYS,
            )

            campaign_metrics["real_dropout_metrics"] = retention_result["metrics"]
            campaign_metrics["weekly_retention"] = retention_result["weekly_retention"]

            save_real_dropout_status_plot(retention_result["user_dropout"])
            save_weekly_retention_plot(retention_result["weekly_retention"])

        except Exception as e:
            logger.error(f"Error calculating real dropout / weekly retention metrics: {e}")

        try:
            generate_descriptive_summary_text(activities, csv_data, campaign_metrics)
        except Exception as e:
            logger.error(f"Error generating descriptive summary text: {e}")

    except Exception as e:
        logger.error(f"Error calculating campaign metrics: {e}")

    # Dropout metrics (last - first activity, days)
    dropout_metrics: Optional[Dict] = None
    user_dropout = pd.DataFrame()

    try:
        dropout_result = compute_dropout_metrics(activities)
        dropout_metrics = dropout_result["metrics"]
        user_dropout = dropout_result["user_dropout"]

        if dropout_metrics is not None and not user_dropout.empty:
            save_dropout_rates_distribution_plot(user_dropout)

    except Exception as e:
        logger.error(f"Error calculating dropout metrics: {e}")

    # Joining metrics (first activity - campaign start)
    joining_metrics: Optional[Dict] = None
    try:
        joining_result = compute_joining_metrics(activities, csv_data)
        joining_metrics = joining_result["metrics"]

        user_dropout = joining_result["user_dropout"]

        if joining_metrics is not None and not user_dropout.empty:
            save_joining_rates_distribution_plot(user_dropout)
            save_combined_dropout_joining_rates_plot(user_dropout)
            save_combined_dropout_joining_boxplots_plot(user_dropout)

    except Exception as e:
        logger.error(f"Error calculating joining metrics: {e}")

    # Churn rate (30-day inactivity)
    try:
        daily = activities.copy()
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
        daily = daily[daily["pid"].notna()].copy()
        daily["pid_str"] = daily["pid"].astype(str)

        series_start = daily["date"].min()
        series_end = daily["date"].max()
        if pd.isna(series_start) or pd.isna(series_end):
            raise ValueError("Invalid date range for churn computation")

        date_index = pd.date_range(series_start, series_end, freq="D")

        active_pivot = (
            daily.groupby(["date", "pid_str"]).size().unstack(fill_value=0).reindex(index=date_index, fill_value=0)
        )
        if active_pivot.shape[1] == 0:
            raise ValueError("No users for churn computation")

        active_bool = active_pivot > 0

        first_activity_series = daily.groupby("pid_str")["date"].min().reindex(active_bool.columns)
        first_activity_series = pd.to_datetime(first_activity_series)

        idx_vals = active_bool.index.values.astype("datetime64[ns]")
        fa_vals = first_activity_series.values.astype("datetime64[ns]")
        joined_mask = pd.DataFrame(
            idx_vals[:, None] >= fa_vals[None, :],
            index=active_bool.index,
            columns=active_bool.columns,
        )

        active_last_30 = active_bool.rolling(window=30, min_periods=1).sum() > 0
        churned_mask = (~active_last_30) & joined_mask

        churned_count = churned_mask.sum(axis=1)
        joined_count = joined_mask.sum(axis=1)
        denom = joined_count.replace(0, pd.NA).astype("Float64")
        churn_rate = (churned_count.astype("Float64") / denom).fillna(0.0)

        churn_df = pd.DataFrame(
            {
                "date": churn_rate.index,
                "churned_count": churned_count.values,
                "joined_count": joined_count.values,
                "churn_rate": churn_rate.values,
            }
        ).set_index("date")

        save_churn_rate_over_time_plot(churn_df)
        save_churn_counts_over_time_plot(churn_df)

    except Exception as e:
        logger.error(f"Error creating churn rate visualization: {e}")

    # Stacked bar: top activity types by date (daily)
    try:
        save_activity_types_stacked_by_date_plot(activities)
    except Exception as e:
        logger.error(f"Error creating stacked bar chart by date: {e}")

    # User engagement heatmap by day (guarded)
    try:
        save_player_engagement_heatmap_plot(activities)
    except Exception as e:
        logger.error(f"Error creating user engagement heatmap: {e}")

    return activities, unique_users_count, dropout_metrics, joining_metrics, campaign_metrics


# -----------------------------------------------------------------------------
# Visualizations / challenges / tasks analysis (desc_* + rewards mapping)
# -----------------------------------------------------------------------------
def analyze_visualizations_challenges_tasks(csv_data: Dict[str, pd.DataFrame]) -> None:
    required_sheets = ["desc_visualizations", "desc_challenges", "desc_tasks", "activities"]
    missing_sheets = [s for s in required_sheets if s not in csv_data]
    if missing_sheets:
        logger.warning(f"Skipping visualization/challenge/task analysis; missing sheets: {missing_sheets}")
        return

    tasks_df = csv_data["desc_tasks"].copy()
    activities_df = csv_data["activities"].copy()

    # Ensure createdAt fields
    if "createdAt" in activities_df.columns and not pd.api.types.is_datetime64_any_dtype(activities_df["createdAt"]):
        activities_df["createdAt"] = pd.to_datetime(activities_df["createdAt"], errors="coerce", utc=True)
    if "date" not in activities_df.columns and "createdAt" in activities_df.columns:
        activities_df["date"] = activities_df["createdAt"].dt.date

    # Activity completion by type
    try:
        if "type" in activities_df.columns:
            activity_counts = activities_df["type"].value_counts()
            save_activity_completion_plot(activity_counts)
    except Exception as e:
        logger.error(f"Error creating activity completion plot: {e}")

    # Heatmaps by hour/day
    try:
        if "createdAt" in activities_df.columns and "type" in activities_df.columns:
            activities_df["hour"] = activities_df["createdAt"].dt.hour
            activities_df["day_of_week"] = activities_df["createdAt"].dt.day_name()
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

            activity_hour_counts = (
                pd.crosstab(activities_df["hour"], activities_df["type"])
                .reindex(range(24), fill_value=0)
            )
            save_activity_type_by_hour_heatmap_plot(activity_hour_counts)

            activity_day_counts = (
                pd.crosstab(activities_df["day_of_week"], activities_df["type"])
                .reindex(day_order)
                .fillna(0)
            )
            save_activity_type_by_day_heatmap_plot(activity_day_counts)

    except Exception as e:
        logger.error(f"Error creating hour/day heatmaps: {e}")

    # # Active users per day
    # try:
    #     if "pid" in activities_df.columns and "date" in activities_df.columns:
    #         active_users_per_day = activities_df.groupby("date")["pid"].nunique().sort_index()
    #         save_active_players_per_day_plot(active_users_per_day)
    # except Exception as e:
    #     logger.error(f"Error plotting active users per day: {e}")

    # Completed tasks analysis (rules from rewardedParticipations -> provider mapping)
    try:
        if "rewardedParticipations" not in activities_df.columns or "pid" not in activities_df.columns:
            logger.warning("Skipping completed task plots; missing rewardedParticipations or pid")
            return

        tmp = activities_df[["pid", "type", "date", "rewardedParticipations"]].copy()
        tmp["detailed_rewards"] = tmp["rewardedParticipations"].apply(extract_detailed_rewards)
        tmp = tmp.explode("detailed_rewards").dropna(subset=["detailed_rewards"])
        tmp = tmp[tmp["detailed_rewards"].apply(lambda x: isinstance(x, dict))]

        tmp["task_name"] = tmp["detailed_rewards"].apply(lambda r: str(r.get("rule")).strip() if r.get("rule") is not None else None)
        tmp = tmp.dropna(subset=["task_name"])

        task_completions_df = tmp[["pid", "task_name", "type", "date"]].rename(columns={"type": "activity_type"})
        if task_completions_df.empty:
            logger.warning("No task completion records (rules) found in rewards")
            return

        unique_task_completions = task_completions_df.drop_duplicates(subset=["pid", "task_name"])

        # tasks_by_provider mapping via desc_tasks
        try:
            col_map = {c.lower(): c for c in tasks_df.columns}
            name_col = col_map.get("name") or col_map.get("label") or col_map.get("rule") or col_map.get("task") or col_map.get("task_name")
            providers_col = col_map.get("dataproviders")

            if name_col and providers_col:
                def _norm_name(s):
                    return re.sub(r"\s+", " ", str(s)).strip().lower()

                def _parse_providers(val):
                    if pd.isna(val):
                        return []
                    if isinstance(val, list):
                        return [str(x).strip() for x in val if str(x).strip()]
                    s = str(val).strip()
                    try:
                        js = re.sub("'", '"', s)
                        parsed = json.loads(js) if (js.startswith("[") and js.endswith("]")) else None
                        if isinstance(parsed, list):
                            return [str(x).strip() for x in parsed if str(x).strip()]
                    except Exception:
                        pass
                    parts = re.split(r"[;,]", s)
                    return [p.strip() for p in parts if p.strip()]

                tasks_df["task_name_norm"] = tasks_df[name_col].apply(_norm_name)
                tasks_df["providers_list"] = tasks_df[providers_col].apply(_parse_providers)

                providers_map = (
                    tasks_df.groupby("task_name_norm")["providers_list"]
                    .apply(lambda s: sorted({p for lst in s for p in (lst or [])}))
                    .to_dict()
                )

                task_completions_df["task_name_norm"] = task_completions_df["task_name"].apply(_norm_name)
                task_completions_df["providers_list"] = task_completions_df["task_name_norm"].map(providers_map)

                mapped = task_completions_df.dropna(subset=["providers_list"]).explode("providers_list").rename(columns={"providers_list": "provider"})
                provider_counts = mapped.dropna(subset=["provider"]).groupby("provider").size().sort_values(ascending=False)

                if not provider_counts.empty:
                    save_tasks_by_provider_plot(provider_counts)

        except Exception as e:
            logger.error(f"Error creating tasks_by_provider: {e}")


        # tasks completed per day (unique user-task per day)
        try:
            if "date" in task_completions_df.columns:
                per_day = (
                    task_completions_df
                    .dropna(subset=["date"])
                    .drop_duplicates(subset=["pid", "task_name", "date"])
                    .groupby("date")
                    .size()
                    .sort_index()
                )
                save_tasks_completed_per_day_plot(per_day)
        except Exception as e:
            logger.error(f"Error creating tasks_completed_per_day plot: {e}")


        # tasks completed per user (unique tasks)
        try:
            per_user = unique_task_completions.groupby("pid").size().sort_values(ascending=False)
            save_tasks_completed_per_player_plot(per_user)
        except Exception as e:
            logger.error(f"Error creating tasks_completed_per_player plot: {e}")

    except Exception as e:
        logger.error(f"Error generating completed tasks analysis: {e}")





# -----------------------------------------------------------------------------
# Daily steps analysis (DAY_AGGREGATE + variants)
# -----------------------------------------------------------------------------
def analyze_day_aggregate_steps(
    json_data: Dict[str, Union[pd.DataFrame, Dict]],
    campaign_metrics: Optional[Dict] = None,
) -> Optional[pd.DataFrame]:
    try:
        records: List[pd.DataFrame] = []

        def _extract_from_df(df: pd.DataFrame, user_id: str) -> Optional[pd.DataFrame]:
            if df is None or df.empty:
                return None
            local = df.copy()

            date_candidates = ["X_DATE", "DATE", "START_DATE", "END_DATE"]
            date_col = next((c for c in date_candidates if c in local.columns), None)
            if date_col is None:
                return None

            local["__date"] = pd.to_datetime(local[date_col], errors="coerce").dt.date
            if local["__date"].isna().all() and "END_DATE" in local.columns and date_col != "END_DATE":
                local["__date"] = pd.to_datetime(local["END_DATE"], errors="coerce").dt.date

            steps_candidates = ["STEPS_SUM", "STEP_SUM", "STEPS"]
            steps_col = next((c for c in steps_candidates if c in local.columns), None)
            if steps_col is None:
                return None

            out = local[["__date", steps_col]].dropna(subset=["__date"]).rename(columns={steps_col: "steps"})
            out["steps"] = pd.to_numeric(out["steps"], errors="coerce").fillna(0)
            if out.empty:
                return None
            out["user_id"] = str(user_id)
            return out

        for key, data in json_data.items():
            k_lower = key.lower()
            m = re.match(r"(?:player|user)_(\d+)_", key)
            user_id = m.group(1) if m else key

            if isinstance(data, pd.DataFrame):
                if "day_aggregate" in k_lower:
                    df_out = _extract_from_df(data, user_id)
                    if df_out is not None:
                        records.append(df_out)

            elif isinstance(data, dict) and (k_lower.startswith("player_") or k_lower.startswith("user_")) and k_lower.endswith("_all_data"):
                for section in ["day_aggregate", "day_aggregate_walk", "day_aggregate_run"]:
                    section_data = data.get(section)
                    if isinstance(section_data, list) and section_data:
                        try:
                            df_section = pd.DataFrame(section_data)
                            df_out = _extract_from_df(df_section, user_id)
                            if df_out is not None:
                                records.append(df_out)
                        except Exception as e:
                            logger.warning(f"Failed parsing section '{section}' for {key}: {e}")

        if not records:
            logger.warning("No DAY_AGGREGATE step data found; skipping steps plots")
            return None

        combined = pd.concat(records, ignore_index=True)
        combined = combined.groupby(["__date", "user_id"], as_index=False)["steps"].sum().rename(columns={"__date": "date"})
        combined["steps"] = pd.to_numeric(combined["steps"], errors="coerce").fillna(0)
        combined = combined.sort_values(["date", "user_id"])

        inferred = combined.groupby("date", as_index=False)["steps"].sum()

        def _parse_ts(val):
            return pd.to_datetime(val, errors="coerce")

        min_ts = _parse_ts(inferred["date"]).min()
        max_ts = _parse_ts(inferred["date"]).max()

        start_ts = _parse_ts((campaign_metrics or {}).get("start_date")) if campaign_metrics else pd.NaT
        end_ts = _parse_ts((campaign_metrics or {}).get("end_date")) if campaign_metrics else pd.NaT
        if pd.isna(start_ts):
            start_ts = min_ts
        if pd.isna(end_ts):
            end_ts = max_ts
        if pd.isna(start_ts) or pd.isna(end_ts) or start_ts > end_ts:
            start_ts, end_ts = min_ts, max_ts

        full_range = pd.date_range(start=start_ts.normalize(), end=end_ts.normalize(), freq="D")

        total = inferred.copy()
        total["date"] = pd.to_datetime(total["date"], errors="coerce")
        total = total.set_index("date").reindex(full_range).fillna(0).rename_axis("date").reset_index()

        save_steps_trend_plot(total)

        return combined

    except Exception as e:
        logger.error(f"Error analyzing DAY_AGGREGATE steps: {e}")
        return None


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    # If analysis is run directly (not via pipeline), configure logging here
    if not logging.getLogger().handlers:
        setup_logging(log_type="analysis")

    console_info(logger, "[ANALYZE] Loading input files")

    try:
        ensure_output_dirs()

        # Load data
        try:
            csv_data = load_excel_files()
            logger.info(f"Loaded {len(csv_data)} Excel sheets" if csv_data else "No Excel data loaded")
        except Exception as e:
            logger.error(f"Error loading Excel files: {e}")
            csv_data = {}

        try:
            json_data = load_json_files()
            logger.info(f"Loaded {len(json_data)} JSON files" if json_data else "No JSON data loaded")
        except Exception as e:
            logger.error(f"Error loading JSON files: {e}")
            json_data = {}

        console_info(logger, f"[ANALYZE] Loaded {len(csv_data)} Excel sheet(s)")
        console_info(logger, f"[ANALYZE] Loaded {len(json_data)} JSON file(s)")

        # Resolve the authoritative participant cohort from users.xlsx.
        # If user_email_mapping.txt is missing or empty, generate it automatically.
        try:
            ensure_user_email_mapping_exists()

            participant_ids = resolve_analysis_user_ids()

            csv_data = filter_tabular_data_to_user_ids(csv_data, participant_ids)
            json_data = filter_json_data_to_user_ids(json_data, participant_ids)

            logger.info(
                f"Resolved authoritative analysis cohort from users.xlsx: "
                f"{len(participant_ids)} participant player ID(s)"
            )
            console_info(
                logger,
                f"[ANALYZE] Participant cohort resolved from users.xlsx: "
                f"{len(participant_ids)} user(s)",
            )

        except Exception as e:
            logger.error(f"Could not resolve participant cohort: {e}", exc_info=True)
            console_error(
                logger,
                "[ANALYZE] Could not resolve participant cohort from users.xlsx. "
                "Analysis stopped to avoid including test accounts.",
            )
            return

        # Analyze activities
        try:
            result = analyze_activities(csv_data)
            if result:
                activities, unique_users_count, dropout_metrics, joining_metrics, campaign_metrics = result
                logger.info(f"Activities analyzed: {unique_users_count} unique active users")
            else:
                result = None
                activities = None
                campaign_metrics = None
        except Exception as e:
            logger.error(f"Error analyzing activities data: {e}")
            result = None
            activities = None
            campaign_metrics = None

        # Analyze geofence
        try:
            analyze_geofence_data(json_data)
        except Exception as e:
            logger.error(f"Error analyzing geofence data: {e}")

        # Analyze daily steps
        try:
            analyze_day_aggregate_steps(json_data, campaign_metrics)
        except Exception as e:
            logger.error(f"Error analyzing daily steps: {e}")

        # Analyze desc_visualizations / challenges / tasks
        try:
            analyze_visualizations_challenges_tasks(csv_data)
        except Exception as e:
            logger.error(f"Error analyzing visualizations/challenges/tasks: {e}")

        if result:
            console_info(logger, f"[ANALYZE] Activities analyzed | unique active users={unique_users_count}")
        else:
            console_info(logger, "[ANALYZE] Activities analysis skipped or produced no result")

        # Report
        try:
            report_result = result if result else activities
            report_path = create_complete_report(csv_data, json_data, report_result)
            logger.info(f"Report written to: {report_path}")
            console_info(logger, f"[ANALYZE] Report written: {report_path}")
        except Exception:
            logger.error("Error generating report", exc_info=True)
            console_info(logger, "[ANALYZE] Report generation failed; see log file")

        console_info(logger, f"[ANALYZE] Outputs saved to: {OUTPUT_VISUALIZATIONS_DIR}")
        console_info(logger, "[ANALYZE] Completed")


    except Exception:
        logger.error("Unexpected error in main()", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console_info(logger, "[ANALYZE] Interrupted by user")
    except Exception:
        console_error(logger, "[ANALYZE] Failed; see log file")
