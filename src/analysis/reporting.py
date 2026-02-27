from __future__ import annotations

import os
import re
import json
from typing import Tuple, Dict, Optional, Union, List
import pandas as pd

from config.paths import PROJECT_ROOT
from src.analysis.common import ensure_dir, safe_filename, OUTPUT_VISUALIZATIONS_DIR, _mean_sd, _fmt_pct, _bucket_hour, _fmt_mean_sd, logger
from src.analysis.loaders import extract_points, extract_detailed_rewards



def generate_descriptive_stats(
    df: pd.DataFrame, title: str, filename: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Robust descriptive stats generation.
    Handles object columns that may contain unhashable values (e.g., lists/dicts).
    """
    stats_dir = ensure_dir(os.path.join(OUTPUT_VISUALIZATIONS_DIR, "statistics"))
    base = safe_filename(filename or title)

    out: Dict[str, pd.DataFrame] = {}

    # Numerical
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if num_cols:
        num_stats = df[num_cols].describe(percentiles=[0.25, 0.5, 0.75, 0.9]).T
        num_stats["median"] = df[num_cols].median(numeric_only=True)
        num_stats["variance"] = df[num_cols].var(numeric_only=True)
        num_stats["skew"] = df[num_cols].skew(numeric_only=True)
        num_stats["kurtosis"] = df[num_cols].kurtosis(numeric_only=True)
        num_stats = num_stats.round(2)
        out["numerical"] = num_stats
        num_stats.to_csv(os.path.join(stats_dir, f"{base}_numerical.csv"), index=True)

    # Helper: stringify values to avoid "unhashable type: list" in nunique/value_counts
    def _stable_str(v) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        try:
            # JSON stringify dict/list in a stable manner
            if isinstance(v, (dict, list)):
                return json.dumps(v, sort_keys=True, ensure_ascii=False)
            # sets -> sorted list
            if isinstance(v, set):
                return json.dumps(sorted(list(v)), ensure_ascii=False)
            return str(v)
        except Exception:
            return str(v)

    # Categorical / other
    cat_cols = [c for c in df.columns if c not in num_cols]
    if cat_cols:
        rows = []
        for c in cat_cols:
            s = df[c]
            missing = int(s.isna().sum())
            n = int(len(s))

            # Convert to a safe, hashable representation for nunique + top
            s_safe = s.dropna().map(_stable_str)

            nunique = int(s_safe.nunique(dropna=True))

            top = s_safe.value_counts().head(1)
            top_val = top.index[0] if len(top) else None
            top_cnt = int(top.iloc[0]) if len(top) else 0

            rows.append(
                {
                    "column": c,
                    "n": n,
                    "missing": missing,
                    "missing_pct": round((missing / n * 100.0), 2) if n else 0.0,
                    "nunique": nunique,
                    "top": top_val,
                    "top_count": top_cnt,
                }
            )

        cat_stats = pd.DataFrame(rows).set_index("column")
        out["categorical"] = cat_stats
        cat_stats.to_csv(os.path.join(stats_dir, f"{base}_categorical.csv"), index=True)

    logger.info(f"Saved descriptive stats ({list(out.keys())}) -> {stats_dir}")
    return out


# -----------------------------------------------------------------------------
# Descriptive summary text (campaign)
# -----------------------------------------------------------------------------
def generate_descriptive_summary_text(
    activities: pd.DataFrame,
    csv_data: Dict[str, pd.DataFrame],
    campaign_metrics: Optional[Dict] = None,
    out_path: Optional[str] = None,
) -> str:
    out_dir = ensure_dir(OUTPUT_VISUALIZATIONS_DIR)
    if out_path is None:
        out_path = os.path.join(out_dir, "descriptive_summary.txt")

    if activities is None or not isinstance(activities, pd.DataFrame) or activities.empty:
        txt = "No activities data available; cannot generate descriptive summary.\n"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(txt)
        return out_path

    df = activities.copy()

    # Ensure datetime + derived fields
    if "createdAt" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["createdAt"]):
        df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce", utc=True)
    if "date" not in df.columns and "createdAt" in df.columns:
        df["date"] = df["createdAt"].dt.floor("D")
    if "hour" not in df.columns and "createdAt" in df.columns:
        df["hour"] = df["createdAt"].dt.hour

    # Total participants: prefer aggregation sheet pid count if available
    total_users = None
    try:
        agg_key = next((k for k in csv_data.keys() if isinstance(k, str) and k.lower() == "aggregation"), None)
        if agg_key:
            agg = csv_data[agg_key]
            if isinstance(agg, pd.DataFrame) and not agg.empty:
                pid_col = next((c for c in agg.columns if isinstance(c, str) and c.lower() == "pid"), None)
                if pid_col:
                    total_users = int(pd.Series(agg[pid_col]).nunique())
    except Exception:
        total_users = None

    users_in_activities = int(df["pid"].nunique()) if "pid" in df.columns else 0
    if total_users is None or total_users <= 0:
        total_users = users_in_activities

    # Points/rewards fields
    if "points" not in df.columns and "rewardedParticipations" in df.columns:
        df["points"] = df["rewardedParticipations"].apply(extract_points)

    if "num_rewards" not in df.columns:
        if "rewardedParticipations" in df.columns:
            df["num_rewards"] = df["rewardedParticipations"].apply(lambda x: len(extract_detailed_rewards(x)))
        else:
            df["num_rewards"] = 0

    per_user = df.groupby("pid", dropna=True).agg(
        total_points=("points", "sum"),
        total_rewards=("num_rewards", "sum"),
        active_days=("date", lambda s: pd.Series(s).dropna().nunique()),
    ).reset_index()

    active_users = per_user[(per_user["total_points"] > 0) | (per_user["total_rewards"] > 0)]
    active_n = int(active_users["pid"].nunique())
    passive_n = max(0, int(total_users) - active_n)

    # Active players per day
    apd_mean, apd_sd = (None, None)
    if "pid" in df.columns and "date" in df.columns:
        active_players_per_day = df.dropna(subset=["pid", "date"]).groupby("date")["pid"].nunique()
        if not active_players_per_day.empty:
            apd_mean, apd_sd = _mean_sd(active_players_per_day)

    # Average active days per participant
    aad_mean, aad_sd = _mean_sd(per_user["active_days"])

    # Retention proxy
    retention_mean, retention_sd = (None, None)
    try:
        user_dropout = df.groupby("pid").agg(first=("createdAt", "min"), last=("createdAt", "max")).dropna()
        user_dropout["retention_days"] = (user_dropout["last"] - user_dropout["first"]).dt.days
        retention_mean, retention_sd = _mean_sd(user_dropout["retention_days"])
    except Exception:
        pass

    # Retention rate per week (wave)
    retention_by_wave_lines: List[str] = []
    if "wave" in df.columns and "pid" in df.columns:
        try:
            wave_active = df.dropna(subset=["pid", "wave"]).groupby("wave")["pid"].nunique().sort_index()
            for w, n in wave_active.items():
                retention_by_wave_lines.append(
                    f"  - Week {int(w)}: {int(n)} active users ({_fmt_pct(int(n), int(total_users))})"
                )
        except Exception:
            pass

    # Usage by day of week
    dow_lines: List[str] = []
    if "date" in df.columns:
        try:
            dow = pd.to_datetime(df["date"], errors="coerce").dt.day_name()
            dow_counts = dow.value_counts()
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            dow_counts = dow_counts.reindex(day_order).fillna(0).astype(int)
            total_acts = int(dow_counts.sum())
            for day in day_order:
                n = int(dow_counts.loc[day])
                dow_lines.append(f"  - {day}: {n} ({_fmt_pct(n, total_acts)})")
        except Exception:
            pass

    # Usage by hour buckets (FIXED LABELS)
    hour_bucket_lines: List[str] = []
    peak_hour_line = "N/A"
    if "hour" in df.columns:
        try:
            hours = pd.to_numeric(df["hour"], errors="coerce").dropna().astype(int)
            if not hours.empty:
                peak_hour = int(hours.value_counts().idxmax())
                peak_hour_line = f"{peak_hour:02d}:00"

                buckets = hours.apply(_bucket_hour).value_counts()
                bucket_order = [
                    "morning (6AM-11AM)",
                    "afternoon (12PM-5PM)",
                    "evening (6PM-11PM)",
                    "night (12AM-5AM)",
                ]
                buckets = buckets.reindex(bucket_order).fillna(0).astype(int)
                total = int(buckets.sum())
                for b in bucket_order:
                    n = int(buckets.loc[b])
                    hour_bucket_lines.append(f"  - {b}: {n} ({_fmt_pct(n, total)})")
        except Exception:
            pass

    # Activities distribution
    type_lines: List[str] = []
    if "type" in df.columns:
        type_counts = df["type"].astype(str).value_counts()
        total = int(type_counts.sum())
        for t, n in type_counts.items():
            type_lines.append(f"  - {t}: {int(n)} ({_fmt_pct(int(n), total)})")

    # Activities per player mean/sd
    acts_per_player = df.groupby("pid").size() if "pid" in df.columns else pd.Series(dtype=float)
    acts_pp_mean, acts_pp_sd = _mean_sd(acts_per_player)

    # Points per player mean/sd
    points_per_player = (
        df.groupby("pid")["points"].sum() if ("pid" in df.columns and "points" in df.columns) else pd.Series(dtype=float)
    )
    ppp_mean, ppp_sd = _mean_sd(points_per_player)

    # Points by activity type
    points_by_type_lines: List[str] = []
    if "type" in df.columns and "points" in df.columns:
        pbt = df.groupby("type")["points"].sum().sort_values(ascending=False)
        total = float(pbt.sum()) if not pbt.empty else 0.0
        for t, pts in pbt.items():
            pct = (float(pts) / total * 100.0) if total > 0 else 0.0
            points_by_type_lines.append(f"  - {t}: {float(pts):.0f} points ({pct:.1f}%)")

    # Tasks by provider (if desc_tasks exists)
    provider_lines: List[str] = []
    try:
        if "desc_tasks" in csv_data and isinstance(csv_data["desc_tasks"], pd.DataFrame) and not csv_data["desc_tasks"].empty:
            tasks_desc_df = csv_data["desc_tasks"].copy()
            col_map = {c.lower(): c for c in tasks_desc_df.columns}
            name_col = col_map.get("name") or col_map.get("label") or col_map.get("rule") or col_map.get("task") or col_map.get("task_name")
            providers_col = col_map.get("dataproviders")

            if name_col and providers_col and "rewardedParticipations" in df.columns:
                tmp = df[["pid", "rewardedParticipations"]].copy()
                tmp["detailed_rewards"] = tmp["rewardedParticipations"].apply(extract_detailed_rewards)
                tmp = tmp.explode("detailed_rewards").dropna(subset=["detailed_rewards"])
                tmp = tmp[tmp["detailed_rewards"].apply(lambda x: isinstance(x, dict))]
                tmp["task_name"] = tmp["detailed_rewards"].apply(lambda r: str(r.get("rule")).strip() if r.get("rule") is not None else None)
                tmp = tmp.dropna(subset=["task_name"])

                def _norm(s: str) -> str:
                    return re.sub(r"\s+", " ", str(s)).strip().lower()

                def _parse_providers(val) -> List[str]:
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

                tasks_desc_df["task_name_norm"] = tasks_desc_df[name_col].apply(_norm)
                tasks_desc_df["providers_list"] = tasks_desc_df[providers_col].apply(_parse_providers)
                providers_map = tasks_desc_df.set_index("task_name_norm")["providers_list"].to_dict()

                tmp["task_name_norm"] = tmp["task_name"].apply(_norm)
                tmp["providers_list"] = tmp["task_name_norm"].map(providers_map)
                mapped = tmp.dropna(subset=["providers_list"]).explode("providers_list").rename(columns={"providers_list": "provider"})
                mapped = mapped.dropna(subset=["provider"])

                if not mapped.empty:
                    counts = mapped.groupby("provider").size().sort_values(ascending=False)
                    total = int(counts.sum())
                    for prov, n in counts.items():
                        provider_lines.append(f"  - {prov}: {int(n)} ({_fmt_pct(int(n), total)})")
    except Exception:
        pass

    # Compose text
    lines: List[str] = []
    lines.append("USAGE METRICS")
    lines.append("=============")
    lines.append("Participating users (completing activities):")
    lines.append(f"  - Active users: {active_n}/{int(total_users)} ({_fmt_pct(active_n, int(total_users))})")
    lines.append(f"  - Passive users: {passive_n}/{int(total_users)} ({_fmt_pct(passive_n, int(total_users))})")
    lines.append("")
    lines.append(f"Average active days / participant: {_fmt_mean_sd(aad_mean, aad_sd)} days")
    lines.append(f"Active players per day: {_fmt_mean_sd(apd_mean, apd_sd)} players/day")
    lines.append(f"Average time to first inactivity (proxy = last-first activity): {_fmt_mean_sd(retention_mean, retention_sd)} days")
    lines.append("")
    lines.append("Retention rate / week (% active users among all participants):")
    lines.extend(retention_by_wave_lines if retention_by_wave_lines else ["  - N/A"])
    lines.append("")
    lines.append("Usage time / day of week (% of activities):")
    lines.extend(dow_lines if dow_lines else ["  - N/A"])
    lines.append("")
    lines.append("Usage time / hour of day buckets (% of activities):")
    lines.extend(hour_bucket_lines if hour_bucket_lines else ["  - N/A"])
    lines.append("")
    lines.append(f"Peak activity hour: {peak_hour_line}")
    lines.append("")
    lines.append("INTERACTION METRICS")
    lines.append("===================")
    lines.append("Activities distribution (% per type):")
    lines.extend(type_lines if type_lines else ["  - N/A"])
    lines.append("")
    lines.append("Tasks by data provider (% of task completions):")
    lines.extend(provider_lines if provider_lines else ["  - N/A (requires desc_tasks + rewards->rule mapping)"])
    lines.append("")
    lines.append(f"Average completed activities / participant: {_fmt_mean_sd(acts_pp_mean, acts_pp_sd)} activities")
    lines.append(f"Average rewarded points / participant: {_fmt_mean_sd(ppp_mean, ppp_sd)} points")
    lines.append("")
    lines.append("Rewarded points by activity type (% of total points):")
    lines.extend(points_by_type_lines if points_by_type_lines else ["  - N/A"])
    lines.append("")

    text = "\n".join(lines)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    logger.info(f"Saved descriptive summary text -> {out_path}")
    return out_path


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------
def generate_analysis_report(
    csv_data: Dict[str, pd.DataFrame],
    json_data: Dict[str, Union[pd.DataFrame, Dict]],
    activities: Optional[pd.DataFrame],
    campaign_metrics: Optional[Dict],
    dropout_metrics: Optional[Dict],
    joining_metrics: Optional[Dict] = None,
) -> str:
    data_analysis_dir = ensure_dir(os.path.join(PROJECT_ROOT, "data_analysis"))
    report_path = os.path.join(data_analysis_dir, "analysis_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("GameBus Data Analysis Report\n")
        f.write("===========================\n\n")

        f.write("1. Data Loading\n")
        f.write("---------------\n")
        if csv_data:
            f.write(f"✓ Excel data loaded successfully: {len(csv_data)} sheets\n")
        else:
            f.write("✗ No Excel data loaded\n")

        if json_data:
            f.write(f"✓ JSON data loaded successfully: {len(json_data)} files\n")
        else:
            f.write("✗ No JSON data loaded\n")

        f.write("\n2. Campaign Summary\n")
        f.write("------------------\n")
        cm = campaign_metrics or {}
        name = cm.get("name", "GameBus Campaign")
        abbr = cm.get("abbreviation", "")
        start_date = cm.get("start_date", "Unknown")
        end_date = cm.get("end_date", "Unknown")
        length_days = cm.get("length_days", 0)
        unique_users = cm.get("unique_users", 0)

        f.write(f"{name} ({abbr})\n\n" if abbr else f"{name}\n\n")
        f.write(f"Start Date: {start_date}\n")
        f.write(f"End Date: {end_date}\n")
        f.write(f"Length: {length_days} days\n")

        # total players from aggregation sheet if possible
        agg_total_users = None
        try:
            agg_key = next((k for k in csv_data.keys() if isinstance(k, str) and k.lower() == "aggregation"), None)
            if agg_key:
                df_agg = csv_data[agg_key]
                if isinstance(df_agg, pd.DataFrame) and not df_agg.empty:
                    pid_col = next((c for c in df_agg.columns if isinstance(c, str) and c.lower() == "pid"), None)
                    if pid_col:
                        agg_total_users = int(pd.Series(df_agg[pid_col]).nunique())
        except Exception:
            pass

        total_users_for_report = agg_total_users if isinstance(agg_total_users, int) and agg_total_users > 0 else unique_users
        f.write(f"Total number of Players in the Campaign: {total_users_for_report}\n")

        active_users_count = cm.get("active_users_count", None)
        passive_users_count = cm.get("passive_users_count", None)
        if active_users_count is not None and passive_users_count is not None:
            f.write(f"Active Players (rewarded): {active_users_count}\n")
            f.write(f"Passive Players (enrolled, never rewarded): {passive_users_count}\n")
            passive_ids = cm.get("passive_user_ids", [])
            if isinstance(passive_ids, list) and passive_ids:
                preview = ", ".join(map(str, passive_ids[:10]))
                f.write(f"Passive Player IDs (first 10): {preview}\n")

        f.write("\n3. Dropout Metrics\n")
        f.write("-----------------\n")
        dm = dropout_metrics or {}
        if dm:
            f.write(f"Average (days): {dm.get('avg_dropout_days', 'N/A')}\n")
            f.write(f"Median (days): {dm.get('median_dropout_days', 'N/A')}\n")
            f.write(f"Min (days): {dm.get('min_dropout_days', 'N/A')}\n")
            f.write(f"Max (days): {dm.get('max_dropout_days', 'N/A')}\n")
        else:
            f.write("N/A\n")

        f.write("\n4. Joining Metrics\n")
        f.write("-----------------\n")
        jm = joining_metrics or {}
        if jm:
            f.write(f"Average (days): {jm.get('avg_joining_days', 'N/A')}\n")
            f.write(f"Median (days): {jm.get('median_joining_days', 'N/A')}\n")
            f.write(f"Min (days): {jm.get('min_joining_days', 'N/A')}\n")
            f.write(f"Max (days): {jm.get('max_joining_days', 'N/A')}\n")
        else:
            f.write("N/A\n")

    logger.info(f"Saved analysis report -> {report_path}")
    return report_path


def create_complete_report(
    csv_data: Dict[str, pd.DataFrame],
    json_data: Dict[str, Union[pd.DataFrame, Dict]],
    activities_result: Optional[Union[pd.DataFrame, Tuple]],
) -> str:
    acts = None
    campaign_metrics = None
    dropout_metrics = None
    joining_metrics = None

    if isinstance(activities_result, tuple) and len(activities_result) >= 5:
        acts, _unique_users_count, dropout_metrics, joining_metrics, campaign_metrics = activities_result
    else:
        acts = activities_result if isinstance(activities_result, pd.DataFrame) else None

    return generate_analysis_report(csv_data, json_data, acts, campaign_metrics, dropout_metrics, joining_metrics)

