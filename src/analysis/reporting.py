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
	cm = campaign_metrics or {}
	app_scope_metrics = cm.get("app_scope_metrics", {}) if isinstance(cm, dict) else {}

	# Ensure datetime + derived fields for global interaction metrics
	if "createdAt" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["createdAt"]):
		df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce", utc=True)
	if "date" not in df.columns and "createdAt" in df.columns:
		df["date"] = df["createdAt"].dt.floor("D")
	if "hour" not in df.columns and "createdAt" in df.columns:
		df["hour"] = df["createdAt"].dt.hour

	# ------------------------------------------------------------------
	# Enrolled users count (prefer campaign_metrics, fallback to aggregation)
	# ------------------------------------------------------------------
	total_users = cm.get("enrolled_users_count", None)

	if not isinstance(total_users, int) or total_users <= 0:
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

	if not isinstance(total_users, int) or total_users <= 0:
		total_users = int(df["pid"].nunique()) if "pid" in df.columns else 0

	# ------------------------------------------------------------------
	# Ensure points / rewards for global interaction metrics
	# ------------------------------------------------------------------
	if "points" not in df.columns and "rewardedParticipations" in df.columns:
		df["points"] = df["rewardedParticipations"].apply(extract_points)

	if "num_rewards" not in df.columns:
		if "rewardedParticipations" in df.columns:
			df["num_rewards"] = df["rewardedParticipations"].apply(lambda x: len(extract_detailed_rewards(x)))
		else:
			df["num_rewards"] = 0

	# ------------------------------------------------------------------
	# Helper formatters
	# ------------------------------------------------------------------
	def _as_series(obj) -> pd.Series:
		if isinstance(obj, pd.Series):
			return obj.copy()
		if isinstance(obj, dict):
			try:
				return pd.Series(obj)
			except Exception:
				return pd.Series(dtype=float)
		return pd.Series(dtype=float)

	def _series_pct_lines(series_obj, ordered_labels: Optional[List[str]] = None) -> List[str]:
		s = _as_series(series_obj)
		if s.empty:
			return ["  - N/A"]

		try:
			s = pd.to_numeric(s, errors="coerce").fillna(0)
		except Exception:
			pass

		if ordered_labels:
			s = s.reindex(ordered_labels).fillna(0)

		total = float(s.sum()) if not s.empty else 0.0
		lines_local: List[str] = []

		for idx, val in s.items():
			n = float(val) if pd.notna(val) else 0.0
			if total > 0:
				pct = n / total * 100.0
			else:
				pct = 0.0

			if abs(n - round(n)) < 1e-9:
				n_text = str(int(round(n)))
			else:
				n_text = f"{n:.2f}"

			lines_local.append(f"  - {idx}: {n_text} ({pct:.1f}%)")

		return lines_local if lines_local else ["  - N/A"]

	def _fmt_peak_hour(val) -> str:
		try:
			if val is None or pd.isna(val):
				return "N/A"
			return f"{int(val):02d}:00"
		except Exception:
			return "N/A"


	# ------------------------------------------------------------------
	# Real dropout and weekly retention
	# ------------------------------------------------------------------
	real_dropout_metrics = cm.get("real_dropout_metrics", {}) if isinstance(cm, dict) else {}
	weekly_retention = cm.get("weekly_retention", pd.DataFrame()) if isinstance(cm, dict) else pd.DataFrame()

	real_dropout_lines: List[str] = []
	if isinstance(real_dropout_metrics, dict) and real_dropout_metrics:
		threshold = real_dropout_metrics.get("inactivity_days_threshold", "N/A")
		campaign_end_source = real_dropout_metrics.get("campaign_end_source", "unknown")

		real_dropout_lines.append(
			f"  - Inactivity threshold: {threshold} days without qualifying activity"
		)
		real_dropout_lines.append(
			f"  - Joined users: {int(real_dropout_metrics.get('joined_users_count', 0))}/{int(total_users)} "
			f"({_fmt_pct(int(real_dropout_metrics.get('joined_users_count', 0)), int(total_users))})"
		)
		real_dropout_lines.append(
			f"  - Never-active users: {int(real_dropout_metrics.get('never_active_users_count', 0))}/{int(total_users)} "
			f"({_fmt_pct(int(real_dropout_metrics.get('never_active_users_count', 0)), int(total_users))})"
		)
		real_dropout_lines.append(
			f"  - Dropped-out users: {int(real_dropout_metrics.get('dropout_users_count', 0))}/{int(total_users)} "
			f"({float(real_dropout_metrics.get('dropout_rate_among_enrolled', 0.0)):.1f}% of enrolled; "
			f"{float(real_dropout_metrics.get('dropout_rate_among_joined', 0.0)):.1f}% of joined)"
		)
		real_dropout_lines.append(
			f"  - Retained/censored users: {int(real_dropout_metrics.get('retained_or_censored_users_count', 0))}/{int(total_users)} "
			f"({float(real_dropout_metrics.get('retention_rate_among_enrolled', 0.0)):.1f}% of enrolled; "
			f"{float(real_dropout_metrics.get('retention_rate_among_joined', 0.0)):.1f}% of joined)"
		)

		if campaign_end_source == "last_activity_fallback":
			real_dropout_lines.append(
				"  - Warning: campaign end was inferred from the last activity because desc_waves was unavailable; "
				"dropout detection may be underestimated."
			)

	weekly_retention_lines: List[str] = []
	if isinstance(weekly_retention, pd.DataFrame) and not weekly_retention.empty:
		for _, row in weekly_retention.sort_values("week").iterrows():
			weekly_retention_lines.append(
				f"  - Week {int(row['week'])} "
				f"({row['week_start']} to {row['week_end']}): "
				f"active {int(row['active_users_count'])}/{int(total_users)} "
				f"({float(row['active_pct_of_enrolled']):.1f}% enrolled); "
				f"retained {int(row['retained_users_count'])}/{int(total_users)} "
				f"({float(row['retention_pct_of_enrolled']):.1f}% enrolled; "
				f"{float(row['retention_pct_of_joined_by_week']):.1f}% joined-by-week)"
			)

	# ------------------------------------------------------------------
	# Global interaction metrics (unchanged logic)
	# ------------------------------------------------------------------
	type_lines: List[str] = []
	if "type" in df.columns:
		type_counts = df["type"].astype(str).value_counts()
		total = int(type_counts.sum()) if not type_counts.empty else 0
		for t, n in type_counts.items():
			type_lines.append(f"  - {t}: {int(n)} ({_fmt_pct(int(n), total)})")

	acts_per_player = df.groupby("pid").size() if "pid" in df.columns else pd.Series(dtype=float)
	acts_pp_mean, acts_pp_sd = _mean_sd(acts_per_player)

	acts_per_day = df.groupby("date").size() if "date" in df.columns else pd.Series(dtype=float)
	acts_pd_mean, acts_pd_sd = _mean_sd(acts_per_day)

	points_per_player = (
		df.groupby("pid")["points"].sum()
		if ("pid" in df.columns and "points" in df.columns)
		else pd.Series(dtype=float)
	)
	ppp_mean, ppp_sd = _mean_sd(points_per_player)

	points_by_type_lines: List[str] = []
	if "type" in df.columns and "points" in df.columns:
		pbt = df.groupby("type")["points"].sum().sort_values(ascending=False)
		total_points = float(pbt.sum()) if not pbt.empty else 0.0
		for t, pts in pbt.items():
			pct = (float(pts) / total_points * 100.0) if total_points > 0 else 0.0
			points_by_type_lines.append(f"  - {t}: {float(pts):.0f} points ({pct:.1f}%)")

	# Tasks by provider (unchanged logic)
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
				tmp["task_name"] = tmp["detailed_rewards"].apply(
					lambda r: str(r.get("rule")).strip() if r.get("rule") is not None else None
				)
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
					total_provider = int(counts.sum()) if not counts.empty else 0
					for prov, n in counts.items():
						provider_lines.append(f"  - {prov}: {int(n)} ({_fmt_pct(int(n), total_provider)})")
	except Exception:
		pass

	# ------------------------------------------------------------------
	# Compose text
	# ------------------------------------------------------------------
	lines: List[str] = []
	lines.append("USAGE METRICS")
	lines.append("=============")
	lines.append(f"Total enrolled participants: {int(total_users)}")
	lines.append("")
	lines.append("Active/passive user definitions:")
	lines.append(
		"  - GameBus active = has at least one of: DRINKING_DIARY, GENERAL_ACTIVITY, "
		"NUTRITION_DIARY, PHYSICAL_ACTIVITY, WALK, BIKE."
	)
	lines.append("  - GameBus passive = enrolled, but has none of the GameBus activity types above.")
	lines.append("  - Nutrida active = has at least one of: PLAN_MEAL, NUTRITION_DIARY_VID.")
	lines.append("  - Nutrida passive = enrolled, but has none of the Nutrida activity types above.")
	lines.append(
		"  - Combined active = has at least one GameBus-or-Nutrida qualifying activity."
	)
	lines.append(
		"  - Combined passive = enrolled, but has none of those qualifying activities in either app."
	)
	lines.append("")
	lines.append("Timing-analysis note:")
	lines.append(
		"  - Usage time / day of week, usage time / hour buckets, and peak activity hour "
		"exclude DAY_AGGREGATE where applicable."
	)
	lines.append(
		"  - Average active days / participant, active players per day, and observed activity span "
		"do not exclude DAY_AGGREGATE for GameBus / Combined."
	)
	lines.append("")

	# Legacy comparison
	legacy_active = cm.get("reward_based_active_users_count", None)
	legacy_passive = cm.get("reward_based_passive_users_count", None)
	if legacy_active is not None and legacy_passive is not None:
		lines.append("Legacy comparison (current reward-based method):")
		lines.append(f"  - Reward-based active users: {int(legacy_active)}/{int(total_users)} ({_fmt_pct(int(legacy_active), int(total_users))})")
		lines.append(f"  - Reward-based passive users: {int(legacy_passive)}/{int(total_users)} ({_fmt_pct(int(legacy_passive), int(total_users))})")
		lines.append("")

	# Per-scope sections
	scope_order = ["gamebus", "nutrida", "combined"]
	day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
	hour_bucket_order = ["00:00-05:59", "06:00-11:59", "12:00-17:59", "18:00-23:59"]

	for scope_key in scope_order:
		scope = app_scope_metrics.get(scope_key)
		if not isinstance(scope, dict):
			continue

		label = scope.get("label", scope_key.title())
		active_n = int(scope.get("active_users_count", 0))
		passive_n = int(scope.get("passive_users_count", 0))

		lines.append(f"{label}")
		lines.append("-" * len(label))
		lines.append(f"  - Active users: {active_n}/{int(total_users)} ({_fmt_pct(active_n, int(total_users))})")
		lines.append(f"  - Passive users: {passive_n}/{int(total_users)} ({_fmt_pct(passive_n, int(total_users))})")
		lines.append("")
		lines.append(
			f"  - Average active days / participant: "
			f"{scope.get('avg_active_days_per_participant', 0.0):.2f} days "
			f"(median {scope.get('median_active_days_per_participant', 0.0):.2f}, "
			f"std {scope.get('std_active_days_per_participant', 0.0):.2f})"
		)

		lines.append(
			f"  - Active players per day: "
			f"{scope.get('avg_active_players_per_day', 0.0):.2f} players/day "
			f"(median {scope.get('median_active_players_per_day', 0.0):.2f}, "
			f"std {scope.get('std_active_players_per_day', 0.0):.2f})"
		)

		lines.append(
			f"  - Observed activity span "
			f"(first to last recorded activity): "
			f"{scope.get('avg_time_to_first_inactivity_days', 0.0):.2f} days "
			f"(median {scope.get('median_time_to_first_inactivity_days', 0.0):.2f}, "
			f"std {scope.get('std_time_to_first_inactivity_days', 0.0):.2f})"
		)

		lines.append("")

		lines.append("")
		lines.append("  Usage time / day of week (% of scoped activities):")
		lines.extend(_series_pct_lines(scope.get("usage_by_day_of_week_series"), ordered_labels=day_order))
		lines.append("")
		lines.append("  Usage time / hour of day buckets (% of scoped activities):")
		lines.extend(_series_pct_lines(scope.get("usage_by_hour_buckets_series"), ordered_labels=hour_bucket_order))
		lines.append("")
		lines.append(f"  Peak activity hour: {_fmt_peak_hour(scope.get('peak_activity_hour'))}")
		lines.append("")


	lines.append("Real dropout status:")
	lines.extend(real_dropout_lines if real_dropout_lines else ["  - N/A"])
	lines.append("")

	lines.append("Weekly activity and retention:")
	lines.extend(weekly_retention_lines if weekly_retention_lines else ["  - N/A"])
	lines.append(
		"  Note: active users = users with qualifying activity in that week; "
		"retained users = users who have started and have not crossed the inactivity threshold by week end."
	)
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
	lines.append(
		f"Average completed activities / active day: "
		f"{_fmt_mean_sd(acts_pd_mean, acts_pd_sd)} activities"
	)
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

	cm = campaign_metrics or {}
	app_scope_metrics = cm.get("app_scope_metrics", {}) if isinstance(cm, dict) else {}

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
		f.write(f"(Analysis window source: first wave start to last wave end from campaign_desc.xlsx / desc_waves))")

		# total players from campaign_metrics first; fallback to aggregation; fallback to unique_users
		total_users_for_report = cm.get("enrolled_users_count", None)

		if not isinstance(total_users_for_report, int) or total_users_for_report <= 0:
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

		# Top-level active/passive now means Combined descriptor-based
		active_users_count = cm.get("active_users_count", None)
		passive_users_count = cm.get("passive_users_count", None)
		if active_users_count is not None and passive_users_count is not None:
			f.write(f"Active Players (combined, descriptor-based): {active_users_count}\n")
			f.write(f"Passive Players (combined, descriptor-based): {passive_users_count}\n")
			passive_ids = cm.get("passive_user_ids", [])
			if isinstance(passive_ids, list) and passive_ids:
				preview = ", ".join(map(str, passive_ids[:10]))
				f.write(f"Combined passive Player IDs (first 10): {preview}\n")

		legacy_active = cm.get("reward_based_active_users_count", None)
		legacy_passive = cm.get("reward_based_passive_users_count", None)
		if legacy_active is not None and legacy_passive is not None:
			f.write(f"Active Players (legacy reward-based): {legacy_active}\n")
			f.write(f"Passive Players (legacy reward-based): {legacy_passive}\n")

		f.write("\nActive/passive definitions:\n")
		f.write("  - GameBus active = has >=1 of: DRINKING_DIARY, GENERAL_ACTIVITY, NUTRITION_DIARY, PHYSICAL_ACTIVITY, WALK, BIKE\n")
		f.write("  - GameBus passive = enrolled, but has none of those GameBus activities\n")
		f.write("  - Nutrida active = has >=1 of: PLAN_MEAL, NUTRITION_DIARY_VID\n")
		f.write("  - Nutrida passive = enrolled, but has none of those Nutrida activities\n")
		f.write("  - Combined active = has >=1 qualifying GameBus-or-Nutrida activity\n")
		f.write("  - Combined passive = enrolled, but has none of those qualifying activities in either app\n")

		if app_scope_metrics:
			f.write("\nPer-scope usage summary:\n")
			for scope_key in ["gamebus", "nutrida", "combined"]:
				scope = app_scope_metrics.get(scope_key)
				if not isinstance(scope, dict):
					continue

				label = scope.get("label", scope_key.title())
				f.write(f"  {label}:\n")
				f.write(f"    - Active users: {int(scope.get('active_users_count', 0))}\n")
				f.write(f"    - Passive users: {int(scope.get('passive_users_count', 0))}\n")
				f.write(f"    - Avg active days / participant: {float(scope.get('avg_active_days_per_participant', 0.0)):.2f}\n")
				f.write(f"    - Avg active players / day: {float(scope.get('avg_active_players_per_day', 0.0)):.2f}\n")
				f.write(
					f"    - Avg observed activity span "
					f"(first-last activity): {float(scope.get('avg_time_to_first_inactivity_days', 0.0)):.2f}\n"
				)
				peak_hour = scope.get("peak_activity_hour", None)
				if peak_hour is None or (isinstance(peak_hour, float) and pd.isna(peak_hour)):
					f.write("    - Peak activity hour: N/A\n")
				else:
					f.write(f"    - Peak activity hour: {int(peak_hour):02d}:00\n")


		f.write("\n3. Real Dropout and Weekly Retention\n")
		f.write("-----------------------------------\n")

		real_dropout_metrics = cm.get("real_dropout_metrics", {}) if isinstance(cm, dict) else {}
		weekly_retention = cm.get("weekly_retention", pd.DataFrame()) if isinstance(cm, dict) else pd.DataFrame()

		if isinstance(real_dropout_metrics, dict) and real_dropout_metrics:
			f.write(
				f"Inactivity threshold: "
				f"{real_dropout_metrics.get('inactivity_days_threshold', 'N/A')} days\n"
			)
			f.write(
				f"Joined users: "
				f"{real_dropout_metrics.get('joined_users_count', 'N/A')}\n"
			)
			f.write(
				f"Never-active users: "
				f"{real_dropout_metrics.get('never_active_users_count', 'N/A')}\n"
			)
			f.write(
				f"Dropped-out users: "
				f"{real_dropout_metrics.get('dropout_users_count', 'N/A')}\n"
			)
			f.write(
				f"Dropout rate among enrolled: "
				f"{real_dropout_metrics.get('dropout_rate_among_enrolled', 'N/A')}%\n"
			)
			f.write(
				f"Dropout rate among joined: "
				f"{real_dropout_metrics.get('dropout_rate_among_joined', 'N/A')}%\n"
			)
			f.write(
				f"Retained/censored users: "
				f"{real_dropout_metrics.get('retained_or_censored_users_count', 'N/A')}\n"
			)
			f.write(
				f"Retention rate among enrolled: "
				f"{real_dropout_metrics.get('retention_rate_among_enrolled', 'N/A')}%\n"
			)
			f.write(
				f"Retention rate among joined: "
				f"{real_dropout_metrics.get('retention_rate_among_joined', 'N/A')}%\n"
			)

			if real_dropout_metrics.get("campaign_end_source") == "last_activity_fallback":
				f.write(
					"Warning: campaign end was inferred from last activity because desc_waves "
					"was unavailable; dropout detection may be underestimated.\n"
				)
		else:
			f.write("N/A\n")

		f.write("\nWeekly retention table:\n")
		if isinstance(weekly_retention, pd.DataFrame) and not weekly_retention.empty:
			for _, row in weekly_retention.sort_values("week").iterrows():
				f.write(
					f"  - Week {int(row['week'])} "
					f"({row['week_start']} to {row['week_end']}): "
					f"active={int(row['active_users_count'])}, "
					f"active_pct_enrolled={float(row['active_pct_of_enrolled']):.1f}%, "
					f"retained={int(row['retained_users_count'])}, "
					f"retention_pct_enrolled={float(row['retention_pct_of_enrolled']):.1f}%, "
					f"retention_pct_joined_by_week={float(row['retention_pct_of_joined_by_week']):.1f}%\n"
				)
		else:
			f.write("  - N/A\n")


		f.write("\n4. Activity-Span Metrics\n")
		f.write("-----------------------\n")
		f.write("Definition: days between each user's first and last recorded activity.\n")
		f.write("This is not a dropout rate by itself.\n")
		dm = dropout_metrics or {}
		if dm:
			f.write(f"Average activity span (days): {dm.get('avg_dropout_days', 'N/A')}\n")
			f.write(f"Median activity span (days): {dm.get('median_dropout_days', 'N/A')}\n")
			f.write(f"Min activity span (days): {dm.get('min_dropout_days', 'N/A')}\n")
			f.write(f"Max activity span (days): {dm.get('max_dropout_days', 'N/A')}\n")
		else:
			f.write("N/A\n")

		f.write("\n5. Joining Metrics\n")
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

	return generate_analysis_report(
		csv_data,
		json_data,
		acts,
		campaign_metrics,
		dropout_metrics,
		joining_metrics,
	)
