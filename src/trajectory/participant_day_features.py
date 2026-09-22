from __future__ import annotations

import os

import pandas as pd

from src.trajectory.common import (
	TrajectoryAuditConfig,
)
from src.trajectory.event_normalization import (
	build_normalized_events,
)
from src.trajectory.participant_state_daily import (
	build_participant_state_daily,
)
from src.trajectory.domain_tool_engagement import (
	build_domain_tool_engagement,
)


FEATURE_COLUMNS = [
	"participant_id",
	"date",

	"analysis_phase",
	"maintenance_reengagement_eligible",

	"engagement_state",

	"first_observed_explicit_engagement_date",
	"last_explicit_engagement_date",

	"days_since_first_explicit_engagement",
	"days_since_last_explicit_engagement",

	"explicit_events_today",

	"explicit_events_7d",
	"explicit_events_14d",
	"explicit_events_28d",

	"active_days_7d",
	"active_days_14d",
	"active_days_28d",

	"explicit_events_previous_28d",
	"active_days_previous_28d",

	"recent_history_days_28d",
	"reference_history_days_28d",

	"explicit_event_ratio_28d",
	"active_day_ratio_28d",

	"cumulative_explicit_events",
	"cumulative_active_days",

	"episodes_started_to_date",
	"reengagements_to_date",

	"long_inactivity_spells_14d_to_date",

	"points_today",
	"points_7d",
	"points_14d",
	"points_28d",

	"behavioral_sensor_events_today",
	"behavioral_sensor_events_7d",
	"behavioral_sensor_events_28d",

	"nutrition_explicit_weight_7d",
	"nutrition_explicit_weight_28d",
	"physical_activity_explicit_weight_7d",
	"physical_activity_explicit_weight_28d",
	"mental_wellbeing_explicit_weight_7d",
	"mental_wellbeing_explicit_weight_28d",
	"unmapped_domain_explicit_weight_7d",
	"unmapped_domain_explicit_weight_28d",

	"explicit_domain_diversity_7d",
	"explicit_domain_diversity_28d",
	"mapped_domain_share_28d",

	"days_since_nutrition_explicit_engagement",
	"days_since_physical_activity_explicit_engagement",
	"days_since_mental_wellbeing_explicit_engagement",

	"gamebus_studio_explicit_events_7d",
	"gamebus_studio_explicit_events_28d",
	"nutrida_explicit_events_7d",
	"nutrida_explicit_events_28d",
	"garmin_explicit_events_7d",
	"garmin_explicit_events_28d",
	"other_tool_explicit_events_7d",
	"other_tool_explicit_events_28d",

	"explicit_tool_diversity_7d",
	"explicit_tool_diversity_28d",

	"days_since_gamebus_studio_explicit_engagement",
	"days_since_nutrida_explicit_engagement",
	"days_since_garmin_explicit_engagement",

	"activity_stream_state",
	"navigation_stream_state",
	"notification_stream_state",
	"sensor_stream_state",
	"garmin_stream_state",
	"nutrida_stream_state",
]


def _numeric(
	frame: pd.DataFrame,
	column: str,
) -> pd.Series:
	return pd.to_numeric(
		frame[column],
		errors="coerce",
	)


def _rolling_sum(
	frame: pd.DataFrame,
	column: str,
	window: int,
) -> pd.Series:
	return (
		frame
		.groupby(
			"participant_id",
			sort=False,
		)[column]
		.transform(
			lambda values: (
				pd.to_numeric(
					values,
					errors="coerce",
				)
				.rolling(
					window=window,
					min_periods=1,
				)
				.sum()
			)
		)
	)


def _previous_window_sum(
	frame: pd.DataFrame,
	column: str,
	window: int,
) -> pd.Series:
	"""
	Sum the window immediately preceding the current
	trailing window.

	For a 28-day window at day t:

		recent:
			t-27 ... t

		previous:
			t-55 ... t-28

	No future information is used.
	"""

	return (
		frame
		.groupby(
			"participant_id",
			sort=False,
		)[column]
		.transform(
			lambda values: (
				pd.to_numeric(
					values,
					errors="coerce",
				)
				.shift(window)
				.rolling(
					window=window,
					min_periods=1,
				)
				.sum()
			)
		)
	)


def _history_days(
	frame: pd.DataFrame,
	window: int,
	*,
	offset: int = 0,
) -> pd.Series:
	"""
	Number of actually available calendar days in a
	trailing window.

	offset=0:
		current trailing window

	offset=window:
		immediately preceding window
	"""

	ones = pd.Series(
		1.0,
		index=frame.index,
	)

	temp = frame[
		[
			"participant_id",
		]
	].copy()

	temp["_one"] = ones

	return (
		temp
		.groupby(
			"participant_id",
			sort=False,
		)["_one"]
		.transform(
			lambda values: (
				values
				.shift(offset)
				.rolling(
					window=window,
					min_periods=1,
				)
				.sum()
			)
		)
		.fillna(0)
		.astype("Int64")
	)


def _safe_ratio(
	numerator: pd.Series,
	denominator: pd.Series,
) -> pd.Series:
	"""
	Return a ratio only when the reference value is
	greater than zero.

	A zero historical denominator is not interpreted
	as infinite improvement.
	"""

	numerator = pd.to_numeric(
		numerator,
		errors="coerce",
	)

	denominator = pd.to_numeric(
		denominator,
		errors="coerce",
	)

	result = pd.Series(
		pd.NA,
		index=numerator.index,
		dtype="Float64",
	)

	valid = (
		denominator.notna()
		& denominator.gt(0)
	)

	result.loc[
		valid
	] = (
		numerator.loc[
			valid
		]
		/ denominator.loc[
			valid
		]
	)

	return result


def _days_since_positive(
	frame: pd.DataFrame,
	column: str,
) -> pd.Series:
	values = pd.to_numeric(
		frame[column],
		errors="coerce",
	).fillna(0)

	positive_date = frame[
		"date"
	].where(
		values.gt(0)
	)

	last_positive_date = (
		positive_date
		.groupby(
			frame[
				"participant_id"
			]
		)
		.ffill()
	)

	return (
		frame["date"]
		- last_positive_date
	).dt.days.astype(
		"Int64"
	)


def _add_domain_tool_history(
	frame: pd.DataFrame,
	domain_tool: pd.DataFrame | None,
) -> pd.DataFrame:
	"""
	Add causal trailing domain/tool history.

	Domain counts use event_weight so a multi-domain
	activity still contributes total weight 1.

	Tool counts deduplicate event_id because the
	domain/tool table can contain several rows for one
	multi-domain event.
	"""

	result = frame.copy()

	domains = (
		"nutrition",
		"physical_activity",
		"mental_wellbeing",
		"unmapped",
	)

	tools = (
		"gamebus_studio",
		"nutrida",
		"garmin",
		"other",
	)

	for domain in domains:
		result[
			f"_domain_{domain}_today"
		] = 0.0

	for tool in tools:
		result[
			f"_tool_{tool}_today"
		] = 0

	if (
		domain_tool is not None
		and not domain_tool.empty
	):
		prepared = domain_tool.copy()

		prepared[
			"participant_id"
		] = pd.to_numeric(
			prepared[
				"participant_id"
			],
			errors="coerce",
		).astype(
			"Int64"
		)

		prepared["date"] = pd.to_datetime(
			prepared["date"],
			utc=True,
			errors="coerce",
		)

		prepared = prepared.dropna(
			subset=[
				"participant_id",
				"date",
			]
		)

		prepared = prepared.loc[
			prepared[
				"event_channel"
			].astype(
				"string"
			).eq(
				"explicit_engagement"
			)
		].copy()

		prepared[
			"event_weight"
		] = pd.to_numeric(
			prepared[
				"event_weight"
			],
			errors="coerce",
		).fillna(
			0.0
		)

		prepared[
			"_domain"
		] = (
			prepared["domain"]
			.astype("string")
			.fillna("unmapped")
			.str.lower()
		)

		prepared.loc[
			~prepared[
				"_domain"
			].isin(
				domains
			),
			"_domain",
		] = "unmapped"

		for domain in domains:
			daily = (
				prepared.loc[
					prepared[
						"_domain"
					].eq(
						domain
					)
				]
				.groupby(
					[
						"participant_id",
						"date",
					],
					as_index=False,
				)[
					"event_weight"
				]
				.sum()
				.rename(
					columns={
						"event_weight": (
							f"_domain_{domain}_today"
						)
					}
				)
			)

			result = result.drop(
				columns=[
					f"_domain_{domain}_today"
				]
			).merge(
				daily,
				on=[
					"participant_id",
					"date",
				],
				how="left",
			)

			result[
				f"_domain_{domain}_today"
			] = (
				pd.to_numeric(
					result[
						f"_domain_{domain}_today"
					],
					errors="coerce",
				)
				.fillna(0.0)
			)

		# One multi-domain event appears several
		# times in domain_tool_engagement. Count it
		# only once when constructing tool history.
		tool_events = (
			prepared
			.drop_duplicates(
				subset=[
					"participant_id",
					"date",
					"event_id",
					"tool",
				]
			)
			.copy()
		)

		tool_events[
			"_tool"
		] = (
			tool_events["tool"]
			.astype("string")
			.fillna("other")
			.str.lower()
		)

		tool_events.loc[
			~tool_events[
				"_tool"
			].isin(
				(
					"gamebus_studio",
					"nutrida",
					"garmin",
				)
			),
			"_tool",
		] = "other"

		for tool in tools:
			daily = (
				tool_events.loc[
					tool_events[
						"_tool"
					].eq(
						tool
					)
				]
				.groupby(
					[
						"participant_id",
						"date",
					]
				)
				.size()
				.rename(
					f"_tool_{tool}_today"
				)
				.reset_index()
			)

			result = result.drop(
				columns=[
					f"_tool_{tool}_today"
				]
			).merge(
				daily,
				on=[
					"participant_id",
					"date",
				],
				how="left",
			)

			result[
				f"_tool_{tool}_today"
			] = (
				pd.to_numeric(
					result[
						f"_tool_{tool}_today"
					],
					errors="coerce",
				)
				.fillna(0)
			)

	# Domain trailing histories.
	for domain in domains:
		daily_column = (
			f"_domain_{domain}_today"
		)

		for window in (
			7,
			28,
		):
			result[
				(
					f"{domain}_explicit_"
					f"weight_{window}d"
				)
			] = _rolling_sum(
				result,
				daily_column,
				window,
			)

	mapped_domains = (
		"nutrition",
		"physical_activity",
		"mental_wellbeing",
	)

	result[
		"explicit_domain_diversity_7d"
	] = sum(
		result[
			f"{domain}_explicit_weight_7d"
		].gt(0).astype(int)
		for domain in mapped_domains
	)

	result[
		"explicit_domain_diversity_28d"
	] = sum(
		result[
			f"{domain}_explicit_weight_28d"
		].gt(0).astype(int)
		for domain in mapped_domains
	)

	mapped_28d = sum(
		result[
			f"{domain}_explicit_weight_28d"
		]
		for domain in mapped_domains
	)

	total_28d = (
		mapped_28d
		+ result[
			"unmapped_domain_explicit_weight_28d"
		]
	)

	result[
		"mapped_domain_share_28d"
	] = _safe_ratio(
		mapped_28d,
		total_28d,
	)

	for domain in mapped_domains:
		result[
			(
				"days_since_"
				f"{domain}_explicit_engagement"
			)
		] = _days_since_positive(
			result,
			f"_domain_{domain}_today",
		)

	# Tool trailing histories.
	tool_prefixes = {
		"gamebus_studio": (
			"gamebus_studio"
		),
		"nutrida": "nutrida",
		"garmin": "garmin",
		"other": "other_tool",
	}

	for tool, prefix in (
		tool_prefixes.items()
	):
		daily_column = (
			f"_tool_{tool}_today"
		)

		for window in (
			7,
			28,
		):
			result[
				(
					f"{prefix}_explicit_"
					f"events_{window}d"
				)
			] = _rolling_sum(
				result,
				daily_column,
				window,
			)

	result[
		"explicit_tool_diversity_7d"
	] = sum(
		result[
			(
				f"{prefix}_explicit_"
				"events_7d"
			)
		].gt(0).astype(int)
		for prefix in (
			tool_prefixes.values()
		)
	)

	result[
		"explicit_tool_diversity_28d"
	] = sum(
		result[
			(
				f"{prefix}_explicit_"
				"events_28d"
			)
		].gt(0).astype(int)
		for prefix in (
			tool_prefixes.values()
		)
	)

	for tool in (
		"gamebus_studio",
		"nutrida",
		"garmin",
	):
		result[
			(
				"days_since_"
				f"{tool}_explicit_engagement"
			)
		] = _days_since_positive(
			result,
			f"_tool_{tool}_today",
		)

	temporary_columns = [
		column
		for column in result.columns
		if (
			column.startswith(
				"_domain_"
			)
			or column.startswith(
				"_tool_"
			)
		)
	]

	return result.drop(
		columns=temporary_columns
	)



def build_participant_day_features(
	state: pd.DataFrame,
	domain_tool: pd.DataFrame | None = None,
) -> pd.DataFrame:
	"""
	Build the canonical longitudinal feature table.

	One row represents the information available for
	one participant at the end of one calendar day.

	All derived features are causal:
	no information occurring after that row's date is
	consulted.
	"""

	if state.empty:
		return pd.DataFrame(
			columns=FEATURE_COLUMNS
		)

	result = state.copy()

	result[
		"participant_id"
	] = pd.to_numeric(
		result[
			"participant_id"
		],
		errors="coerce",
	).astype(
		"Int64"
	)

	result["date"] = pd.to_datetime(
		result["date"],
		utc=True,
		errors="coerce",
	)

	result = (
		result
		.dropna(
			subset=[
				"participant_id",
				"date",
			]
		)
		.sort_values(
			[
				"participant_id",
				"date",
			]
		)
		.reset_index(
			drop=True
		)
	)

	# -------------------------------------------------
	# First explicit engagement
	# -------------------------------------------------

	result[
		"first_observed_explicit_engagement_date"
	] = pd.to_datetime(
		result[
			"first_observed_explicit_engagement_date"
		],
		utc=True,
		errors="coerce",
	)

	result[
		"last_explicit_engagement_date"
	] = pd.to_datetime(
		result[
			"last_explicit_engagement_date"
		],
		utc=True,
		errors="coerce",
	)

	result[
		"days_since_first_explicit_engagement"
	] = (
		result["date"]
		- result[
			"first_observed_explicit_engagement_date"
		]
	).dt.days.astype(
		"Int64"
	)

	# -------------------------------------------------
	# Basic daily values
	# -------------------------------------------------

	explicit_today = _numeric(
		result,
		"explicit_events_today",
	)

	result[
		"_active_today"
	] = (
		explicit_today
		.fillna(0)
		.gt(0)
		.astype(int)
	)

	# -------------------------------------------------
	# 28-day trailing history
	# -------------------------------------------------

	result[
		"explicit_events_28d"
	] = _rolling_sum(
		result,
		"explicit_events_today",
		28,
	)

	result[
		"active_days_28d"
	] = _rolling_sum(
		result,
		"_active_today",
		28,
	)

	result[
		"explicit_events_previous_28d"
	] = _previous_window_sum(
		result,
		"explicit_events_today",
		28,
	)

	result[
		"active_days_previous_28d"
	] = _previous_window_sum(
		result,
		"_active_today",
		28,
	)

	result[
		"recent_history_days_28d"
	] = _history_days(
		result,
		28,
	)

	result[
		"reference_history_days_28d"
	] = _history_days(
		result,
		28,
		offset=28,
	)

	result[
		"explicit_event_ratio_28d"
	] = _safe_ratio(
		result[
			"explicit_events_28d"
		],
		result[
			"explicit_events_previous_28d"
		],
	)

	result[
		"active_day_ratio_28d"
	] = _safe_ratio(
		result[
			"active_days_28d"
		],
		result[
			"active_days_previous_28d"
		],
	)

	# -------------------------------------------------
	# Cumulative history
	# -------------------------------------------------

	result[
		"cumulative_explicit_events"
	] = (
		explicit_today
		.fillna(0)
		.groupby(
			result[
				"participant_id"
			]
		)
		.cumsum()
	)

	result[
		"cumulative_active_days"
	] = (
		result[
			"_active_today"
		]
		.groupby(
			result[
				"participant_id"
			]
		)
		.cumsum()
	)

	# -------------------------------------------------
	# Participation episodes / re-engagement history
	# -------------------------------------------------

	episode_number = pd.to_numeric(
		result[
			"episode_number"
		],
		errors="coerce",
	)

	result[
		"episodes_started_to_date"
	] = (
		episode_number
		.groupby(
			result[
				"participant_id"
			]
		)
		.cummax()
		.fillna(0)
		.astype(
			"Int64"
		)
	)

	result[
		"reengagements_to_date"
	] = (
		result[
			"episodes_started_to_date"
		]
		- 1
	).clip(
		lower=0
	).astype(
		"Int64"
	)

	# Each transition to exactly 14 days since the last
	# explicit engagement represents one newly reached
	# 14-day inactivity spell.
	reached_14d_today = (
		pd.to_numeric(
			result[
				"days_since_last_explicit_engagement"
			],
			errors="coerce",
		)
		.eq(14)
		.astype(int)
	)

	result[
		"long_inactivity_spells_14d_to_date"
	] = (
		reached_14d_today
		.groupby(
			result[
				"participant_id"
			]
		)
		.cumsum()
		.astype(
			"Int64"
		)
	)

	# -------------------------------------------------
	# Points history
	# -------------------------------------------------

	if "points_today" in result.columns:

		result[
			"points_28d"
		] = _rolling_sum(
			result,
			"points_today",
			28,
		)

	else:
		result[
			"points_28d"
		] = pd.NA

	# -------------------------------------------------
	# Passive / behavioral observation history
	# -------------------------------------------------

	if (
		"behavioral_sensor_events_today"
		in result.columns
	):

		result[
			"behavioral_sensor_events_7d"
		] = _rolling_sum(
			result,
			"behavioral_sensor_events_today",
			7,
		)

		result[
			"behavioral_sensor_events_28d"
		] = _rolling_sum(
			result,
			"behavioral_sensor_events_today",
			28,
		)

	else:

		result[
			"behavioral_sensor_events_7d"
		] = pd.NA

		result[
			"behavioral_sensor_events_28d"
		] = pd.NA

	result = _add_domain_tool_history(
		result,
		domain_tool,
	)

	result = result.drop(
		columns=[
			"_active_today",
		]
	)

	# -------------------------------------------------
	# Stable schema
	# -------------------------------------------------

	for column in FEATURE_COLUMNS:

		if column not in result.columns:
			result[column] = pd.NA

	# Keep machine-readable ISO dates in CSV output.
	for column in (
		"date",
		"first_observed_explicit_engagement_date",
		"last_explicit_engagement_date",
	):

		result[column] = (
			pd.to_datetime(
				result[column],
				utc=True,
				errors="coerce",
			)
			.dt.date
		)

	return result[
		FEATURE_COLUMNS
	]


def run_participant_day_features(
	config: TrajectoryAuditConfig | None = None,
) -> pd.DataFrame:
	"""
	Build and save participant_day_features.csv.
	"""

	config = (
		config
		or TrajectoryAuditConfig()
	)

	events = build_normalized_events(
		config
	)

	domain_tool = (
		build_domain_tool_engagement(
			config,
			events,
		)
	)

	state = build_participant_state_daily(
		config,
		events,
	)

	features = (
		build_participant_day_features(
			state,
			domain_tool,
		)
	)

	os.makedirs(
		config.output_dir,
		exist_ok=True,
	)

	output_path = os.path.join(
		config.output_dir,
		"participant_day_features.csv",
	)

	features.to_csv(
		output_path,
		index=False,
	)

	return features


if __name__ == "__main__":

	features = (
		run_participant_day_features()
	)

	print(
		"Participant-day features "
		"written successfully."
	)

	print(
		"Rows:",
		len(
			features
		),
	)