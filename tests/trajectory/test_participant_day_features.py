from __future__ import annotations

import unittest

import pandas as pd

from src.trajectory.participant_day_features import (
	build_participant_day_features,
)


class TestParticipantDayFeatures(
	unittest.TestCase
):

	def _state(
		self,
	) -> pd.DataFrame:

		dates = pd.date_range(
			"2026-01-01",
			periods=70,
			freq="D",
			tz="UTC",
		)

		explicit = [
			0
			for _ in dates
		]

		# Explicit engagement on days:
		# 1, 30 and 60.
		explicit[0] = 1
		explicit[29] = 2
		explicit[59] = 1

		rows = []

		last_engagement = None
		episode = None

		for index, date in enumerate(
			dates
		):

			if explicit[index] > 0:

				if (
					last_engagement
					is None
					or (
						date
						- last_engagement
					).days > 14
				):
					episode = (
						1
						if episode is None
						else episode + 1
					)

				last_engagement = date

			days_since = (
				(
					date
					- last_engagement
				).days
				if last_engagement
				is not None
				else pd.NA
			)

			current_episode = (
				episode
				if (
					last_engagement
					is not None
					and days_since <= 14
				)
				else pd.NA
			)

			rows.append(
				{
					"participant_id": 1,
					"date": date,
					"analysis_phase": (
						"maintenance_reengagement"
						if index >= 0
						else (
							"before_first_observed_"
							"explicit_engagement"
						)
					),
					(
						"maintenance_reengagement_"
						"eligible"
					): True,
					"engagement_state": (
						"active"
						if explicit[index] > 0
						else "quiet"
					),
					(
						"first_observed_explicit_"
						"engagement_date"
					): dates[0],
					(
						"last_explicit_"
						"engagement_date"
					): last_engagement,
					(
						"days_since_last_"
						"explicit_engagement"
					): days_since,
					"episode_number": (
						current_episode
					),
					"explicit_events_today": (
						explicit[index]
					),
					"explicit_events_7d": 0,
					"explicit_events_14d": 0,
					"active_days_7d": 0,
					"active_days_14d": 0,
					"points_today": (
						explicit[index]
					),
					"points_7d": 0,
					"points_14d": 0,
					(
						"behavioral_sensor_"
						"events_today"
					): 0,
					"activity_stream_state": (
						"available"
					),
					"navigation_stream_state": (
						"available_empty"
					),
					"notification_stream_state": (
						"available_empty"
					),
					"sensor_stream_state": (
						"available_empty"
					),
					"garmin_stream_state": (
						"unavailable"
					),
					"nutrida_stream_state": (
						"unavailable"
					),
				}
			)

		return pd.DataFrame(
			rows
		)

	def test_previous_window_does_not_use_future_data(
		self,
	) -> None:

		result = (
			build_participant_day_features(
				self._state()
			)
		)

		# Day 28:
		# recent window contains day-1 engagement;
		# there is no preceding history yet.
		day_28 = result.iloc[
			27
		]

		self.assertEqual(
			day_28[
				"explicit_events_28d"
			],
			1,
		)

		self.assertTrue(
			pd.isna(
				day_28[
					"explicit_event_ratio_28d"
				]
			)
		)

		# Day 29:
		# day-1 has moved into the immediately
		# preceding 28-day window.
		day_29 = result.iloc[
			28
		]

		self.assertEqual(
			day_29[
				"explicit_events_28d"
			],
			0,
		)

		self.assertEqual(
			day_29[
				"explicit_events_previous_28d"
			],
			1,
		)

	def test_future_engagement_does_not_change_earlier_row(
		self,
	) -> None:

		state = self._state()

		first = (
			build_participant_day_features(
				state
			)
		)

		earlier = first.loc[
			first["date"]
			== pd.Timestamp(
				"2026-02-10"
			).date()
		].iloc[
			0
		]

		# Add a large event count well in the future.
		state.loc[
			state["date"]
			== pd.Timestamp(
				"2026-03-01",
				tz="UTC",
			),
			"explicit_events_today",
		] = 1000

		second = (
			build_participant_day_features(
				state
			)
		)

		earlier_after = second.loc[
			second["date"]
			== pd.Timestamp(
				"2026-02-10"
			).date()
		].iloc[
			0
		]

		self.assertEqual(
			earlier[
				"explicit_events_28d"
			],
			earlier_after[
				"explicit_events_28d"
			],
		)

		self.assertEqual(
			earlier[
				"cumulative_explicit_events"
			],
			earlier_after[
				"cumulative_explicit_events"
			],
		)

	def test_reengagement_history_is_causal(
		self,
	) -> None:

		result = (
			build_participant_day_features(
				self._state()
			)
		)

		before_second_episode = (
			result.iloc[
				28
			]
		)

		second_episode_day = (
			result.iloc[
				29
			]
		)

		self.assertEqual(
			before_second_episode[
				"reengagements_to_date"
			],
			0,
		)

		self.assertEqual(
			second_episode_day[
				"reengagements_to_date"
			],
			1,
		)

	def test_domain_and_tool_history_does_not_double_count_multidomain_event(
			self,
	) -> None:
			state = self._state()

			domain_tool = pd.DataFrame(
				[
					{
						"event_id": "e1",
						"participant_id": 1,
						"date": (
							"2026-01-01"
						),
						"event_channel": (
							"explicit_engagement"
						),
						"domain": "nutrition",
						"tool": "gamebus_studio",
						"event_weight": 0.5,
					},
					{
						"event_id": "e1",
						"participant_id": 1,
						"date": (
							"2026-01-01"
						),
						"event_channel": (
							"explicit_engagement"
						),
						"domain": (
							"physical_activity"
						),
						"tool": "gamebus_studio",
						"event_weight": 0.5,
					},
					{
						"event_id": "e2",
						"participant_id": 1,
						"date": (
							"2026-01-02"
						),
						"event_channel": (
							"explicit_engagement"
						),
						"domain": "nutrition",
						"tool": "nutrida",
						"event_weight": 1.0,
					},
				]
			)

			result = (
				build_participant_day_features(
					state,
					domain_tool,
				)
			)

			day_2 = result.loc[
				result["date"]
				== pd.Timestamp(
					"2026-01-02"
				).date()
				].iloc[
				0
			]

			self.assertEqual(
				day_2[
					"nutrition_explicit_weight_7d"
				],
				1.5,
			)

			self.assertEqual(
				day_2[
					(
						"physical_activity_"
						"explicit_weight_7d"
					)
				],
				0.5,
			)

			self.assertEqual(
				day_2[
					"explicit_domain_diversity_7d"
				],
				2,
			)

			# e1 has two domain rows but represents
			# one GameBus event.
			self.assertEqual(
				day_2[
					(
						"gamebus_studio_"
						"explicit_events_7d"
					)
				],
				1,
			)

			self.assertEqual(
				day_2[
					"nutrida_explicit_events_7d"
				],
				1,
			)

			self.assertEqual(
				day_2[
					"explicit_tool_diversity_7d"
				],
				2,
			)

			self.assertEqual(
				day_2[
					"mapped_domain_share_28d"
				],
				1.0,
			)

			self.assertEqual(
				day_2[
					(
						"days_since_physical_activity_"
						"explicit_engagement"
					)
				],
				1,
			)

			self.assertEqual(
				day_2[
					(
						"days_since_nutrition_"
						"explicit_engagement"
					)
				],
				0,
			)

if __name__ == "__main__":
	unittest.main()