from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import (
    patch,
)

import pandas as pd

from src.analysis.common import (
    OUTPUT_VISUALIZATIONS_DIR,
)
from src.analysis.dataset_analysis import (
    DatasetAnalysisError,
    run_dataset_analysis,
)


class TestDatasetAnalysis(
    unittest.TestCase
):

    def _inspection(
        self,
        root: Path,
        *,
        cohort_manifest: (
            dict | None
        ),
    ):
        return SimpleNamespace(
            dataset_dir=(
                root
            ),
            campaign_data_path=(
                root
                / "campaign-export.zip"
            ),
            campaign_description_path=(
                root
                / "campaign.xlsx"
            ),
            cohort_manifest=(
                cohort_manifest
            ),
            raw_data_dir=(
                root
                / "data_raw"
            ),
            analysis_dir=(
                root
                / "data_analysis"
            ),
        )

    def test_analysis_uses_manifest_cohort_and_dataset_paths(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(
                tmp
            ).resolve()

            raw_dir = (
                root
                / "data_raw"
            )

            raw_dir.mkdir()

            manifest = {
                "participants": [
                    {
                        "pid": "497",
                        "selected_for_analysis": (
                            True
                        ),
                    },
                    {
                        "pid": "498",
                        "selected_for_analysis": (
                            False
                        ),
                    },
                ]
            }

            inspection = self._inspection(
                root,
                cohort_manifest=(
                    manifest
                ),
            )

            csv_data = {
                "activities": pd.DataFrame(
                    {
                        "pid": [
                            "497",
                            "498",
                        ],
                    }
                ),
            }

            json_data = {
                "player_497_test": {
                    "value": 1,
                },
            }

            filtered_csv = {
                "activities": pd.DataFrame(
                    {
                        "pid": [
                            "497",
                        ],
                    }
                ),
            }

            filtered_json = {
                "player_497_test": {
                    "value": 1,
                },
            }

            activity_result = (
                pd.DataFrame(
                    {
                        "pid": [
                            "497",
                        ],
                    }
                ),
                1,
                {},
                {},
                {},
            )

            def check_output_context(
                _csv_data,
            ):
                self.assertEqual(
                    os.fspath(
                        OUTPUT_VISUALIZATIONS_DIR
                    ),
                    os.fspath(
                        (
                            root
                            / "data_analysis"
                        ).resolve()
                    ),
                )

                return activity_result

            with (
                patch(
                    "src.analysis.dataset_analysis."
                    "inspect_dataset",
                    return_value=(
                        inspection
                    ),
                ),
                patch(
                    "src.analysis.dataset_analysis."
                    "load_excel_files",
                    return_value=(
                        csv_data
                    ),
                ) as load_campaign,
                patch(
                    "src.analysis.dataset_analysis."
                    "load_json_files",
                    return_value=(
                        json_data
                    ),
                ) as load_raw,
                patch(
                    "src.analysis.dataset_analysis."
                    "filter_tabular_data_to_user_ids",
                    return_value=(
                        filtered_csv
                    ),
                ) as filter_campaign,
                patch(
                    "src.analysis.dataset_analysis."
                    "filter_json_data_to_user_ids",
                    return_value=(
                        filtered_json
                    ),
                ) as filter_raw,
                patch(
                    "src.analysis.dataset_analysis."
                    "analyze_activities",
                    side_effect=(
                        check_output_context
                    ),
                ),
                patch(
                    "src.analysis.dataset_analysis."
                    "analyze_geofence_data"
                ),
                patch(
                    "src.analysis.dataset_analysis."
                    "analyze_day_aggregate_steps"
                ),
                patch(
                    "src.analysis.dataset_analysis."
                    "analyze_visualizations_challenges_tasks"
                ),
                patch(
                    "src.analysis.dataset_analysis."
                    "create_complete_report",
                    return_value=str(
                        root
                        / "data_analysis"
                        / "analysis_report.txt"
                    ),
                ),
            ):
                result = (
                    run_dataset_analysis(
                        root
                    )
                )

            load_campaign.assert_called_once_with(
                campaign_data_path=(
                    inspection
                    .campaign_data_path
                ),
                campaign_description_path=(
                    inspection
                    .campaign_description_path
                ),
            )

            load_raw.assert_called_once_with(
                raw_data_dir=(
                    inspection
                    .raw_data_dir
                )
            )

            filter_campaign.assert_called_once_with(
                csv_data,
                {
                    "497",
                },
            )

            filter_raw.assert_called_once_with(
                json_data,
                {
                    "497",
                },
            )

            self.assertEqual(
                result.selected_participants,
                1,
            )

            self.assertEqual(
                result.unique_active_users,
                1,
            )

            self.assertEqual(
                result.campaign_tables_loaded,
                1,
            )

            self.assertEqual(
                result.raw_json_files_loaded,
                1,
            )

    def test_existing_analysis_output_is_replaced(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(
                tmp
            ).resolve()

            (
                root
                / "data_raw"
            ).mkdir()

            output_dir = (
                root
                / "data_analysis"
            )

            output_dir.mkdir()

            stale_file = (
                output_dir
                / "old-result.txt"
            )

            stale_file.write_text(
                "old",
                encoding="utf-8",
            )

            manifest = {
                "participants": [
                    {
                        "pid": "497",
                        "selected_for_analysis": (
                            True
                        ),
                    }
                ]
            }

            inspection = self._inspection(
                root,
                cohort_manifest=(
                    manifest
                ),
            )

            with (
                patch(
                    "src.analysis.dataset_analysis."
                    "inspect_dataset",
                    return_value=(
                        inspection
                    ),
                ),
                patch(
                    "src.analysis.dataset_analysis."
                    "load_excel_files",
                    return_value={
                        "activities": (
                            pd.DataFrame(
                                {
                                    "pid": [
                                        "497",
                                    ],
                                }
                            )
                        ),
                    },
                ),
                patch(
                    "src.analysis.dataset_analysis."
                    "load_json_files",
                    return_value={},
                ),
                patch(
                    "src.analysis.dataset_analysis."
                    "filter_tabular_data_to_user_ids",
                    side_effect=lambda data, ids: data,
                ),
                patch(
                    "src.analysis.dataset_analysis."
                    "filter_json_data_to_user_ids",
                    side_effect=lambda data, ids: data,
                ),
                patch(
                    "src.analysis.dataset_analysis."
                    "analyze_activities",
                    return_value=None,
                ),
                patch(
                    "src.analysis.dataset_analysis."
                    "analyze_geofence_data"
                ),
                patch(
                    "src.analysis.dataset_analysis."
                    "analyze_day_aggregate_steps"
                ),
                patch(
                    "src.analysis.dataset_analysis."
                    "analyze_visualizations_challenges_tasks"
                ),
                patch(
                    "src.analysis.dataset_analysis."
                    "create_complete_report",
                    return_value=str(
                        output_dir
                        / "analysis_report.txt"
                    ),
                ),
            ):
                run_dataset_analysis(
                    root
                )

            self.assertFalse(
                stale_file.exists()
            )

            self.assertTrue(
                output_dir.is_dir()
            )

            self.assertTrue(
                (
                    output_dir
                    / "statistics"
                ).is_dir()
            )

    def test_analysis_requires_saved_cohort(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(
                tmp
            ).resolve()

            inspection = self._inspection(
                root,
                cohort_manifest=None,
            )

            with patch(
                "src.analysis.dataset_analysis."
                "inspect_dataset",
                return_value=(
                    inspection
                ),
            ):
                with self.assertRaisesRegex(
                    DatasetAnalysisError,
                    "cohort_manifest",
                ):
                    run_dataset_analysis(
                        root
                    )


if __name__ == "__main__":
    unittest.main()