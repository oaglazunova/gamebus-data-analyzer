from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import zipfile

import pandas as pd

from src.analysis.loaders import (
    load_excel_files,
    load_json_files,
)


class TestDatasetLoaders(
    unittest.TestCase
):

    def test_loads_explicit_campaign_files(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(
                tmp
            )

            zip_path = (
                root
                / "campaign-456-export.zip"
            )

            description_path = (
                root
                / "campaign-456.xlsx"
            )

            with zipfile.ZipFile(
                zip_path,
                "w",
            ) as archive:
                archive.writestr(
                    "1-aggregated-data.csv",
                    "pid,value\n10,1\n",
                )

                archive.writestr(
                    "2-activities.csv",
                    (
                        "pid,type,"
                        "rewardedParticipations,"
                        "createdAt\n"
                        "10,WALK,[],"
                        "2026-01-01T10:00:00Z\n"
                    ),
                )

                archive.writestr(
                    "3-navigation-events.csv",
                    "pid\n10\n",
                )

                archive.writestr(
                    "4-notification-events.csv",
                    "pid\n10\n",
                )

                archive.writestr(
                    "5-sensor-events.csv",
                    "pid\n10\n",
                )

            with pd.ExcelWriter(
                description_path,
                engine="openpyxl",
            ) as writer:
                pd.DataFrame(
                    {
                        "id": [
                            1,
                        ],
                        "name": [
                            "Task A",
                        ],
                    }
                ).to_excel(
                    writer,
                    sheet_name="tasks",
                    index=False,
                )

            data = load_excel_files(
                campaign_data_path=(
                    zip_path
                ),
                campaign_description_path=(
                    description_path
                ),
            )

            self.assertIn(
                "activities",
                data,
            )

            self.assertIn(
                "aggregation",
                data,
            )

            self.assertIn(
                "desc_tasks",
                data,
            )

            self.assertEqual(
                str(
                    data[
                        "activities"
                    ].iloc[
                        0
                    ][
                        "pid"
                    ]
                ),
                "10",
            )

    def test_loads_explicit_raw_data_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = (
                Path(
                    tmp
                )
                / "data_raw"
            )

            raw_dir.mkdir()

            (
                raw_dir
                / "player_497_test.json"
            ).write_text(
                json.dumps(
                    [
                        {
                            "X_PLAYER_ID": (
                                497
                            ),
                            "value": (
                                10
                            ),
                        }
                    ]
                ),
                encoding="utf-8",
            )

            data = load_json_files(
                raw_data_dir=(
                    raw_dir
                )
            )

            self.assertIn(
                "player_497_test",
                data,
            )

            frame = data[
                "player_497_test"
            ]

            self.assertIsInstance(
                frame,
                pd.DataFrame,
            )

            self.assertEqual(
                len(
                    frame
                ),
                1,
            )

    def test_missing_raw_directory_returns_empty(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = (
                Path(
                    tmp
                )
                / "missing"
            )

            data = load_json_files(
                raw_data_dir=(
                    missing
                )
            )

            self.assertEqual(
                data,
                {},
            )


if __name__ == "__main__":
    unittest.main()