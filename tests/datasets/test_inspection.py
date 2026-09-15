from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import zipfile

from src.datasets.inspection import (
    DatasetValidationError,
    inspect_dataset,
)


def _write_campaign_zip(
    path: Path,
) -> None:
    with zipfile.ZipFile(
        path,
        "w",
    ) as archive:
        archive.writestr(
            "1-aggregated-data.csv",
            "pid,value\n1,10\n",
        )

        archive.writestr(
            "2-activities.csv",
            "pid,type\n1,WALK\n",
        )

        archive.writestr(
            "3-navigation-events.csv",
            "pid\n1\n",
        )

        archive.writestr(
            "4-notification-events.csv",
            "pid\n1\n",
        )

        archive.writestr(
            "5-sensor-events.csv",
            "pid\n1\n",
        )


class TestDatasetInspection(
    unittest.TestCase
):

    def test_new_dataset_is_inspected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(
                tmp
            )

            data_path = (
                dataset_dir
                / "campaign-456-export.zip"
            )

            description_path = (
                dataset_dir
                / "campaign-456.xlsx"
            )

            _write_campaign_zip(
                data_path
            )

            description_path.write_bytes(
                b"xlsx-placeholder"
            )

            extraction_manifest = {
                "schema_version": 1,
                "campaign": {
                    "abbreviation": (
                        "HW8 BIPS"
                    ),
                    "id": "456",
                },
                "campaign_files": {
                    "data_export": (
                        data_path.name
                    ),
                    "description": (
                        description_path.name
                    ),
                },
            }

            cohort_manifest = {
                "schema_version": 1,
                "campaign": {
                    "abbreviation": (
                        "HW8 BIPS"
                    ),
                    "id": "456",
                },
                "participants": [
                    {
                        "pid": "1",
                        "email": (
                            "a@example.org"
                        ),
                        "selected_for_analysis": (
                            True
                        ),
                    },
                    {
                        "pid": "2",
                        "email": (
                            "b@example.org"
                        ),
                        "selected_for_analysis": (
                            False
                        ),
                    },
                ],
            }

            (
                dataset_dir
                / "extraction_manifest.json"
            ).write_text(
                json.dumps(
                    extraction_manifest
                ),
                encoding="utf-8",
            )

            (
                dataset_dir
                / "cohort_manifest.json"
            ).write_text(
                json.dumps(
                    cohort_manifest
                ),
                encoding="utf-8",
            )

            inspection = (
                inspect_dataset(
                    dataset_dir
                )
            )

            self.assertEqual(
                inspection.campaign_id,
                "456",
            )

            self.assertEqual(
                inspection.campaign_abbreviation,
                "HW8 BIPS",
            )

            self.assertTrue(
                inspection.has_extraction_manifest
            )

            self.assertTrue(
                inspection.has_cohort_manifest
            )

            self.assertEqual(
                inspection.participant_count,
                2,
            )

            self.assertEqual(
                inspection.selected_for_analysis_count,
                1,
            )

    def test_legacy_dataset_is_discovered(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(
                tmp
            )

            data_path = (
                dataset_dir
                / "campaign_data.zip"
            )

            description_path = (
                dataset_dir
                / "campaign_desc.xlsx"
            )

            _write_campaign_zip(
                data_path
            )

            description_path.write_bytes(
                b"xlsx-placeholder"
            )

            inspection = (
                inspect_dataset(
                    dataset_dir
                )
            )

            self.assertFalse(
                inspection.has_extraction_manifest
            )

            self.assertFalse(
                inspection.has_cohort_manifest
            )

            self.assertEqual(
                inspection.campaign_data_path,
                data_path.resolve(),
            )

            self.assertEqual(
                inspection.campaign_description_path,
                description_path.resolve(),
            )

    def test_missing_campaign_data_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(
                tmp
            )

            (
                dataset_dir
                / "campaign_desc.xlsx"
            ).write_bytes(
                b"xlsx-placeholder"
            )

            with self.assertRaises(
                DatasetValidationError
            ):
                inspect_dataset(
                    dataset_dir
                )

    def test_manifest_campaign_mismatch_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(
                tmp
            )

            data_path = (
                dataset_dir
                / "campaign-456-export.zip"
            )

            description_path = (
                dataset_dir
                / "campaign-456.xlsx"
            )

            _write_campaign_zip(
                data_path
            )

            description_path.write_bytes(
                b"xlsx-placeholder"
            )

            (
                dataset_dir
                / "extraction_manifest.json"
            ).write_text(
                json.dumps(
                    {
                        "campaign": {
                            "abbreviation": (
                                "HW8 BIPS"
                            ),
                            "id": "456",
                        },
                        "campaign_files": {
                            "data_export": (
                                data_path.name
                            ),
                            "description": (
                                description_path.name
                            ),
                        },
                    }
                ),
                encoding="utf-8",
            )

            (
                dataset_dir
                / "cohort_manifest.json"
            ).write_text(
                json.dumps(
                    {
                        "campaign": {
                            "abbreviation": (
                                "HW8 BIPS"
                            ),
                            "id": "999",
                        },
                        "participants": [],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                DatasetValidationError,
                "Campaign ID",
            ):
                inspect_dataset(
                    dataset_dir
                )


if __name__ == "__main__":
    unittest.main()