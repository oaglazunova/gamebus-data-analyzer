from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from src.datasets.manifests import (
    build_cohort_manifest,
    build_extraction_manifest,
    read_manifest,
    write_manifest,
)


class TestDatasetManifests(unittest.TestCase):

    def test_cohort_manifest_keeps_analysis_and_extraction_separate(
        self,
    ) -> None:
        manifest = build_cohort_manifest(
            campaign_abbreviation="UNISG",
            campaign_id=379,
            verified_at=datetime(
                2026,
                9,
                11,
                14,
                0,
            ),
            participants=[
                {
                    "pid": 245,
                    "email": (
                        "participant1@example.org"
                    ),
                    "selected_for_analysis": True,
                    "credentials_available": True,
                    "selected_for_participant_extraction": True,
                    "participant_data_extracted": True,
                },
                {
                    "pid": 978,
                    "email": (
                        "participant2@example.org"
                    ),
                    "selected_for_analysis": True,
                    "credentials_available": False,
                },
            ],
        )

        self.assertEqual(
            manifest["counts"][
                "selected_for_analysis"
            ],
            2,
        )

        self.assertEqual(
            manifest["counts"][
                "selected_for_participant_extraction"
            ],
            1,
        )

        self.assertEqual(
            manifest["counts"][
                "participant_data_extracted"
            ],
            1,
        )

    def test_selected_extraction_requires_analysis_selection(
        self,
    ) -> None:
        with self.assertRaises(
            ValueError
        ):
            build_cohort_manifest(
                campaign_abbreviation="UNISG",
                campaign_id=379,
                participants=[
                    {
                        "pid": 245,
                        "email": (
                            "participant@example.org"
                        ),
                        "selected_for_analysis": False,
                        "credentials_available": True,
                        "selected_for_participant_extraction": True,
                    }
                ],
            )

    def test_selected_extraction_requires_credentials(
        self,
    ) -> None:
        with self.assertRaises(
            ValueError
        ):
            build_cohort_manifest(
                campaign_abbreviation="UNISG",
                campaign_id=379,
                participants=[
                    {
                        "pid": 245,
                        "email": (
                            "participant@example.org"
                        ),
                        "selected_for_analysis": True,
                        "credentials_available": False,
                        "selected_for_participant_extraction": True,
                    }
                ],
            )

    def test_password_is_rejected(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Sensitive credential field",
        ):
            build_cohort_manifest(
                campaign_abbreviation="UNISG",
                campaign_id=379,
                participants=[
                    {
                        "pid": 245,
                        "email": (
                            "participant@example.org"
                        ),
                        "password": "secret",
                    }
                ],
            )

    def test_extraction_manifest_does_not_store_credentials_file(
        self,
    ) -> None:
        manifest = build_extraction_manifest(
            campaign_abbreviation="UNISG",
            campaign_id=379,
            extracted_at=datetime(
                2026,
                9,
                11,
                13,
                43,
            ),
            campaign_data_filename=(
                "campaign-379-export-on-"
                "2026-09-11-1343.zip"
            ),
            campaign_description_filename=(
                "UNISG_CAMPAGNA_STUDY "
                "- campaign-379.xlsx"
            ),
            credentials_supplied=True,
            credentials_source_filename=(
                "UNISG-379-users.xlsx"
            ),
        )

        credentials = manifest[
            "participant_credentials"
        ]

        self.assertTrue(
            credentials["supplied"]
        )

        self.assertFalse(
            credentials[
                "stored_in_dataset"
            ]
        )

        self.assertEqual(
            credentials[
                "source_filename"
            ],
            "UNISG-379-users.xlsx",
        )

    def test_manifest_round_trip(
        self,
    ) -> None:
        manifest = build_extraction_manifest(
            campaign_abbreviation="UNISG",
            campaign_id=379,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = (
                Path(temp_dir)
                / "extraction_manifest.json"
            )

            write_manifest(
                path,
                manifest,
            )

            loaded = read_manifest(
                path
            )

            self.assertEqual(
                loaded,
                manifest,
            )


if __name__ == "__main__":
    unittest.main()