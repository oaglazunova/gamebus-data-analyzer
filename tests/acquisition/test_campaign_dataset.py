from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from collections.abc import Callable

from src.acquisition.campaign_dataset import (
    bootstrap_campaign_dataset,
)
from src.datasets.manifests import (
    read_manifest,
)


class TestCampaignDatasetBootstrap(
    unittest.TestCase
):

    def test_bootstrap_preserves_filename_and_writes_manifest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            datasets_dir = Path(temp_dir)

            def fake_download(
                *,
                campaign_abbreviation,
                destination_dir,
                **kwargs,
            ):
                self.assertEqual(
                    campaign_abbreviation,
                    "HW8 BIPS",
                )

                path = (
                    Path(destination_dir)
                    / "campaign-456.xlsx"
                )

                path.write_bytes(
                    b"fake xlsx content"
                )

                return path

            def fake_export_download(
                *,
                campaign_abbreviation,
                destination_dir,
                **kwargs,
            ):
                self.assertEqual(
                    campaign_abbreviation,
                    "HW8 BIPS",
                )

                path = (
                    Path(destination_dir)
                    / "campaign-456-export.zip"
                )

                path.write_bytes(
                    b"fake zip content"
                )

                return path

            with patch(
                "src.acquisition.campaign_dataset."
                "download_campaign_description",
                side_effect=fake_download,
            ), patch(
                "src.acquisition.campaign_dataset."
                "download_campaign_data_export",
                side_effect=fake_export_download,
            ):
                dataset_dir = (
                    bootstrap_campaign_dataset(
                        campaign_abbreviation="HW8 BIPS",
                        datasets_dir=datasets_dir,
                        extracted_at=datetime(
                            2026,
                            9,
                            14,
                            13,
                            45,
                        ),
                    )
                )

            self.assertEqual(
                dataset_dir.name,
                (
                    "HW8_BIPS_456_"
                    "2026-09-14_1345"
                ),
            )

            description = (
                dataset_dir
                / "campaign-456.xlsx"
            )

            self.assertTrue(
                description.exists()
            )

            self.assertEqual(
                description.read_bytes(),
                b"fake xlsx content",
            )

            data_export = (
                dataset_dir
                / "campaign-456-export.zip"
            )

            self.assertTrue(
                data_export.exists()
            )

            self.assertEqual(
                data_export.read_bytes(),
                b"fake zip content",
            )

            manifest = read_manifest(
                dataset_dir
                / "extraction_manifest.json"
            )

            self.assertEqual(
                manifest["campaign"][
                    "abbreviation"
                ],
                "HW8 BIPS",
            )

            self.assertEqual(
                manifest["campaign"]["id"],
                "456",
            )

            self.assertEqual(
                manifest["campaign_files"][
                    "data_export"
                ],
                "campaign-456-export.zip",
            )

            self.assertEqual(
                manifest["campaign_files"][
                    "description"
                ],
                "campaign-456.xlsx",
            )

            self.assertFalse(
                manifest[
                    "participant_credentials"
                ]["supplied"]
            )

    def test_failed_bootstrap_does_not_leave_dataset(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            datasets_dir = Path(temp_dir)

            with patch(
                "src.acquisition.campaign_dataset."
                "download_campaign_description",
                side_effect=RuntimeError(
                    "download failed"
                ),
            ):
                with self.assertRaises(
                    RuntimeError
                ):
                    bootstrap_campaign_dataset(
                        campaign_abbreviation=(
                            "HW8 BIPS"
                        ),
                        datasets_dir=datasets_dir,
                    )

            self.assertEqual(
                list(
                    datasets_dir.iterdir()
                ),
                [],
            )


if __name__ == "__main__":
    unittest.main()