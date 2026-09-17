from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from src.datasets.layout import (
    build_dataset_name,
    create_dataset_directory,
    get_analysis_dir,
    get_cohort_manifest_path,
    get_extraction_manifest_path,
    get_raw_data_dir,
)


class TestDatasetLayout(unittest.TestCase):

    def test_build_dataset_name(self) -> None:
        name = build_dataset_name(
            "UNISG",
            379,
            datetime(2026, 9, 11, 13, 43),
        )

        self.assertEqual(
            name,
            "UNISG_379_2026-09-11_1343",
        )

    def test_spaces_are_normalized(self) -> None:
        name = build_dataset_name(
            "Healthy W8 DK",
            456,
            datetime(2026, 9, 13, 22, 30),
        )

        self.assertEqual(
            name,
            "Healthy_W8_DK_456_2026-09-13_2230",
        )

    def test_create_dataset_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)

            dataset_dir = create_dataset_directory(
                "UNISG",
                379,
                datetime(2026, 9, 11, 13, 43),
                datasets_dir=base,
            )

            self.assertTrue(
                dataset_dir.exists()
            )

            self.assertTrue(
                get_raw_data_dir(
                    dataset_dir
                ).exists()
            )

            self.assertFalse(
                get_analysis_dir(
                    dataset_dir
                ).exists()
            )

            self.assertEqual(
                get_cohort_manifest_path(
                    dataset_dir
                ),
                dataset_dir
                / "cohort_manifest.json",
            )

            self.assertEqual(
                get_extraction_manifest_path(
                    dataset_dir
                ),
                dataset_dir
                / "extraction_manifest.json",
            )

    def test_existing_dataset_directory_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)

            create_dataset_directory(
                "UNISG",
                379,
                datetime(2026, 9, 11, 13, 43),
                datasets_dir=base,
            )

            with self.assertRaises(
                FileExistsError
            ):
                create_dataset_directory(
                    "UNISG",
                    379,
                    datetime(
                        2026,
                        9,
                        11,
                        13,
                        43,
                    ),
                    datasets_dir=base,
                )


if __name__ == "__main__":
    unittest.main()