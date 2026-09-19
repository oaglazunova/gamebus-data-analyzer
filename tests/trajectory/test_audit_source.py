from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile
from datetime import datetime, timezone

import pandas as pd

from src.trajectory.audit_source import (
    TrajectoryAuditSource,
    prepare_gamebus_trajectory_source,
    prepare_uploaded_trajectory_source,
    remove_trajectory_source,
)


def _campaign_zip_bytes() -> bytes:
    with tempfile.NamedTemporaryFile(
        suffix=".zip",
        delete=False,
    ) as handle:
        path = Path(
            handle.name
        )

    try:
        with zipfile.ZipFile(
            path,
            "w",
        ) as archive:
            archive.writestr(
                "1-aggregated-data.csv",
                (
                    "pid,numberOfActivities\n"
                    "497,1\n"
                    "498,0\n"
                ),
            )

            archive.writestr(
                "2-activities.csv",
                (
                    "pid,type\n"
                    "497,WALK\n"
                ),
            )

            archive.writestr(
                "3-navigation-events.csv",
                "pid\n497\n",
            )

            archive.writestr(
                "4-notification-events.csv",
                "pid\n497\n",
            )

            archive.writestr(
                "5-sensor-events.csv",
                "pid\n497\n",
            )

        return path.read_bytes()

    finally:
        path.unlink(
            missing_ok=True
        )


def _campaign_xlsx_bytes() -> bytes:
    with tempfile.NamedTemporaryFile(
        suffix=".xlsx",
        delete=False,
    ) as handle:
        path = Path(
            handle.name
        )

    try:
        with pd.ExcelWriter(
            path,
            engine="openpyxl",
        ) as writer:
            pd.DataFrame(
                {
                    "id": [
                        283,
                    ],
                    "abbreviation": [
                        "TEST",
                    ],
                }
            ).to_excel(
                writer,
                sheet_name="campaigns",
                index=False,
            )

        return path.read_bytes()

    finally:
        path.unlink(
            missing_ok=True
        )


class TestTrajectoryAuditSource(
    unittest.TestCase
):

    def test_uploaded_source_extracts_campaign_and_pids(
        self,
    ) -> None:
        source = (
            prepare_uploaded_trajectory_source(
                campaign_description_filename=(
                    "campaign-283.xlsx"
                ),
                campaign_description_bytes=(
                    _campaign_xlsx_bytes()
                ),
                campaign_data_filename=(
                    "campaign-283-export.zip"
                ),
                campaign_data_bytes=(
                    _campaign_zip_bytes()
                ),
            )
        )

        try:
            self.assertEqual(
                source.source_type,
                "uploaded_files",
            )

            self.assertEqual(
                source.campaign_abbreviation,
                "TEST",
            )

            self.assertEqual(
                source.campaign_id,
                "283",
            )

            self.assertEqual(
                [
                    participant.pid
                    for participant
                    in source.participants
                ],
                [
                    497,
                    498,
                ],
            )

            self.assertTrue(
                source.campaign_data_path.exists()
            )

            self.assertTrue(
                source
                .campaign_description_path
                .exists()
            )

        finally:
            staging_dir = (
                source.staging_dir
            )

            remove_trajectory_source(
                source
            )

            self.assertFalse(
                staging_dir.exists()
            )

    @patch(
        "src.trajectory.audit_source."
        "list_campaign_users"
    )
    @patch(
        "src.trajectory.audit_source."
        "download_campaign_data_export"
    )
    @patch(
        "src.trajectory.audit_source."
        "download_campaign_description"
    )
    def test_gamebus_source_reuses_campaign_access(
        self,
        mock_description,
        mock_data,
        mock_users,
    ) -> None:
        staging_paths = []

        def write_description(
            **kwargs,
        ):
            path = (
                kwargs[
                    "destination_dir"
                ]
                / "campaign-283.xlsx"
            )

            path.write_bytes(
                _campaign_xlsx_bytes()
            )

            staging_paths.append(
                path
            )

            return path

        def write_data(
            **kwargs,
        ):
            path = (
                kwargs[
                    "destination_dir"
                ]
                / "campaign-283-export.zip"
            )

            path.write_bytes(
                _campaign_zip_bytes()
            )

            staging_paths.append(
                path
            )

            return path

        mock_description.side_effect = (
            write_description
        )

        mock_data.side_effect = (
            write_data
        )

        mock_users.return_value = [
            {
                "account_id": "501",
                "pid": "497",
                "email": "one@example.org",
            },
            {
                "account_id": "502",
                "pid": "498",
                "email": "two@example.org",
            },
            {
                "account_id": "503",
                "pid": None,
                "email": "no-player@example.org",
            },
        ]

        before_snapshot = datetime.now(
            timezone.utc
        )

        source = (
            prepare_gamebus_trajectory_source(
                campaign_abbreviation=(
                    "TEST"
                ),
                organizer_email=(
                    "organizer@example.org"
                ),
                organizer_password=(
                    "secret"
                ),
                cookie_file=Path(
                    "cookies.json"
                ),
            )
        )

        after_snapshot = datetime.now(
            timezone.utc
        )

        try:
            self.assertEqual(
                source.source_type,
                "gamebus",
            )

            self.assertEqual(
                source.campaign_id,
                "283",
            )

            self.assertEqual(
                source.accounts_without_pid,
                1,
            )

            self.assertEqual(
                [
                    participant.pid
                    for participant
                    in source.participants
                ],
                [
                    497,
                    498,
                ],
            )

            self.assertEqual(
                source.participants[
                    0
                ].email,
                "one@example.org",
            )

            mock_description.assert_called_once()

            mock_data.assert_called_once()

            mock_users.assert_called_once()

            self.assertIsNotNone(
                source.snapshot_time
            )

            self.assertGreaterEqual(
                source.snapshot_time,
                before_snapshot,
            )

            self.assertLessEqual(
                source.snapshot_time,
                after_snapshot,
            )

        finally:
            remove_trajectory_source(
                source
            )


if __name__ == "__main__":
    unittest.main()