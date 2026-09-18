from __future__ import annotations

from datetime import datetime, timezone
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile

import pandas as pd

from src.trajectory.audit_run import (
    TrajectoryAuditRunError,
    read_campaign_export_participant_ids,
    run_trajectory_audit_snapshot,
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


def _write_campaign_xlsx(
    path: Path,
) -> None:
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


class TestTrajectoryAuditRun(
    unittest.TestCase
):

    def test_reads_historical_candidate_cohort(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            campaign_zip = (
                Path(tmp)
                / "campaign-283-export.zip"
            )

            _write_campaign_zip(
                campaign_zip
            )

            self.assertEqual(
                read_campaign_export_participant_ids(
                    campaign_zip
                ),
                [
                    497,
                    498,
                ],
            )

    @patch(
        "src.trajectory.audit_run."
        "run_trajectory_audit"
    )
    def test_run_is_saved_outside_dataset_structure(
        self,
        mocked_audit,
    ) -> None:
        mocked_audit.return_value = {
            "audit_summary": {
                "cohort": {
                    "participants": 1,
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(
                tmp
            )

            source_dir = (
                root
                / "source"
            )

            output_root = (
                root
                / "trajectory_audits"
            )

            source_dir.mkdir()

            campaign_zip = (
                source_dir
                / "campaign-283-export.zip"
            )

            campaign_xlsx = (
                source_dir
                / "campaign-283.xlsx"
            )

            _write_campaign_zip(
                campaign_zip
            )

            _write_campaign_xlsx(
                campaign_xlsx
            )

            created_at = datetime(
                2026,
                9,
                17,
                15,
                30,
                0,
                tzinfo=timezone.utc,
            )

            result = (
                run_trajectory_audit_snapshot(
                    campaign_data_path=(
                        campaign_zip
                    ),
                    campaign_description_path=(
                        campaign_xlsx
                    ),
                    candidate_participant_ids=[
                        497,
                        498,
                    ],
                    selected_participant_ids=[
                        497,
                    ],
                    source_type=(
                        "uploaded_files"
                    ),
                    output_root=(
                        output_root
                    ),
                    created_at=(
                        created_at
                    ),
                )
            )

            self.assertEqual(
                result.run_dir.name,
                "TEST_283_2026-09-17_153000",
            )

            self.assertEqual(
                result.run_dir.parent,
                output_root.resolve(),
            )

            self.assertEqual(
                result.campaign_data_path.read_bytes(),
                campaign_zip.read_bytes(),
            )

            self.assertEqual(
                result.campaign_description_path.read_bytes(),
                campaign_xlsx.read_bytes(),
            )

            cohort = json.loads(
                result.cohort_path.read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                cohort["source"],
                "campaign_export",
            )

            self.assertEqual(
                cohort["participants"],
                [
                    {
                        "pid": "497",
                        "included": True,
                    },
                    {
                        "pid": "498",
                        "included": False,
                    },
                ],
            )

            run_manifest = json.loads(
                result.run_manifest_path.read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                run_manifest["source_type"],
                "uploaded_files",
            )

            mocked_audit.assert_called_once()

            config = (
                mocked_audit.call_args.args[
                    0
                ]
            )

            self.assertEqual(
                Path(
                    config.output_dir
                ).resolve(),
                result.results_dir.resolve(),
            )

            self.assertEqual(
                config.analysis_participant_ids,
                {
                    497,
                },
            )

    @patch(
        "src.trajectory.audit_run."
        "run_trajectory_audit",
        return_value={},
    )
    def test_each_run_gets_a_new_folder(
        self,
        mocked_audit,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(
                tmp
            )

            campaign_zip = (
                root
                / "campaign-283-export.zip"
            )

            campaign_xlsx = (
                root
                / "campaign-283.xlsx"
            )

            _write_campaign_zip(
                campaign_zip
            )

            _write_campaign_xlsx(
                campaign_xlsx
            )

            when = datetime(
                2026,
                9,
                17,
                15,
                30,
                0,
                tzinfo=timezone.utc,
            )

            first = run_trajectory_audit_snapshot(
                campaign_data_path=(
                    campaign_zip
                ),
                campaign_description_path=(
                    campaign_xlsx
                ),
                candidate_participant_ids=[
                    497,
                    498,
                ],
                selected_participant_ids=[
                    497,
                ],
                source_type="gamebus",
                output_root=(
                    root
                    / "audits"
                ),
                created_at=when,
            )

            second = run_trajectory_audit_snapshot(
                campaign_data_path=(
                    campaign_zip
                ),
                campaign_description_path=(
                    campaign_xlsx
                ),
                candidate_participant_ids=[
                    497,
                    498,
                ],
                selected_participant_ids=[
                    497,
                ],
                source_type="gamebus",
                output_root=(
                    root
                    / "audits"
                ),
                created_at=when,
            )

            self.assertNotEqual(
                first.run_dir,
                second.run_dir,
            )

            self.assertTrue(
                first.run_dir.exists()
            )

            self.assertTrue(
                second.run_dir.exists()
            )

            self.assertEqual(
                second.run_dir.name,
                "TEST_283_2026-09-17_153000_2",
            )

            self.assertEqual(
                mocked_audit.call_count,
                2,
            )

    def test_selected_participants_must_be_candidates(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(
                tmp
            )

            campaign_zip = (
                root
                / "campaign-283-export.zip"
            )

            campaign_xlsx = (
                root
                / "campaign-283.xlsx"
            )

            _write_campaign_zip(
                campaign_zip
            )

            _write_campaign_xlsx(
                campaign_xlsx
            )

            with self.assertRaisesRegex(
                TrajectoryAuditRunError,
                "not present in the candidate cohort",
            ):
                run_trajectory_audit_snapshot(
                    campaign_data_path=(
                        campaign_zip
                    ),
                    campaign_description_path=(
                        campaign_xlsx
                    ),
                    candidate_participant_ids=[
                        497,
                    ],
                    selected_participant_ids=[
                        999,
                    ],
                    source_type=(
                        "uploaded_files"
                    ),
                    output_root=(
                        root
                        / "audits"
                    ),
                )


if __name__ == "__main__":
    unittest.main()