from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

from src.acquisition.participant_credentials import (
    build_initial_cohort_candidates,
    load_participant_credentials,
)


class TestParticipantCredentials(
    unittest.TestCase
):

    def _write_credentials(
        self,
        path: Path,
        rows: list[list[object]],
    ) -> None:
        workbook = Workbook()

        sheet = workbook.active

        sheet.append(
            [
                "Email",
                "Password",
                "Study arm",
            ]
        )

        for row in rows:
            sheet.append(
                row
            )

        workbook.save(
            path
        )

        workbook.close()

    def test_credentials_are_loaded_without_extra_columns(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = (
                Path(temp_dir)
                / "HW8 BIPS 456 - users.xlsx"
            )

            self._write_credentials(
                path,
                [
                    [
                        "A@example.org",
                        "secret-a",
                        "intervention",
                    ],
                    [
                        "B@example.org",
                        "secret-b",
                        "control",
                    ],
                ],
            )

            credentials = (
                load_participant_credentials(
                    path
                )
            )

            self.assertEqual(
                credentials.count,
                2,
            )

            self.assertTrue(
                credentials.has_email(
                    "a@example.org"
                )
            )

            self.assertEqual(
                credentials.get_password(
                    "B@example.org"
                ),
                "secret-b",
            )

    def test_repr_does_not_reveal_password(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = (
                Path(temp_dir)
                / "users.xlsx"
            )

            self._write_credentials(
                path,
                [
                    [
                        "a@example.org",
                        "very-secret-password",
                        "x",
                    ]
                ],
            )

            credentials = (
                load_participant_credentials(
                    path
                )
            )

            representation = repr(
                credentials
            )

            self.assertNotIn(
                "very-secret-password",
                representation,
            )

            self.assertIn(
                "count=1",
                representation,
            )

    def test_without_credentials_all_campaign_users_are_selected(
        self,
    ) -> None:
        campaign_users = [
            {
                "account_id": "1",
                "pid": "101",
                "email": "a@example.org",
            },
            {
                "account_id": "2",
                "pid": "102",
                "email": "b@example.org",
            },
        ]

        candidates, summary = (
            build_initial_cohort_candidates(
                campaign_users=campaign_users,
                credentials=None,
            )
        )

        self.assertTrue(
            all(
                candidate[
                    "selected_for_analysis"
                ]
                for candidate
                in candidates
            )
        )

        self.assertTrue(
            all(
                not candidate[
                    "credentials_available"
                ]
                for candidate
                in candidates
            )
        )

        self.assertTrue(
            all(
                not candidate[
                    "selected_for_participant_extraction"
                ]
                for candidate
                in candidates
            )
        )

        self.assertEqual(
            summary.campaign_accounts,
            2,
        )

    def test_with_credentials_only_matches_are_preselected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = (
                Path(temp_dir)
                / "users.xlsx"
            )

            self._write_credentials(
                path,
                [
                    [
                        "b@example.org",
                        "secret-b",
                        "x",
                    ],
                    [
                        "outside@example.org",
                        "secret-outside",
                        "x",
                    ],
                ],
            )

            credentials = (
                load_participant_credentials(
                    path
                )
            )

            campaign_users = [
                {
                    "account_id": "1",
                    "pid": "101",
                    "email": "a@example.org",
                },
                {
                    "account_id": "2",
                    "pid": "102",
                    "email": "b@example.org",
                },
            ]

            candidates, summary = (
                build_initial_cohort_candidates(
                    campaign_users=(
                        campaign_users
                    ),
                    credentials=credentials,
                )
            )

            first = candidates[0]
            second = candidates[1]

            self.assertFalse(
                first[
                    "selected_for_analysis"
                ]
            )

            self.assertFalse(
                first[
                    "credentials_available"
                ]
            )

            self.assertTrue(
                second[
                    "selected_for_analysis"
                ]
            )

            self.assertTrue(
                second[
                    "credentials_available"
                ]
            )

            self.assertTrue(
                second[
                    "selected_for_participant_extraction"
                ]
            )

            self.assertEqual(
                summary.matched_accounts,
                1,
            )

            self.assertEqual(
                summary.credentials_not_in_campaign,
                1,
            )

    def test_duplicate_credentials_are_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = (
                Path(temp_dir)
                / "users.xlsx"
            )

            self._write_credentials(
                path,
                [
                    [
                        "A@example.org",
                        "secret-a",
                        "x",
                    ],
                    [
                        "a@example.org",
                        "secret-b",
                        "x",
                    ],
                ],
            )

            with self.assertRaisesRegex(
                ValueError,
                "duplicate participant email",
            ):
                load_participant_credentials(
                    path
                )


if __name__ == "__main__":
    unittest.main()