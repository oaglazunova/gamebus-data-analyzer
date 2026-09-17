from __future__ import annotations

import unittest

from src.analysis.cohort import (
    resolve_analysis_user_ids_from_manifest,
)


class TestAnalysisCohort(
    unittest.TestCase
):

    def test_only_selected_participants_are_returned(
        self,
    ) -> None:
        manifest = {
            "participants": [
                {
                    "pid": "497",
                    "email": "a@example.org",
                    "selected_for_analysis": True,
                },
                {
                    "pid": "498",
                    "email": "b@example.org",
                    "selected_for_analysis": False,
                },
            ]
        }

        result = (
            resolve_analysis_user_ids_from_manifest(
                manifest
            )
        )

        self.assertEqual(
            result,
            {
                "497",
            },
        )

    def test_extracted_participant_can_be_excluded(
        self,
    ) -> None:
        manifest = {
            "participants": [
                {
                    "pid": "497",
                    "selected_for_analysis": False,
                    "credentials_available": True,
                    "selected_for_participant_extraction": True,
                    "participant_data_extracted": True,
                },
                {
                    "pid": "498",
                    "selected_for_analysis": True,
                    "participant_data_extracted": False,
                },
            ]
        }

        result = (
            resolve_analysis_user_ids_from_manifest(
                manifest
            )
        )

        self.assertEqual(
            result,
            {
                "498",
            },
        )

    def test_player_ids_are_normalized(
        self,
    ) -> None:
        manifest = {
            "participants": [
                {
                    "pid": 497.0,
                    "selected_for_analysis": True,
                },
                {
                    "pid": "498.0",
                    "selected_for_analysis": True,
                },
            ]
        }

        result = (
            resolve_analysis_user_ids_from_manifest(
                manifest
            )
        )

        self.assertEqual(
            result,
            {
                "497",
                "498",
            },
        )

    def test_selected_participant_requires_pid(
        self,
    ) -> None:
        manifest = {
            "participants": [
                {
                    "pid": None,
                    "email": "a@example.org",
                    "selected_for_analysis": True,
                }
            ]
        }

        with self.assertRaisesRegex(
            ValueError,
            "does not have a GameBus player ID",
        ):
            resolve_analysis_user_ids_from_manifest(
                manifest
            )

    def test_empty_analysis_cohort_is_rejected(
        self,
    ) -> None:
        manifest = {
            "participants": [
                {
                    "pid": "497",
                    "selected_for_analysis": False,
                }
            ]
        }

        with self.assertRaisesRegex(
            ValueError,
            "No participants",
        ):
            resolve_analysis_user_ids_from_manifest(
                manifest
            )


if __name__ == "__main__":
    unittest.main()