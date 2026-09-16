from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.datasets.campaign_users_snapshot import (
    build_campaign_users_snapshot,
    read_campaign_users_snapshot,
    write_campaign_users_snapshot,
)


class TestCampaignUsersSnapshot(
    unittest.TestCase
):

    def test_snapshot_keeps_only_safe_fields(
        self,
    ) -> None:
        snapshot = (
            build_campaign_users_snapshot(
                campaign_abbreviation=(
                    "HW8_YA_HB"
                ),
                campaign_id=283,
                users=[
                    {
                        "account_id": "501",
                        "pid": "497",
                        "email": (
                            "user@example.org"
                        ),
                        "password": "secret",
                        "password_hash": (
                            "hash"
                        ),
                        "activation_token": (
                            "token"
                        ),
                    }
                ],
            )
        )

        self.assertEqual(
            snapshot[
                "accounts"
            ],
            [
                {
                    "account_id": "501",
                    "pid": "497",
                    "email": (
                        "user@example.org"
                    ),
                }
            ],
        )

        serialized = str(
            snapshot
        ).lower()

        self.assertNotIn(
            "secret",
            serialized,
        )

        self.assertNotIn(
            "password_hash",
            serialized,
        )

        self.assertNotIn(
            "activation_token",
            serialized,
        )

    def test_snapshot_round_trip(
        self,
    ) -> None:
        snapshot = (
            build_campaign_users_snapshot(
                campaign_abbreviation="TEST",
                campaign_id="123",
                users=[
                    {
                        "account_id": "10",
                        "pid": "20",
                        "email": (
                            "a@example.org"
                        ),
                    }
                ],
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / "campaign_users.json"
            )

            write_campaign_users_snapshot(
                path,
                snapshot,
            )

            loaded = (
                read_campaign_users_snapshot(
                    path
                )
            )

        self.assertEqual(
            loaded,
            snapshot,
        )

    def test_missing_values_are_allowed(
        self,
    ) -> None:
        snapshot = (
            build_campaign_users_snapshot(
                campaign_abbreviation="TEST",
                campaign_id="123",
                users=[
                    {
                        "account_id": "10",
                        "pid": None,
                        "email": None,
                    }
                ],
            )
        )

        self.assertEqual(
            snapshot[
                "accounts"
            ][
                0
            ],
            {
                "account_id": "10",
                "pid": None,
                "email": None,
            },
        )


if __name__ == "__main__":
    unittest.main()