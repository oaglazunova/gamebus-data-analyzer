from __future__ import annotations

import unittest

from src.acquisition.campaign_users import (
    parse_campaign_users,
)


class TestCampaignUsers(
    unittest.TestCase
):

    def test_parse_campaign_users(
        self,
    ) -> None:
        payload = {
            "type": "data",
            "nodes": [
                {
                    "type": "skip",
                },
                {
                    "type": "data",
                    "data": [
                        {
                            "users": 1,
                        },
                        [
                            2,
                        ],
                        {
                            "id": 3,
                            "email": 4,
                            "password": 5,
                            "password_reset_token": 6,
                            "activation_token": 6,
                            "linkToPlayer": 7,
                        },
                        312,
                        "participant@example.org",
                        (
                            "{bcrypt}$2a$10$"
                            "secret-hash"
                        ),
                        None,
                        {
                            "id": 8,
                        },
                        308,
                    ],
                },
            ],
        }

        users = parse_campaign_users(
            payload
        )

        self.assertEqual(
            users,
            [
                {
                    "account_id": "312",
                    "pid": "308",
                    "email": (
                        "participant@example.org"
                    ),
                }
            ],
        )

    def test_sensitive_fields_are_not_returned(
        self,
    ) -> None:
        payload = {
            "type": "data",
            "nodes": [
                {
                    "type": "data",
                    "data": [
                        {
                            "users": 1,
                        },
                        [
                            2,
                        ],
                        {
                            "id": 3,
                            "email": 4,
                            "password": 5,
                            "linkToPlayer": 6,
                        },
                        312,
                        "participant@example.org",
                        "secret-hash",
                        {
                            "id": 7,
                        },
                        308,
                    ],
                }
            ],
        }

        user = parse_campaign_users(
            payload
        )[0]

        self.assertEqual(
            set(user),
            {
                "account_id",
                "pid",
                "email",
            },
        )

        self.assertNotIn(
            "password",
            user,
        )

    def test_account_id_and_pid_are_distinct(
        self,
    ) -> None:
        payload = {
            "type": "data",
            "nodes": [
                {
                    "type": "data",
                    "data": [
                        {
                            "users": 1,
                        },
                        [
                            2,
                        ],
                        {
                            "id": 3,
                            "email": 4,
                            "linkToPlayer": 5,
                        },
                        312,
                        "participant@example.org",
                        {
                            "id": 6,
                        },
                        308,
                    ],
                }
            ],
        }

        user = parse_campaign_users(
            payload
        )[0]

        self.assertEqual(
            user["account_id"],
            "312",
        )

        self.assertEqual(
            user["pid"],
            "308",
        )


if __name__ == "__main__":
    unittest.main()