from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.acquisition.participant_data import (
    extract_participant_data,
)


class FakeCredentials:
    def __init__(
        self,
        passwords: dict[str, str],
    ):
        self.passwords = passwords

    def get_password(
        self,
        email: str,
    ):
        return self.passwords.get(
            email.casefold()
        )


class ParticipantDataExtractionTests(
    unittest.TestCase
):
    @patch(
        "src.acquisition.participant_data."
        "require_authcode",
        return_value="test-authcode",
    )
    @patch(
        "src.acquisition.participant_data."
        "AllDataCollector",
    )
    @patch(
        "src.acquisition.participant_data."
        "GameBusClient",
    )
    def test_extracts_only_selected_participants(
        self,
        client_class,
        collector_class,
        require_authcode_mock,
    ) -> None:
        client = Mock()

        client.get_user_token.return_value = (
            "token"
        )

        client.get_user_id.return_value = (
            123,
            "a@example.org",
        )

        client_class.return_value = client

        collector = Mock()

        collector.collect.return_value = (
            {
                "steps": [
                    {
                        "value": 1000
                    }
                ]
            },
            [
                "player_123_steps.json",
                "player_123_all_raw.json",
            ],
        )

        collector_class.return_value = (
            collector
        )

        participants = [
            {
                "pid": "123",
                "email": (
                    "a@example.org"
                ),
                "selected_for_participant_extraction": (
                    True
                ),
            },
            {
                "pid": "456",
                "email": (
                    "b@example.org"
                ),
                "selected_for_participant_extraction": (
                    False
                ),
            },
        ]

        credentials = FakeCredentials(
            {
                "a@example.org": (
                    "password-a"
                ),
                "b@example.org": (
                    "password-b"
                ),
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            results = (
                extract_participant_data(
                    dataset_dir=Path(
                        tmp
                    ),
                    participants=participants,
                    credentials=credentials,
                )
            )

        self.assertEqual(
            len(results),
            1,
        )

        self.assertTrue(
            results[0].succeeded
        )

        self.assertEqual(
            results[0].expected_pid,
            "123",
        )

        client.get_user_token.assert_called_once_with(
            "a@example.org",
            "password-a",
        )

        collector_class.assert_called_once()

    @patch(
        "src.acquisition.participant_data."
        "require_authcode",
        return_value="test-authcode",
    )
    @patch(
        "src.acquisition.participant_data."
        "AllDataCollector",
    )
    @patch(
        "src.acquisition.participant_data."
        "GameBusClient",
    )
    def test_pid_mismatch_is_not_extracted(
        self,
        client_class,
        collector_class,
        require_authcode_mock,
    ) -> None:
        client = Mock()

        client.get_user_token.return_value = (
            "token"
        )

        client.get_user_id.return_value = (
            999,
            "a@example.org",
        )

        client_class.return_value = client

        participants = [
            {
                "pid": "123",
                "email": (
                    "a@example.org"
                ),
                "selected_for_participant_extraction": (
                    True
                ),
            }
        ]

        credentials = FakeCredentials(
            {
                "a@example.org": (
                    "password-a"
                )
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            results = (
                extract_participant_data(
                    dataset_dir=Path(
                        tmp
                    ),
                    participants=participants,
                    credentials=credentials,
                )
            )

        self.assertEqual(
            results[0].status,
            "pid_mismatch",
        )

        collector_class.assert_not_called()

    @patch(
        "src.acquisition.participant_data."
        "require_authcode",
        return_value="test-authcode",
    )
    @patch(
        "src.acquisition.participant_data."
        "GameBusClient",
    )
    def test_missing_credentials_are_not_used(
        self,
        client_class,
        require_authcode_mock,
    ) -> None:
        participants = [
            {
                "pid": "123",
                "email": (
                    "a@example.org"
                ),
                "selected_for_participant_extraction": (
                    True
                ),
            }
        ]

        credentials = FakeCredentials(
            {}
        )

        with tempfile.TemporaryDirectory() as tmp:
            results = (
                extract_participant_data(
                    dataset_dir=Path(
                        tmp
                    ),
                    participants=participants,
                    credentials=credentials,
                )
            )

        self.assertEqual(
            results[0].status,
            "missing_credentials",
        )

        client_class.return_value\
            .get_user_token\
            .assert_not_called()


if __name__ == "__main__":
    unittest.main()