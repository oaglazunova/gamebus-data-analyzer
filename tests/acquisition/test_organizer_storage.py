from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.acquisition import organizer_storage


class TestOrganizerStorage(
    unittest.TestCase
):

    def test_settings_do_not_contain_password(
        self,
    ) -> None:
        normalized = (
            organizer_storage
            ._normalize_settings(
                {
                    "organizer_email": (
                        " organizer@example.org "
                    ),
                    "remember_credentials": True,
                    "last_campaign_abbreviation": (
                        " UNISG "
                    ),
                }
            )
        )

        self.assertEqual(
            normalized[
                "organizer_email"
            ],
            "organizer@example.org",
        )

        self.assertEqual(
            normalized[
                "last_campaign_abbreviation"
            ],
            "UNISG",
        )

        self.assertNotIn(
            "password",
            normalized,
        )

    def test_password_is_saved_via_keyring(
        self,
    ) -> None:
        with patch(
            "src.acquisition.organizer_storage."
            "keyring.set_password"
        ) as mocked:
            organizer_storage.save_organizer_password(
                "organizer@example.org",
                "secret",
            )

            mocked.assert_called_once_with(
                organizer_storage
                .KEYRING_SERVICE_NAME,
                "organizer@example.org",
                "secret",
            )

    def test_password_is_loaded_via_keyring(
        self,
    ) -> None:
        with patch(
            "src.acquisition.organizer_storage."
            "keyring.get_password",
            return_value="secret",
        ):
            value = (
                organizer_storage
                .load_organizer_password(
                    "organizer@example.org"
                )
            )

        self.assertEqual(
            value,
            "secret",
        )

    def test_cookie_file_is_outside_dataset_logic(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_dir = Path(
                temp_dir
            )

            with patch.object(
                organizer_storage,
                "APP_DIR",
                fake_dir,
            ), patch.object(
                organizer_storage,
                "COOKIE_FILE",
                (
                    fake_dir
                    / "session_cookies.json"
                ),
            ):
                cookie_file = (
                    organizer_storage
                    .get_cookie_file()
                )

                self.assertEqual(
                    cookie_file,
                    (
                        fake_dir
                        / "session_cookies.json"
                    ),
                )

                self.assertTrue(
                    fake_dir.exists()
                )


if __name__ == "__main__":
    unittest.main()