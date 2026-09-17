from __future__ import annotations

import unittest
import tempfile
from pathlib import Path



from src.acquisition.gamebus_campaigns import (
    CampaignDownloadError,
    extract_campaign_id_from_filename,
    extract_download_filename,
    _download_campaign_data_with_session,
    _validate_zip_response,
    )


class _FakeResponse:

    def __init__(
        self,
        content_disposition: str,
    ) -> None:
        self.headers = {
            "Content-Disposition": (
                content_disposition
            )
        }


class TestGameBusCampaignAcquisition(
    unittest.TestCase
):

    def test_original_filename_is_preserved(
        self,
    ) -> None:
        response = _FakeResponse(
            'attachment; filename="'
            "UNISG_CAMPAGNA_STUDY "
            '- campaign-379.xlsx"'
        )

        filename = (
            extract_download_filename(
                response
            )
        )

        self.assertEqual(
            filename,
            (
                "UNISG_CAMPAGNA_STUDY "
                "- campaign-379.xlsx"
            ),
        )

    def test_encoded_filename_is_decoded(
        self,
    ) -> None:
        response = _FakeResponse(
            "attachment; "
            "filename*=UTF-8''"
            "UNISG%20CAMPAGNA%20STUDY"
            "%20-%20campaign-379.xlsx"
        )

        filename = (
            extract_download_filename(
                response
            )
        )

        self.assertEqual(
            filename,
            (
                "UNISG CAMPAGNA STUDY "
                "- campaign-379.xlsx"
            ),
        )

    def test_campaign_id_is_extracted(
        self,
    ) -> None:
        campaign_id = (
            extract_campaign_id_from_filename(
                (
                    "UNISG_CAMPAGNA_STUDY "
                    "- campaign-379.xlsx"
                )
            )
        )

        self.assertEqual(
            campaign_id,
            "379",
        )

    def test_missing_campaign_id_is_rejected(
        self,
    ) -> None:
        with self.assertRaises(
            CampaignDownloadError
        ):
            extract_campaign_id_from_filename(
                "UNISG.xlsx"
            )

    def test_missing_download_filename_is_rejected(
        self,
    ) -> None:
        response = _FakeResponse(
            ""
        )

        with self.assertRaises(
            CampaignDownloadError
        ):
            extract_download_filename(
                response
            )

    def test_campaign_data_export_endpoint(
            self,
    ) -> None:
        session = _FakeSession()

        with tempfile.TemporaryDirectory() as temp_dir:
            path = (
                _download_campaign_data_with_session(
                    session=session,
                    base_url=(
                        "https://campaigns."
                        "healthyw8.gamebus.eu"
                    ),
                    campaign_abbreviation=(
                        "HW8 BIPS"
                    ),
                    destination_dir=Path(
                        temp_dir
                    ),
                    timeout_download=60,
                )
            )

            self.assertEqual(
                session.requested_url,
                (
                    "https://campaigns."
                    "healthyw8.gamebus.eu/"
                    "api/campaigns/"
                    "HW8%20BIPS/"
                    "analytics/download"
                ),
            )

            self.assertIsNotNone(
                path
            )

            self.assertEqual(
                path.name,
                (
                    "campaign-456-export-on-"
                    "2026-09-14-1415.zip"
                ),
            )

    def test_non_zip_export_is_rejected(
            self,
    ) -> None:
        response = _FakeResponse(
            'attachment; filename="error.html"'
        )

        response.headers[
            "Content-Type"
        ] = "text/html"

        with self.assertRaises(
                CampaignDownloadError
        ):
            _validate_zip_response(
                response
            )


class _FakeZipResponse:

    def __init__(self) -> None:
        self.status_code = 200

        self.headers = {
            "Content-Type": "application/zip",
            "Content-Disposition": (
                'attachment; filename="'
                "campaign-456-export-on-"
                '2026-09-14-1415.zip"'
            ),
        }

    def raise_for_status(
        self,
    ) -> None:
        return None

    def iter_content(
        self,
        chunk_size: int,
    ):
        yield b"fake zip content"


class _FakeSession:

    def __init__(self) -> None:
        self.requested_url = None

    def post(
        self,
        url,
        timeout,
        stream,
    ):
        self.requested_url = url

        return _FakeZipResponse()



if __name__ == "__main__":
    unittest.main()