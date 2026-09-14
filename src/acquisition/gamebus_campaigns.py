from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional
from urllib.parse import quote, unquote

import requests


DEFAULT_CAMPAIGNS_BASE_URL = (
    "https://campaigns.healthyw8.gamebus.eu"
)


class CampaignDownloadError(RuntimeError):
    """Raised when organizer login or campaign download fails."""


def save_session_cookies(
    session: requests.Session,
    cookie_file: Path,
) -> None:
    """
    Persist only the authenticated session cookies.

    Organizer passwords are not written here. Later, the UI can
    use the OS keyring exactly as the Campaign Assistant does.
    """
    cookie_file = Path(cookie_file)

    cookie_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cookies = requests.utils.dict_from_cookiejar(
        session.cookies
    )

    cookie_file.write_text(
        json.dumps(
            cookies,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_session_cookies(
    session: requests.Session,
    cookie_file: Path,
) -> bool:
    """
    Load previously stored organizer session cookies.

    Returns False when no usable cookie file exists.
    """
    cookie_file = Path(cookie_file)

    if not cookie_file.exists():
        return False

    try:
        data = json.loads(
            cookie_file.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(
            data,
            dict,
        ):
            return False

        session.cookies.update(
            data
        )

        return True

    except Exception:
        return False


def extract_download_filename(
    response: requests.Response,
) -> str:
    """
    Extract the original filename supplied by GameBus in the
    Content-Disposition response header.

    The filename is preserved rather than replaced by an
    application-defined name.
    """
    content_disposition = (
        response.headers.get(
            "Content-Disposition",
            "",
        )
    )

    if not content_disposition:
        raise CampaignDownloadError(
            "GameBus did not provide a "
            "Content-Disposition filename."
        )

    # RFC 5987 form:
    #
    # filename*=UTF-8''UNISG%20campaign.xlsx
    extended_match = re.search(
        r"filename\*\s*=\s*"
        r"(?:UTF-8'')?"
        r"([^;]+)",
        content_disposition,
        flags=re.IGNORECASE,
    )

    if extended_match:
        filename = unquote(
            extended_match.group(1)
        ).strip().strip("\"'")

        filename = Path(
            filename
        ).name

        if filename:
            return filename

    # Regular form:
    #
    # filename="UNISG_CAMPAGNA_STUDY - campaign-379.xlsx"
    regular_match = re.search(
        r'filename\s*=\s*"([^"]+)"',
        content_disposition,
        flags=re.IGNORECASE,
    )

    if regular_match:
        filename = Path(
            regular_match.group(1).strip()
        ).name

        if filename:
            return filename

    regular_match = re.search(
        r"filename\s*=\s*([^;]+)",
        content_disposition,
        flags=re.IGNORECASE,
    )

    if regular_match:
        filename = Path(
            regular_match.group(1)
            .strip()
            .strip("\"'")
        ).name

        if filename:
            return filename

    raise CampaignDownloadError(
        "Could not determine the original "
        "GameBus download filename."
    )


def extract_campaign_id_from_filename(
    filename: str,
) -> str:
    """
    Extract the GameBus campaign ID from its standard filename.

    Example:

        UNISG_CAMPAGNA_STUDY - campaign-379.xlsx

    becomes:

        379
    """
    match = re.search(
        r"campaign-(\d+)",
        filename,
        flags=re.IGNORECASE,
    )

    if not match:
        raise CampaignDownloadError(
            "Could not determine campaign ID "
            f"from filename: {filename}"
        )

    return match.group(1)


def _validate_xlsx_response(
    response: requests.Response,
) -> None:
    content_type = (
        response.headers.get(
            "Content-Type",
            "",
        )
    )

    content_disposition = (
        response.headers.get(
            "Content-Disposition",
            "",
        )
    )

    if (
        "spreadsheetml.sheet"
        in content_type.lower()
    ):
        return

    if (
        ".xlsx"
        in content_disposition.lower()
    ):
        return

    raise CampaignDownloadError(
        "Unexpected campaign-description "
        "response type. "
        f"Content-Type={content_type!r}, "
        "Content-Disposition="
        f"{content_disposition!r}"
    )


def _download_description_with_session(
    *,
    session: requests.Session,
    base_url: str,
    campaign_abbreviation: str,
    destination_dir: Path,
    timeout_download: int,
) -> Optional[Path]:
    """
    Try the GameBus Studio campaign-description download
    with an already authenticated session.

    Returns None when the session is unauthorized/expired.
    """
    response = session.post(
        (
            f"{base_url}/api/campaigns/"
            f"{campaign_abbreviation}/download"
        ),
        timeout=timeout_download,
        stream=True,
    )

    if response.status_code in {
        401,
        403,
    }:
        return None

    try:
        response.raise_for_status()
    except requests.RequestException as exc:
        raise CampaignDownloadError(
            "Campaign-description download "
            f"failed: {exc}"
        ) from exc

    _validate_xlsx_response(
        response
    )

    filename = extract_download_filename(
        response
    )

    destination_dir = Path(
        destination_dir
    )

    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        destination_dir
        / filename
    )

    with output_path.open(
        "wb"
    ) as handle:
        for chunk in response.iter_content(
            chunk_size=8192
        ):
            if chunk:
                handle.write(
                    chunk
                )

    return output_path


def login_organizer(
    *,
    session: requests.Session,
    base_url: str,
    email: str,
    password: str,
    timeout_login: int = 30,
) -> None:
    """
    Authenticate a campaign organizer using the same
    GameBus Studio mechanism used by the Campaign Assistant.
    """
    email = email.strip()

    if not email:
        raise CampaignDownloadError(
            "Organizer email is empty."
        )

    if not password:
        raise CampaignDownloadError(
            "Organizer password is empty."
        )

    try:
        response = session.post(
            f"{base_url}/api/auth/token",
            json={
                "email": email,
                "password": password,
            },
            timeout=timeout_login,
        )

        response.raise_for_status()

    except requests.RequestException as exc:
        raise CampaignDownloadError(
            f"Organizer login failed: {exc}"
        ) from exc

    if (
        "__session"
        not in session.cookies
    ):
        raise CampaignDownloadError(
            "Organizer login succeeded, "
            "but GameBus did not provide "
            "a __session cookie."
        )


def download_campaign_description(
    *,
    campaign_abbreviation: str,
    destination_dir: Path,
    organizer_email: str | None = None,
    organizer_password: str | None = None,
    cookie_file: Path | None = None,
    base_url: str = (
        DEFAULT_CAMPAIGNS_BASE_URL
    ),
    timeout_login: int = 30,
    timeout_download: int = 60,
) -> Path:
    """
    Download the GameBus campaign-description XLSX.

    Strategy:

    1. Reuse saved organizer session cookies when available.
    2. If the session is unavailable/expired, authenticate with
       organizer email/password.
    3. Save refreshed cookies when a cookie_file was supplied.
    4. Preserve the original filename supplied by GameBus.

    The participant credentials XLSX is unrelated to this login.
    """
    base_url = (
        base_url
        .strip()
        .rstrip("/")
    )

    campaign_abbreviation = (
        campaign_abbreviation
        .strip()
    )

    if not base_url:
        raise CampaignDownloadError(
            "GameBus campaigns base URL is empty."
        )

    if not campaign_abbreviation:
        raise CampaignDownloadError(
            "Campaign abbreviation is empty."
        )

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": (
                "GameBus-Data-Analyzer"
            )
        }
    )

    # First try an existing organizer session.
    if (
        cookie_file is not None
        and load_session_cookies(
            session,
            cookie_file,
        )
    ):
        path = (
            _download_description_with_session(
                session=session,
                base_url=base_url,
                campaign_abbreviation=(
                    campaign_abbreviation
                ),
                destination_dir=(
                    destination_dir
                ),
                timeout_download=(
                    timeout_download
                ),
            )
        )

        if path is not None:
            return path

    # Existing session unavailable or expired:
    # organizer credentials are required.
    email = (
        organizer_email or ""
    ).strip()

    password = (
        organizer_password or ""
    )

    if (
        not email
        or not password
    ):
        raise CampaignDownloadError(
            "No valid organizer session was found, "
            "and organizer credentials were not "
            "provided."
        )

    login_organizer(
        session=session,
        base_url=base_url,
        email=email,
        password=password,
        timeout_login=timeout_login,
    )

    if cookie_file is not None:
        save_session_cookies(
            session,
            cookie_file,
        )

    path = (
        _download_description_with_session(
            session=session,
            base_url=base_url,
            campaign_abbreviation=(
                campaign_abbreviation
            ),
            destination_dir=(
                destination_dir
            ),
            timeout_download=(
                timeout_download
            ),
        )
    )

    if path is None:
        raise CampaignDownloadError(
            "Campaign-description download "
            "remained unauthorized after "
            "successful organizer login."
        )

    return path



def _validate_zip_response(
    response: requests.Response,
) -> None:
    content_type = (
        response.headers.get(
            "Content-Type",
            "",
        )
    )

    content_disposition = (
        response.headers.get(
            "Content-Disposition",
            "",
        )
    )

    if (
        "zip"
        in content_type.lower()
    ):
        return

    if (
        ".zip"
        in content_disposition.lower()
    ):
        return

    raise CampaignDownloadError(
        "Unexpected campaign-data export "
        "response type. "
        f"Content-Type={content_type!r}, "
        "Content-Disposition="
        f"{content_disposition!r}"
    )


def _download_campaign_data_with_session(
    *,
    session: requests.Session,
    base_url: str,
    campaign_abbreviation: str,
    destination_dir: Path,
    timeout_download: int,
) -> Optional[Path]:
    """
    Download the campaign analytics/data ZIP using an
    already authenticated organizer session.

    Returns None when the organizer session is no longer
    authorized.
    """
    encoded_abbreviation = quote(
        campaign_abbreviation,
        safe="",
    )

    response = session.post(
        (
            f"{base_url}/api/campaigns/"
            f"{encoded_abbreviation}/analytics/download"
        ),
        timeout=timeout_download,
        stream=True,
    )

    if response.status_code in {
        401,
        403,
    }:
        return None

    try:
        response.raise_for_status()
    except requests.RequestException as exc:
        raise CampaignDownloadError(
            "Campaign-data export download "
            f"failed: {exc}"
        ) from exc

    _validate_zip_response(
        response
    )

    filename = extract_download_filename(
        response
    )

    destination_dir = Path(
        destination_dir
    )

    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        destination_dir
        / filename
    )

    with output_path.open(
        "wb"
    ) as handle:
        for chunk in response.iter_content(
            chunk_size=8192
        ):
            if chunk:
                handle.write(
                    chunk
                )

    return output_path


def download_campaign_data_export(
    *,
    campaign_abbreviation: str,
    destination_dir: Path,
    organizer_email: str | None = None,
    organizer_password: str | None = None,
    cookie_file: Path | None = None,
    base_url: str = (
        DEFAULT_CAMPAIGNS_BASE_URL
    ),
    timeout_login: int = 30,
    timeout_download: int = 120,
) -> Path:
    """
    Download the organizer-accessible campaign-data ZIP.

    Uses the same authentication/session mechanism as the
    campaign-description download and preserves the original
    filename supplied by GameBus.
    """
    base_url = (
        base_url
        .strip()
        .rstrip("/")
    )

    campaign_abbreviation = (
        campaign_abbreviation
        .strip()
    )

    if not base_url:
        raise CampaignDownloadError(
            "GameBus campaigns base URL is empty."
        )

    if not campaign_abbreviation:
        raise CampaignDownloadError(
            "Campaign abbreviation is empty."
        )

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": (
                "GameBus-Data-Analyzer"
            )
        }
    )

    # First try the saved organizer session.
    if (
        cookie_file is not None
        and load_session_cookies(
            session,
            cookie_file,
        )
    ):
        path = (
            _download_campaign_data_with_session(
                session=session,
                base_url=base_url,
                campaign_abbreviation=(
                    campaign_abbreviation
                ),
                destination_dir=(
                    destination_dir
                ),
                timeout_download=(
                    timeout_download
                ),
            )
        )

        if path is not None:
            return path

    # Saved session unavailable or expired.
    email = (
        organizer_email or ""
    ).strip()

    password = (
        organizer_password or ""
    )

    if (
        not email
        or not password
    ):
        raise CampaignDownloadError(
            "No valid organizer session was found, "
            "and organizer credentials were not "
            "provided."
        )

    login_organizer(
        session=session,
        base_url=base_url,
        email=email,
        password=password,
        timeout_login=timeout_login,
    )

    if cookie_file is not None:
        save_session_cookies(
            session,
            cookie_file,
        )

    path = (
        _download_campaign_data_with_session(
            session=session,
            base_url=base_url,
            campaign_abbreviation=(
                campaign_abbreviation
            ),
            destination_dir=(
                destination_dir
            ),
            timeout_download=(
                timeout_download
            ),
        )
    )

    if path is None:
        raise CampaignDownloadError(
            "Campaign-data export remained "
            "unauthorized after successful "
            "organizer login."
        )

    return path



