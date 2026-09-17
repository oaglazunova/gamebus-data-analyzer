from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import keyring


APP_NAME = "GameBus Data Analyzer"
APP_ID = "gamebus_data_analyzer"

KEYRING_SERVICE_NAME = APP_NAME

LOCAL_APPDATA = Path(
    os.environ.get(
        "LOCALAPPDATA",
        Path.home(),
    )
)

APP_DIR = (
    LOCAL_APPDATA
    / APP_ID
)

SETTINGS_FILE = (
    APP_DIR
    / "settings.json"
)

COOKIE_FILE = (
    APP_DIR
    / "session_cookies.json"
)


DEFAULT_SETTINGS: dict[str, Any] = {
    "organizer_email": "",
    "remember_credentials": True,
    "last_campaign_abbreviation": "",
    "campaign_abbreviations": [],
    "datasets_dir": "",
}


def ensure_app_dir() -> None:
    APP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def _normalize_settings(
    settings: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = DEFAULT_SETTINGS.copy()

    if isinstance(
        settings,
        dict,
    ):
        merged.update(
            settings
        )

    if not isinstance(
        merged.get(
            "organizer_email"
        ),
        str,
    ):
        merged[
            "organizer_email"
        ] = ""

    if not isinstance(
        merged.get(
            "last_campaign_abbreviation"
        ),
        str,
    ):
        merged[
            "last_campaign_abbreviation"
        ] = ""

    if not isinstance(
        merged.get(
            "remember_credentials"
        ),
        bool,
    ):
        merged[
            "remember_credentials"
        ] = True

    if not isinstance(
        merged.get(
            "datasets_dir"
        ),
        str,
    ):
        merged[
            "datasets_dir"
        ] = ""

    campaigns = merged.get(
        "campaign_abbreviations"
    )

    if not isinstance(
        campaigns,
        list,
    ):
        campaigns = []

    normalized_campaigns = []

    for campaign in campaigns:
        if not isinstance(
            campaign,
            str,
        ):
            continue

        campaign = campaign.strip()

        if (
            campaign
            and campaign
            not in normalized_campaigns
        ):
            normalized_campaigns.append(
                campaign
            )

    merged[
        "organizer_email"
    ] = (
        merged[
            "organizer_email"
        ]
        .strip()
    )

    merged[
        "last_campaign_abbreviation"
    ] = (
        merged[
            "last_campaign_abbreviation"
        ]
        .strip()
    )

    merged[
        "datasets_dir"
    ] = (
        merged[
            "datasets_dir"
        ]
        .strip()
    )

    last_campaign = merged[
        "last_campaign_abbreviation"
    ]

    if (
        last_campaign
        and last_campaign
        not in normalized_campaigns
    ):
        normalized_campaigns.append(
            last_campaign
        )

    merged[
        "campaign_abbreviations"
    ] = normalized_campaigns

    return merged


def load_settings() -> dict[str, Any]:
    """
    Load non-sensitive UI settings.

    Passwords are never stored in this JSON file.
    """
    ensure_app_dir()

    if not SETTINGS_FILE.exists():
        save_settings(
            DEFAULT_SETTINGS
        )

        return (
            DEFAULT_SETTINGS.copy()
        )

    try:
        data = json.loads(
            SETTINGS_FILE.read_text(
                encoding="utf-8"
            )
        )

        return _normalize_settings(
            data
        )

    except Exception:
        save_settings(
            DEFAULT_SETTINGS
        )

        return (
            DEFAULT_SETTINGS.copy()
        )


def save_settings(
    settings: dict[str, Any],
) -> None:
    """
    Save only non-sensitive local settings.
    """
    ensure_app_dir()

    normalized = (
        _normalize_settings(
            settings
        )
    )

    SETTINGS_FILE.write_text(
        json.dumps(
            normalized,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def save_organizer_password(
    email: str,
    password: str,
) -> None:
    """
    Store organizer password in the operating-system keyring.

    On Windows this is handled by the available keyring backend
    rather than being written into the project or dataset files.
    """
    email = email.strip()

    if not email:
        raise ValueError(
            "Organizer email is empty."
        )

    if not password:
        raise ValueError(
            "Organizer password is empty."
        )

    keyring.set_password(
        KEYRING_SERVICE_NAME,
        email,
        password,
    )


def load_organizer_password(
    email: str,
) -> str | None:
    email = email.strip()

    if not email:
        return None

    return keyring.get_password(
        KEYRING_SERVICE_NAME,
        email,
    )


def delete_organizer_password(
    email: str,
) -> None:
    email = email.strip()

    if not email:
        return

    try:
        keyring.delete_password(
            KEYRING_SERVICE_NAME,
            email,
        )

    except Exception:
        pass


def get_cookie_file() -> Path:
    """
    Return the local GameBus organizer-session cookie file.

    This file is outside the repository and outside dataset
    folders.
    """
    ensure_app_dir()

    return COOKIE_FILE


def delete_cookie_file() -> None:
    try:
        get_cookie_file().unlink(
            missing_ok=True
        )

    except Exception:
        pass