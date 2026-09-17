from __future__ import annotations

from typing import Any

import requests

from src.acquisition.gamebus_campaigns import (
    CampaignDownloadError,
    DEFAULT_CAMPAIGNS_BASE_URL,
    load_session_cookies,
    login_organizer,
    save_session_cookies,
)


def _find_users_data(
    payload: dict[str, Any],
) -> list[Any]:
    """
    Locate the flattened SvelteKit data array containing
    the campaign user list.
    """
    nodes = payload.get("nodes")

    if not isinstance(nodes, list):
        raise CampaignDownloadError(
            "Unexpected campaign-users response: "
            "missing nodes."
        )

    for node in nodes:
        if not isinstance(node, dict):
            continue

        data = node.get("data")

        if not isinstance(data, list):
            continue

        if not data:
            continue

        root = data[0]

        if (
            isinstance(root, dict)
            and "users" in root
        ):
            return data

    raise CampaignDownloadError(
        "Could not locate the campaign user list "
        "in the GameBus response."
    )


def _value_at(
    data: list[Any],
    reference: Any,
) -> Any:
    """
    Resolve one reference in the flattened SvelteKit payload.

    GameBus Studio serializes records as an array in which
    object fields contain indexes into that same array.
    """
    if not isinstance(reference, int):
        return reference

    if (
        reference < 0
        or reference >= len(data)
    ):
        raise CampaignDownloadError(
            "Invalid reference in GameBus "
            "campaign-users response."
        )

    return data[reference]


def parse_campaign_users(
    payload: dict[str, Any],
) -> list[dict[str, str | None]]:
    """
    Convert the raw GameBus Studio users response into a
    deliberately minimal safe representation.

    Returned fields:

        account_id
        pid
        email

    Sensitive fields from the Studio response, including
    password hashes and reset/activation tokens, are never
    copied into the result.
    """
    data = _find_users_data(
        payload
    )

    root = data[0]

    users_reference = root[
        "users"
    ]

    user_references = _value_at(
        data,
        users_reference,
    )

    if not isinstance(
        user_references,
        list,
    ):
        raise CampaignDownloadError(
            "Unexpected campaign-users response: "
            "users is not a list."
        )

    users: list[
        dict[str, str | None]
    ] = []

    for user_reference in user_references:
        user = _value_at(
            data,
            user_reference,
        )

        if not isinstance(
            user,
            dict,
        ):
            raise CampaignDownloadError(
                "Unexpected campaign-users response: "
                "user record is not an object."
            )

        account_id_value = _value_at(
            data,
            user.get("id"),
        )

        email_value = _value_at(
            data,
            user.get("email"),
        )

        player = _value_at(
            data,
            user.get(
                "linkToPlayer"
            ),
        )

        pid_value = None

        if isinstance(
            player,
            dict,
        ):
            pid_value = _value_at(
                data,
                player.get("id"),
            )

        account_id = (
            None
            if account_id_value is None
            else str(account_id_value)
        )

        email = (
            None
            if email_value is None
            else str(email_value).strip()
        )

        if email == "":
            email = None

        pid = (
            None
            if pid_value is None
            else str(pid_value)
        )

        users.append(
            {
                "account_id": account_id,
                "pid": pid,
                "email": email,
            }
        )

    return users


def _fetch_campaign_users_with_session(
    *,
    session: requests.Session,
    base_url: str,
    campaign_id: int | str,
    timeout: int,
) -> list[dict[str, str | None]] | None:
    """
    Fetch and immediately normalize the campaign user list.

    The raw Studio payload is never written to disk.

    Returns None if the organizer session is unauthorized.
    """
    response = session.get(
        (
            f"{base_url}/editor/for/"
            f"{campaign_id}/users/"
            "__data.json"
        ),
        params={
            "x-sveltekit-invalidated": (
                "001"
            )
        },
        timeout=timeout,
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
            "Campaign-user retrieval failed: "
            f"{exc}"
        ) from exc

    content_type = (
        response.headers.get(
            "Content-Type",
            "",
        )
    )

    if (
        "application/json"
        not in content_type.lower()
    ):
        raise CampaignDownloadError(
            "Unexpected campaign-users response "
            f"type: {content_type!r}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise CampaignDownloadError(
            "Campaign-users response was not "
            "valid JSON."
        ) from exc

    if not isinstance(
        payload,
        dict,
    ):
        raise CampaignDownloadError(
            "Unexpected campaign-users JSON "
            "structure."
        )

    return parse_campaign_users(
        payload
    )


def list_campaign_users(
    *,
    campaign_id: int | str,
    organizer_email: str | None = None,
    organizer_password: str | None = None,
    cookie_file=None,
    base_url: str = (
        DEFAULT_CAMPAIGNS_BASE_URL
    ),
    timeout_login: int = 30,
    timeout: int = 60,
) -> list[dict[str, str | None]]:
    """
    Retrieve all campaign accounts visible to the organizer.

    Only account ID, player ID (pid), and email are returned.
    Raw Studio user records are never persisted.
    """
    base_url = (
        base_url
        .strip()
        .rstrip("/")
    )

    campaign_id_text = str(
        campaign_id
    ).strip()

    if not campaign_id_text:
        raise CampaignDownloadError(
            "Campaign ID is empty."
        )

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": (
                "GameBus-Data-Analyzer"
            )
        }
    )

    if (
        cookie_file is not None
        and load_session_cookies(
            session,
            cookie_file,
        )
    ):
        users = (
            _fetch_campaign_users_with_session(
                session=session,
                base_url=base_url,
                campaign_id=(
                    campaign_id_text
                ),
                timeout=timeout,
            )
        )

        if users is not None:
            return users

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

    users = (
        _fetch_campaign_users_with_session(
            session=session,
            base_url=base_url,
            campaign_id=campaign_id_text,
            timeout=timeout,
        )
    )

    if users is None:
        raise CampaignDownloadError(
            "Campaign-user retrieval remained "
            "unauthorized after organizer login."
        )

    return users