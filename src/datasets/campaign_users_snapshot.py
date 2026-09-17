from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1


def _optional_text(
    value: Any,
) -> str | None:
    if value is None:
        return None

    text = str(value).strip()

    return (
        text
        if text
        else None
    )


def build_campaign_users_snapshot(
    *,
    campaign_abbreviation: str,
    campaign_id: int | str,
    users: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build a safe snapshot of the campaign accounts returned
    by GameBus Studio.

    Only explicitly allowed fields are retained. Any other
    fields in the input records are discarded.
    """
    accounts = []

    for user in users:
        account_id = _optional_text(
            user.get(
                "account_id"
            )
        )

        pid = _optional_text(
            user.get(
                "pid"
            )
        )

        email = _optional_text(
            user.get(
                "email"
            )
        )

        accounts.append(
            {
                "account_id": account_id,
                "pid": pid,
                "email": email,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "campaign": {
            "abbreviation": str(
                campaign_abbreviation
            ).strip(),
            "id": str(
                campaign_id
            ).strip(),
        },
        "source": (
            "gamebus_studio_users"
        ),
        "accounts": accounts,
        "count": len(
            accounts
        ),
    }


def write_campaign_users_snapshot(
    path: str | Path,
    snapshot: dict[str, Any],
) -> None:
    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            snapshot,
            handle,
            indent=2,
            ensure_ascii=False,
        )


def read_campaign_users_snapshot(
    path: str | Path,
) -> dict[str, Any]:
    path = Path(
        path
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        snapshot = json.load(
            handle
        )

    accounts = snapshot.get(
        "accounts"
    )

    if not isinstance(
        accounts,
        list,
    ):
        raise ValueError(
            "campaign_users.json does not "
            "contain a valid accounts list."
        )

    for account in accounts:
        if not isinstance(
            account,
            dict,
        ):
            raise ValueError(
                "campaign_users.json contains "
                "an invalid account record."
            )

        unexpected = (
            set(
                account
            )
            - {
                "account_id",
                "pid",
                "email",
            }
        )

        if unexpected:
            raise ValueError(
                "campaign_users.json contains "
                "unexpected account fields: "
                + ", ".join(
                    sorted(
                        unexpected
                    )
                )
            )

    return snapshot