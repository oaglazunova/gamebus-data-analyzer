from __future__ import annotations

from typing import Any


def _normalize_pid(
    value: Any,
) -> str | None:
    """
    Convert a GameBus player ID to a stable string.

    Examples:
        497      -> "497"
        497.0    -> "497"
        "497.0"  -> "497"
    """
    if value is None:
        return None

    if isinstance(
        value,
        int,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        float,
    ):
        if value.is_integer():
            return str(
                int(
                    value
                )
            )

        return str(
            value
        ).strip()

    text = str(
        value
    ).strip()

    if not text:
        return None

    if text.casefold() in {
        "nan",
        "none",
        "null",
    }:
        return None

    if (
        text.endswith(
            ".0"
        )
        and text[:-2].isdigit()
    ):
        return text[:-2]

    return text


def resolve_analysis_user_ids_from_manifest(
    cohort_manifest: dict,
) -> set[str]:
    """
    Return the GameBus player IDs currently selected for
    analysis in cohort_manifest.json.

    Acquisition history is deliberately ignored here.

    In particular:

        participant_data_extracted=True

    does NOT imply that the participant is currently included
    in analysis. Only selected_for_analysis controls this.
    """
    participants = cohort_manifest.get(
        "participants"
    )

    if not isinstance(
        participants,
        list,
    ):
        raise ValueError(
            "cohort_manifest.json does not contain "
            "a valid participants list."
        )

    selected_ids: set[str] = set()

    for participant in participants:
        if not isinstance(
            participant,
            dict,
        ):
            raise ValueError(
                "cohort_manifest.json contains an "
                "invalid participant record."
            )

        if not participant.get(
            "selected_for_analysis",
            False,
        ):
            continue

        pid = _normalize_pid(
            participant.get(
                "pid"
            )
        )

        if pid is None:
            email = (
                participant.get(
                    "email"
                )
                or "unknown participant"
            )

            raise ValueError(
                "Participant selected for analysis "
                "does not have a GameBus player ID: "
                f"{email}"
            )

        selected_ids.add(
            pid
        )

    if not selected_ids:
        raise ValueError(
            "No participants are currently "
            "selected for analysis."
        )

    return selected_ids