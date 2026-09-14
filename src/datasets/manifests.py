from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


MANIFEST_SCHEMA_VERSION = 1

_SENSITIVE_FIELD_NAMES = {
    "password",
    "passwd",
    "pwd",
}


def _timestamp(
    value: datetime | None = None,
) -> str:
    timestamp = (
        value
        if value is not None
        else datetime.now().astimezone()
    )

    return timestamp.isoformat(
        timespec="seconds"
    )


def _optional_text(
    value: Any,
) -> str | None:
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    return text


def _participant_id(
    value: Any,
) -> str | None:
    """
    Normalize a participant/player ID for manifests.

    Examples:
        454      -> "454"
        454.0    -> "454"
        "454.0"  -> "454"
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
        )

    text = str(
        value
    ).strip()

    if not text:
        return None

    if (
        text.endswith(
            ".0"
        )
        and text[:-2].isdigit()
    ):
        return text[:-2]

    return text


def _check_for_sensitive_fields(
    value: Any,
    path: str = "manifest",
) -> None:
    """
    Prevent participant passwords from accidentally being
    persisted in dataset manifests.
    """
    if isinstance(
        value,
        dict,
    ):
        for key, child in value.items():
            key_text = (
                str(
                    key
                )
                .strip()
                .lower()
            )

            if (
                key_text
                in _SENSITIVE_FIELD_NAMES
            ):
                raise ValueError(
                    "Sensitive credential field cannot "
                    "be stored in manifest: "
                    f"{path}.{key}"
                )

            _check_for_sensitive_fields(
                child,
                f"{path}.{key}",
            )

    elif isinstance(
        value,
        list,
    ):
        for index, child in enumerate(
            value
        ):
            _check_for_sensitive_fields(
                child,
                f"{path}[{index}]",
            )


def build_cohort_manifest(
    *,
    campaign_abbreviation: str,
    campaign_id: int | str,
    participants: Iterable[
        dict[str, Any]
    ],
    verified_at: datetime | None = None,
    candidate_source: str = "campaign_export",
) -> dict[str, Any]:
    """
    Build the researcher-confirmed cohort manifest.

    Each participant may contain:

        pid
        email
        selected_for_analysis
        credentials_available
        selected_for_participant_extraction
        participant_data_extracted

    Analysis selection and acquisition history are separate.

    A participant may therefore later be excluded from
    analysis even when participant-level data were previously
    selected for extraction or successfully extracted.

    Passwords must never be passed or persisted.
    """
    normalized_participants: list[
        dict[str, Any]
    ] = []

    for participant in participants:
        _check_for_sensitive_fields(
            participant,
            "participant",
        )

        pid = _participant_id(
            participant.get(
                "pid"
            )
        )

        email = _optional_text(
            participant.get(
                "email"
            )
        )

        if (
            pid is None
            and email is None
        ):
            raise ValueError(
                "Each participant must have at least "
                "a PID or email."
            )

        selected_for_analysis = bool(
            participant.get(
                "selected_for_analysis",
                False,
            )
        )

        credentials_available = bool(
            participant.get(
                "credentials_available",
                False,
            )
        )

        selected_for_extraction = bool(
            participant.get(
                "selected_for_participant_extraction",
                False,
            )
        )

        participant_data_extracted = bool(
            participant.get(
                "participant_data_extracted",
                False,
            )
        )

        # Extraction still requires credentials.
        #
        # We deliberately DO NOT require
        # selected_for_analysis=True here.
        #
        # Analysis selection may be changed later without
        # erasing the historical fact that participant-level
        # data were selected for or completed extraction.
        if (
            selected_for_extraction
            and not credentials_available
        ):
            raise ValueError(
                "Participant-level extraction cannot "
                "be selected when credentials are "
                "not available."
            )

        if (
            participant_data_extracted
            and not selected_for_extraction
        ):
            raise ValueError(
                "participant_data_extracted cannot be "
                "true unless participant-level "
                "extraction was selected."
            )

        normalized_participants.append(
            {
                "pid": (
                    pid
                ),
                "email": (
                    email
                ),
                "selected_for_analysis": (
                    selected_for_analysis
                ),
                "credentials_available": (
                    credentials_available
                ),
                "selected_for_participant_extraction": (
                    selected_for_extraction
                ),
                "participant_data_extracted": (
                    participant_data_extracted
                ),
            }
        )

    selected_analysis = sum(
        1
        for participant in (
            normalized_participants
        )
        if participant[
            "selected_for_analysis"
        ]
    )

    credentials_count = sum(
        1
        for participant in (
            normalized_participants
        )
        if participant[
            "credentials_available"
        ]
    )

    selected_extraction = sum(
        1
        for participant in (
            normalized_participants
        )
        if participant[
            "selected_for_participant_extraction"
        ]
    )

    extracted_count = sum(
        1
        for participant in (
            normalized_participants
        )
        if participant[
            "participant_data_extracted"
        ]
    )

    manifest = {
        "schema_version": (
            MANIFEST_SCHEMA_VERSION
        ),
        "campaign": {
            "abbreviation": (
                str(
                    campaign_abbreviation
                ).strip()
            ),
            "id": (
                str(
                    campaign_id
                ).strip()
            ),
        },
        "verification": {
            "verified_at": (
                _timestamp(
                    verified_at
                )
            ),
            "method": (
                "manual_verification"
            ),
            "candidate_source": (
                candidate_source
            ),
        },
        "counts": {
            "accounts_available": (
                len(
                    normalized_participants
                )
            ),
            "selected_for_analysis": (
                selected_analysis
            ),
            "credentials_available": (
                credentials_count
            ),
            "selected_for_participant_extraction": (
                selected_extraction
            ),
            "participant_data_extracted": (
                extracted_count
            ),
        },
        "participants": (
            normalized_participants
        ),
    }

    _check_for_sensitive_fields(
        manifest
    )

    return manifest


def build_extraction_manifest(
    *,
    campaign_abbreviation: str,
    campaign_id: int | str,
    extracted_at: datetime | None = None,
    campaign_data_filename: (
        str | None
    ) = None,
    campaign_description_filename: (
        str | None
    ) = None,
    credentials_supplied: bool = False,
    credentials_source_filename: (
        str | None
    ) = None,
) -> dict[str, Any]:
    """
    Build metadata describing how a dataset was acquired.

    The credentials XLSX itself is deliberately not stored
    in the dataset. Only its original filename may be
    recorded for traceability.
    """
    if (
        credentials_source_filename
        and not credentials_supplied
    ):
        raise ValueError(
            "A credentials source filename was supplied "
            "while credentials_supplied is False."
        )

    manifest = {
        "schema_version": (
            MANIFEST_SCHEMA_VERSION
        ),
        "campaign": {
            "abbreviation": (
                str(
                    campaign_abbreviation
                ).strip()
            ),
            "id": (
                str(
                    campaign_id
                ).strip()
            ),
        },
        "extracted_at": (
            _timestamp(
                extracted_at
            )
        ),
        "campaign_files": {
            "data_export": (
                _optional_text(
                    campaign_data_filename
                )
            ),
            "description": (
                _optional_text(
                    campaign_description_filename
                )
            ),
        },
        "participant_credentials": {
            "supplied": bool(
                credentials_supplied
            ),
            "source_filename": (
                _optional_text(
                    credentials_source_filename
                )
            ),
            "stored_in_dataset": (
                False
            ),
        },
    }

    _check_for_sensitive_fields(
        manifest
    )

    return manifest


def write_manifest(
    path: Path,
    manifest: dict[str, Any],
) -> None:
    """
    Write a manifest as readable UTF-8 JSON.
    """
    _check_for_sensitive_fields(
        manifest
    )

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
            manifest,
            handle,
            indent=2,
            ensure_ascii=False,
        )

        handle.write(
            "\n"
        )


def read_manifest(
    path: Path,
) -> dict[str, Any]:
    path = Path(
        path
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(
            handle
        )

    if not isinstance(
        manifest,
        dict,
    ):
        raise ValueError(
            "Manifest must contain a JSON object."
        )

    _check_for_sensitive_fields(
        manifest
    )

    return manifest