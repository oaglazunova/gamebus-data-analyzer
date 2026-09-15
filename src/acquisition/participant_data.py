from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from config.credentials import require_authcode
from src.acquisition.participant_credentials import (
    CredentialStore,
)
from src.extraction.data_collectors import (
    AllDataCollector,
)
from src.extraction.gamebus_client import (
    GameBusClient,
)


ProgressCallback = Callable[
    [
        int,
        int,
        str,
        str,
    ],
    None,
]


@dataclass(frozen=True)
class ParticipantExtractionResult:
    email: str
    expected_pid: str
    actual_pid: str | None
    status: str
    data_types: int = 0
    files_written: int = 0
    message: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == "complete"


def _report_progress(
    callback: ProgressCallback | None,
    *,
    current: int,
    total: int,
    email: str,
    stage: str,
) -> None:
    if callback is None:
        return

    callback(
        current,
        total,
        email,
        stage,
    )


def _normalize_pid(
    value,
) -> str:
    if value is None:
        return ""

    return str(value).strip()


def extract_participant_data(
    *,
    dataset_dir: Path,
    participants: list[dict],
    credentials: CredentialStore,
    progress_callback: ProgressCallback | None = None,
) -> list[ParticipantExtractionResult]:
    """
    Extract participant GameBus data.

    Only participants explicitly marked with
    selected_for_participant_extraction=True are processed.

    Passwords are retrieved from CredentialStore only when needed
    and are never written to the dataset.
    """
    dataset_dir = Path(
        dataset_dir
    )

    raw_data_dir = (
        dataset_dir
        / "data_raw"
    )

    raw_data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected = [
        participant
        for participant in participants
        if participant.get(
            "selected_for_participant_extraction",
            False,
        )
    ]

    total = len(
        selected
    )

    if total == 0:
        return []

    authcode = require_authcode()

    client = GameBusClient(
        authcode
    )

    results: list[
        ParticipantExtractionResult
    ] = []

    for current, participant in enumerate(
        selected,
        start=1,
    ):
        email = str(
            participant.get(
                "email",
                "",
            )
            or ""
        ).strip()

        expected_pid = _normalize_pid(
            participant.get(
                "pid"
            )
        )

        _report_progress(
            progress_callback,
            current=current,
            total=total,
            email=email,
            stage="authenticating",
        )

        password = (
            credentials.get_password(
                email
            )
        )

        if not password:
            results.append(
                ParticipantExtractionResult(
                    email=email,
                    expected_pid=(
                        expected_pid
                    ),
                    actual_pid=None,
                    status=(
                        "missing_credentials"
                    ),
                    message=(
                        "No password is available "
                        "for this account."
                    ),
                )
            )

            continue

        token = client.get_user_token(
            email,
            password,
        )

        if not token:
            results.append(
                ParticipantExtractionResult(
                    email=email,
                    expected_pid=(
                        expected_pid
                    ),
                    actual_pid=None,
                    status=(
                        "authentication_failed"
                    ),
                    message=(
                        "GameBus authentication "
                        "failed."
                    ),
                )
            )

            continue

        _report_progress(
            progress_callback,
            current=current,
            total=total,
            email=email,
            stage="checking_player",
        )

        user_id_result = (
            client.get_user_id(
                token
            )
        )

        if not user_id_result:
            actual_pid = None
            authenticated_email = None

        else:
            (
                actual_pid,
                authenticated_email,
            ) = user_id_result

        if actual_pid is None:
            results.append(
                ParticipantExtractionResult(
                    email=email,
                    expected_pid=(
                        expected_pid
                    ),
                    actual_pid=None,
                    status=(
                        "player_lookup_failed"
                    ),
                    message=(
                        "Could not determine the "
                        "authenticated GameBus "
                        "Player ID."
                    ),
                )
            )

            continue

        actual_pid_text = (
            _normalize_pid(
                actual_pid
            )
        )

        # This is an important safety check:
        # the supplied credential must resolve
        # to the same Player ID as the campaign
        # account selected by the researcher.
        if (
            expected_pid
            and actual_pid_text
            != expected_pid
        ):
            results.append(
                ParticipantExtractionResult(
                    email=email,
                    expected_pid=(
                        expected_pid
                    ),
                    actual_pid=(
                        actual_pid_text
                    ),
                    status="pid_mismatch",
                    message=(
                        "The supplied credentials "
                        "authenticated as a different "
                        "GameBus Player ID."
                    ),
                )
            )

            continue

        _report_progress(
            progress_callback,
            current=current,
            total=total,
            email=email,
            stage="downloading",
        )

        collector = AllDataCollector(
            client=client,
            token=token,
            user_id=actual_pid,
            user_email=(
                authenticated_email
            ),
            output_dir=(
                raw_data_dir
            ),
        )

        try:
            (
                data_dict,
                file_paths,
            ) = collector.collect()

        except Exception as exc:
            results.append(
                ParticipantExtractionResult(
                    email=email,
                    expected_pid=(
                        expected_pid
                    ),
                    actual_pid=(
                        actual_pid_text
                    ),
                    status=(
                        "extraction_failed"
                    ),
                    message=str(
                        exc
                    ),
                )
            )

            continue

        results.append(
            ParticipantExtractionResult(
                email=email,
                expected_pid=(
                    expected_pid
                ),
                actual_pid=(
                    actual_pid_text
                ),
                status="complete",
                data_types=len(
                    data_dict
                ),
                files_written=len(
                    file_paths
                ),
            )
        )

        _report_progress(
            progress_callback,
            current=current,
            total=total,
            email=email,
            stage="complete",
        )

    return results