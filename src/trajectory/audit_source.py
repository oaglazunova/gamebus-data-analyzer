from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
from typing import Any

from src.acquisition.campaign_users import (
    list_campaign_users,
)
from src.acquisition.gamebus_campaigns import (
    DEFAULT_CAMPAIGNS_BASE_URL,
    CampaignDownloadError,
    download_campaign_data_export,
    download_campaign_description,
    extract_campaign_id_from_filename,
)
from src.trajectory.audit_run import (
    TrajectoryAuditRunError,
    read_campaign_export_participant_ids,
    read_campaign_identity,
)


class TrajectorySourceError(
    RuntimeError
):
    pass


@dataclass(
    frozen=True
)
class TrajectorySourceParticipant:
    pid: int
    email: str | None = None
    account_id: str | None = None


@dataclass(
    frozen=True
)
class TrajectoryAuditSource:
    source_type: str

    staging_dir: Path

    campaign_data_path: Path
    campaign_description_path: Path

    campaign_abbreviation: str | None
    campaign_id: str | None

    participants: tuple[
        TrajectorySourceParticipant,
        ...
    ]

    accounts_without_pid: int = 0


def _normalize_pid(
    value: Any,
) -> int | None:
    if value is None:
        return None

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
        text = text[:-2]

    if not text.isdigit():
        return None

    return int(
        text
    )


def _create_staging_directory() -> Path:
    return Path(
        tempfile.mkdtemp(
            prefix=(
                "gamebus_trajectory_source_"
            )
        )
    ).resolve()


def remove_trajectory_source(
    source: TrajectoryAuditSource | None,
) -> None:
    """
    Remove temporary source files after a run or when
    the user replaces the selected campaign source.
    """
    if source is None:
        return

    try:
        shutil.rmtree(
            source.staging_dir,
            ignore_errors=True,
        )
    except Exception:
        pass


def prepare_uploaded_trajectory_source(
    *,
    campaign_description_filename: str,
    campaign_description_bytes: bytes,
    campaign_data_filename: str,
    campaign_data_bytes: bytes,
) -> TrajectoryAuditSource:
    """
    Prepare manually uploaded historical campaign files.

    Files are staged temporarily. The audit runner later
    copies the exact source files into the permanent,
    timestamped audit folder.
    """
    if not campaign_description_bytes:
        raise TrajectorySourceError(
            "Campaign description XLSX is empty."
        )

    if not campaign_data_bytes:
        raise TrajectorySourceError(
            "Campaign analytics ZIP is empty."
        )

    description_name = Path(
        campaign_description_filename
    ).name

    data_name = Path(
        campaign_data_filename
    ).name

    if not description_name.lower().endswith(
        ".xlsx"
    ):
        raise TrajectorySourceError(
            "Campaign description must be an XLSX file."
        )

    if not data_name.lower().endswith(
        ".zip"
    ):
        raise TrajectorySourceError(
            "Campaign analytics must be a ZIP file."
        )

    staging_dir = (
        _create_staging_directory()
    )

    description_path = (
        staging_dir
        / description_name
    )

    data_path = (
        staging_dir
        / data_name
    )

    try:
        description_path.write_bytes(
            campaign_description_bytes
        )

        data_path.write_bytes(
            campaign_data_bytes
        )

        (
            abbreviation,
            campaign_id,
        ) = read_campaign_identity(
            description_path
        )

        participant_ids = (
            read_campaign_export_participant_ids(
                data_path
            )
        )

        participants = tuple(
            TrajectorySourceParticipant(
                pid=pid
            )
            for pid in participant_ids
        )

        if not participants:
            raise TrajectorySourceError(
                "No participants were found in "
                "1-aggregated-data.csv."
            )

        return TrajectoryAuditSource(
            source_type=(
                "uploaded_files"
            ),
            staging_dir=(
                staging_dir
            ),
            campaign_data_path=(
                data_path
            ),
            campaign_description_path=(
                description_path
            ),
            campaign_abbreviation=(
                abbreviation
            ),
            campaign_id=(
                campaign_id
            ),
            participants=(
                participants
            ),
        )

    except Exception as exc:
        shutil.rmtree(
            staging_dir,
            ignore_errors=True,
        )

        if isinstance(
            exc,
            TrajectorySourceError,
        ):
            raise

        raise TrajectorySourceError(
            "Could not prepare uploaded campaign "
            f"files: {exc}"
        ) from exc


def prepare_gamebus_trajectory_source(
    *,
    campaign_abbreviation: str,
    organizer_email: str | None = None,
    organizer_password: str | None = None,
    cookie_file: Path | None = None,
    base_url: str = (
        DEFAULT_CAMPAIGNS_BASE_URL
    ),
) -> TrajectoryAuditSource:
    """
    Retrieve the current campaign snapshot directly
    from GameBus.

    No HW8 dataset is created.

    The temporary snapshot contains only the campaign
    description and campaign analytics export. The
    normalized campaign-user list is kept in memory.
    """
    abbreviation = (
        campaign_abbreviation
        .strip()
    )

    if not abbreviation:
        raise TrajectorySourceError(
            "Campaign abbreviation is required."
        )

    staging_dir = (
        _create_staging_directory()
    )

    try:
        description_path = (
            download_campaign_description(
                campaign_abbreviation=(
                    abbreviation
                ),
                destination_dir=(
                    staging_dir
                ),
                organizer_email=(
                    organizer_email
                ),
                organizer_password=(
                    organizer_password
                ),
                cookie_file=(
                    cookie_file
                ),
                base_url=(
                    base_url
                ),
            )
        )

        data_path = (
            download_campaign_data_export(
                campaign_abbreviation=(
                    abbreviation
                ),
                destination_dir=(
                    staging_dir
                ),
                organizer_email=(
                    organizer_email
                ),
                organizer_password=(
                    organizer_password
                ),
                cookie_file=(
                    cookie_file
                ),
                base_url=(
                    base_url
                ),
            )
        )

        description_campaign_id = (
            extract_campaign_id_from_filename(
                description_path.name
            )
        )

        data_campaign_id = (
            extract_campaign_id_from_filename(
                data_path.name
            )
        )

        if (
            description_campaign_id
            != data_campaign_id
        ):
            raise TrajectorySourceError(
                "Campaign ID mismatch between "
                "the GameBus description and "
                "analytics export: "
                f"{description_campaign_id} != "
                f"{data_campaign_id}"
            )

        campaign_id = (
            description_campaign_id
        )

        accounts = list_campaign_users(
            campaign_id=(
                campaign_id
            ),
            organizer_email=(
                organizer_email
            ),
            organizer_password=(
                organizer_password
            ),
            cookie_file=(
                cookie_file
            ),
            base_url=(
                base_url
            ),
        )

        participants = []
        accounts_without_pid = 0

        seen_pids: set[int] = set()

        for account in accounts:
            pid = _normalize_pid(
                account.get(
                    "pid"
                )
            )

            if pid is None:
                accounts_without_pid += 1
                continue

            if pid in seen_pids:
                continue

            seen_pids.add(
                pid
            )

            participants.append(
                TrajectorySourceParticipant(
                    pid=pid,
                    email=(
                        account.get(
                            "email"
                        )
                    ),
                    account_id=(
                        account.get(
                            "account_id"
                        )
                    ),
                )
            )

        if not participants:
            raise TrajectorySourceError(
                "GameBus Studio returned no campaign "
                "accounts with a valid player ID."
            )

        return TrajectoryAuditSource(
            source_type="gamebus",
            staging_dir=(
                staging_dir
            ),
            campaign_data_path=(
                Path(
                    data_path
                )
            ),
            campaign_description_path=(
                Path(
                    description_path
                )
            ),
            campaign_abbreviation=(
                abbreviation
            ),
            campaign_id=(
                campaign_id
            ),
            participants=tuple(
                participants
            ),
            accounts_without_pid=(
                accounts_without_pid
            ),
        )

    except (
        CampaignDownloadError,
        TrajectorySourceError,
    ) as exc:
        shutil.rmtree(
            staging_dir,
            ignore_errors=True,
        )

        if isinstance(
            exc,
            TrajectorySourceError,
        ):
            raise

        raise TrajectorySourceError(
            str(
                exc
            )
        ) from exc

    except Exception as exc:
        shutil.rmtree(
            staging_dir,
            ignore_errors=True,
        )

        raise TrajectorySourceError(
            "Could not retrieve the campaign "
            f"from GameBus: {exc}"
        ) from exc