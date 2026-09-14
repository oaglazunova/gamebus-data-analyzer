from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import pandas as pd
import streamlit as st

from src.acquisition.campaign_dataset import bootstrap_campaign_dataset
from src.acquisition.campaign_users import list_campaign_users
from src.acquisition.gamebus_campaigns import CampaignDownloadError
from src.acquisition.organizer_storage import (
    delete_organizer_password,
    get_cookie_file,
    load_organizer_password,
    load_settings,
    save_organizer_password,
    save_settings,
)
from src.acquisition.participant_credentials import (
    build_initial_cohort_candidates,
    load_participant_credentials_bytes,
)
from src.acquisition.participant_data import extract_participant_data
from src.datasets.layout import (
    DATASETS_DIR,
    get_cohort_manifest_path,
    get_extraction_manifest_path,
)
from src.datasets.manifests import (
    build_cohort_manifest,
    read_manifest,
    write_manifest,
)


DATASET_STATE_KEY = "get_data_dataset_dir"
USERS_STATE_KEY = "get_data_campaign_users"
CAMPAIGN_ID_STATE_KEY = "get_data_campaign_id"
COHORT_SAVED_STATE_KEY = "get_data_cohort_saved"
DATASETS_DIR_INPUT_KEY = "get_data_datasets_dir_input"
CREDENTIALS_STATE_KEY = "get_data_credentials"
CREDENTIALS_FILENAME_STATE_KEY = "get_data_credentials_filename"
COHORT_EDITOR_KEY = "get_data_cohort_editor"
STORAGE_ERROR_KEY = "get_data_storage_error"


def _choose_directory(
    initial_directory: str,
) -> str | None:
    root = tk.Tk()

    try:
        root.withdraw()

        root.attributes(
            "-topmost",
            True,
        )

        initial_path = Path(
            initial_directory
        ).expanduser()

        if not initial_path.exists():
            initial_path = Path.home()

        selected = filedialog.askdirectory(
            parent=root,
            initialdir=str(
                initial_path
            ),
            title=(
                "Choose where GameBus datasets "
                "should be saved"
            ),
        )

        return (
            selected
            or None
        )

    finally:
        root.destroy()


def _browse_datasets_directory() -> None:
    current = st.session_state.get(
        DATASETS_DIR_INPUT_KEY,
        str(DATASETS_DIR),
    )

    try:
        selected = _choose_directory(
            current
        )

    except Exception as exc:
        st.session_state[
            STORAGE_ERROR_KEY
        ] = (
            "Could not open the folder "
            f"selector: {exc}"
        )

        return

    if selected:
        st.session_state[
            DATASETS_DIR_INPUT_KEY
        ] = selected

        st.session_state.pop(
            STORAGE_ERROR_KEY,
            None,
        )


def _load_saved_password(
    email: str,
    remember_credentials: bool,
) -> str:
    if (
        not remember_credentials
        or not email
    ):
        return ""

    try:
        return (
            load_organizer_password(
                email
            )
            or ""
        )

    except Exception:
        return ""


def _remember_settings(
    *,
    email: str,
    password: str,
    remember_credentials: bool,
    campaign_abbreviation: str,
    datasets_dir: Path,
) -> None:
    current = load_settings()

    campaigns = list(
        current.get(
            "campaign_abbreviations",
            [],
        )
    )

    if (
        campaign_abbreviation
        not in campaigns
    ):
        campaigns.append(
            campaign_abbreviation
        )

    save_settings(
        {
            "organizer_email": (
                email
            ),
            "remember_credentials": (
                remember_credentials
            ),
            "last_campaign_abbreviation": (
                campaign_abbreviation
            ),
            "campaign_abbreviations": (
                campaigns
            ),
            "datasets_dir": str(
                datasets_dir
            ),
        }
    )

    if remember_credentials:
        if password:
            save_organizer_password(
                email,
                password,
            )

    else:
        delete_organizer_password(
            email
        )


def _clear_campaign_state() -> None:
    for key in (
        DATASET_STATE_KEY,
        USERS_STATE_KEY,
        CAMPAIGN_ID_STATE_KEY,
        COHORT_SAVED_STATE_KEY,
        CREDENTIALS_STATE_KEY,
        CREDENTIALS_FILENAME_STATE_KEY,
        COHORT_EDITOR_KEY,
    ):
        st.session_state.pop(
            key,
            None,
        )


def _campaign_selector(
    settings: dict,
) -> str:
    campaigns = list(
        settings.get(
            "campaign_abbreviations",
            [],
        )
    )

    last_campaign = (
        settings.get(
            "last_campaign_abbreviation",
            "",
        )
        .strip()
    )

    if (
        last_campaign
        and last_campaign
        not in campaigns
    ):
        campaigns.append(
            last_campaign
        )

    if (
        last_campaign
        and last_campaign
        in campaigns
    ):
        default_index = (
            campaigns.index(
                last_campaign
            )
        )

    else:
        default_index = None

    selected = st.selectbox(
        "Campaign abbreviation",
        options=campaigns,
        index=default_index,
        placeholder=(
            "Select a saved campaign or "
            "enter a new abbreviation"
        ),
        accept_new_options=True,
    )

    return (
        selected.strip()
        if selected
        else ""
    )


def _show_dataset_location(
    dataset_dir: Path,
    extraction_manifest: dict,
) -> None:
    with st.expander(
        "Where are the downloaded files?",
        expanded=False,
    ):
        st.write(
            "This extraction is stored in:"
        )

        st.code(
            str(
                dataset_dir.resolve()
            )
        )

        campaign_files = (
            extraction_manifest[
                "campaign_files"
            ]
        )

        st.write(
            "Downloaded from GameBus Studio:"
        )

        st.markdown(
            f"- `{campaign_files['description']}`"
        )

        st.markdown(
            f"- `{campaign_files['data_export']}`"
        )

        st.caption(
            "You can analyze this dataset later "
            "by selecting this dataset folder "
            "under Analyze existing data."
        )


def _apply_extraction_results(
    cohort_manifest: dict,
    results,
) -> dict:
    successful_pids = {
        str(
            result.expected_pid
        )
        for result in results
        if result.succeeded
    }

    for participant in (
        cohort_manifest.get(
            "participants",
            [],
        )
    ):
        pid = str(
            participant.get(
                "pid",
                "",
            )
        )

        if pid in successful_pids:
            participant[
                "participant_data_extracted"
            ] = True

    return cohort_manifest


def _load_saved_cohort_state(
    cohort_manifest_path: Path,
) -> tuple[
    dict[str, bool],
    dict[str, bool],
    dict[str, bool],
]:
    selected_by_pid: dict[
        str,
        bool,
    ] = {}

    selected_by_email: dict[
        str,
        bool,
    ] = {}

    extracted_by_pid: dict[
        str,
        bool,
    ] = {}

    if not cohort_manifest_path.exists():
        return (
            selected_by_pid,
            selected_by_email,
            extracted_by_pid,
        )

    existing_manifest = read_manifest(
        cohort_manifest_path
    )

    for participant in (
        existing_manifest.get(
            "participants",
            [],
        )
    ):
        selected = bool(
            participant.get(
                "selected_for_analysis",
                False,
            )
        )

        pid = participant.get(
            "pid"
        )

        email = participant.get(
            "email"
        )

        if pid:
            pid_text = str(
                pid
            )

            selected_by_pid[
                pid_text
            ] = selected

            extracted_by_pid[
                pid_text
            ] = bool(
                participant.get(
                    "participant_data_extracted",
                    False,
                )
            )

        if email:
            selected_by_email[
                str(
                    email
                ).casefold()
            ] = selected

    return (
        selected_by_pid,
        selected_by_email,
        extracted_by_pid,
    )


def _render_participant_review(
    *,
    dataset_dir: Path,
    users: list[dict],
    extraction_manifest: dict,
    credentials,
    candidates,
) -> None:
    cohort_manifest_path = (
        get_cohort_manifest_path(
            dataset_dir
        )
    )

    if credentials is None:
        review_label = (
            "Review participants (optional)"
        )

    else:
        review_label = (
            "Review participants before "
            "participant-level extraction"
        )

    with st.expander(
        review_label,
        expanded=(
            credentials is not None
        ),
    ):
        st.write(
            "The email addresses shown here are "
            "the account identifiers used in "
            "GameBus Studio for this campaign."
        )

        st.write(
            "You may exclude accounts that should "
            "not be part of the analysis, for "
            "example test or administrator accounts. "
            "Excluding an account does not delete "
            "any downloaded data; it only determines "
            "which participants are included in "
            "later analyses."
        )

        st.caption(
            "You can change this selection later "
            "under Analyze existing data."
        )

        (
            saved_selection_by_pid,
            saved_selection_by_email,
            saved_extracted_by_pid,
        ) = _load_saved_cohort_state(
            cohort_manifest_path
        )

        credentials_by_pid: dict[
            str,
            bool,
        ] = {}

        if candidates is not None:
            credentials_by_pid = {
                str(
                    candidate.get(
                        "pid"
                    )
                ): bool(
                    candidate[
                        "credentials_available"
                    ]
                )
                for candidate in candidates
                if candidate.get(
                    "pid"
                )
            }

        rows = []

        for user in users:
            pid = (
                user.get(
                    "pid"
                )
                or ""
            )

            email = (
                user.get(
                    "email"
                )
                or ""
            )

            pid_text = str(
                pid
            )

            credentials_available = (
                credentials_by_pid.get(
                    pid_text,
                    False,
                )
            )

            if credentials is None:
                selected = True

            else:
                selected = (
                    credentials_available
                )

            if (
                pid_text
                in saved_selection_by_pid
            ):
                selected = (
                    saved_selection_by_pid[
                        pid_text
                    ]
                )

            elif (
                email
                and email.casefold()
                in saved_selection_by_email
            ):
                selected = (
                    saved_selection_by_email[
                        email.casefold()
                    ]
                )

            rows.append(
                {
                    "Include": selected,
                    "Email": email,
                    "PID": pid,
                    "Credentials": (
                        credentials_available
                    ),
                }
            )

        edited_table = (
            st.data_editor(
                pd.DataFrame(
                    rows
                ),
                key=COHORT_EDITOR_KEY,
                hide_index=True,
                use_container_width=True,
                num_rows="fixed",
                disabled=[
                    "Email",
                    "PID",
                    "Credentials",
                ],
                column_config={
                    "Include": (
                        st.column_config.CheckboxColumn(
                            "Include",
                            help=(
                                "Include this account "
                                "in later analyses."
                            ),
                        )
                    ),
                    "Email": (
                        st.column_config.TextColumn(
                            "Email"
                        )
                    ),
                    "PID": (
                        st.column_config.TextColumn(
                            "PID"
                        )
                    ),
                    "Credentials": (
                        st.column_config.CheckboxColumn(
                            "Credentials",
                            help=(
                                "Participant credentials "
                                "are available for "
                                "additional data extraction."
                            ),
                        )
                    ),
                },
            )
        )

        selected_count = int(
            edited_table[
                "Include"
            ].sum()
        )

        st.caption(
            f"{selected_count} of "
            f"{len(users)} accounts included."
        )

        if st.button(
            "Save participant selection",
            type="primary",
            use_container_width=True,
        ):
            participants = []

            for index, user in enumerate(
                users
            ):
                selected = bool(
                    edited_table.iloc[
                        index
                    ][
                        "Include"
                    ]
                )

                credentials_available = bool(
                    edited_table.iloc[
                        index
                    ][
                        "Credentials"
                    ]
                )

                pid_text = str(
                    user.get(
                        "pid",
                        "",
                    )
                )

                was_extracted = (
                    saved_extracted_by_pid.get(
                        pid_text,
                        False,
                    )
                )

                if was_extracted:
                    selected_for_extraction = (
                        True
                    )

                else:
                    selected_for_extraction = (
                        selected
                        and credentials_available
                    )

                participants.append(
                    {
                        "pid": user.get(
                            "pid"
                        ),
                        "email": user.get(
                            "email"
                        ),
                        "selected_for_analysis": (
                            selected
                        ),
                        "credentials_available": (
                            credentials_available
                        ),
                        "selected_for_participant_extraction": (
                            selected_for_extraction
                        ),
                        "participant_data_extracted": (
                            was_extracted
                        ),
                    }
                )

            campaign = (
                extraction_manifest[
                    "campaign"
                ]
            )

            cohort_manifest = (
                build_cohort_manifest(
                    campaign_abbreviation=(
                        campaign[
                            "abbreviation"
                        ]
                    ),
                    campaign_id=(
                        campaign[
                            "id"
                        ]
                    ),
                    participants=(
                        participants
                    ),
                    candidate_source=(
                        "gamebus_studio_users"
                    ),
                )
            )

            write_manifest(
                cohort_manifest_path,
                cohort_manifest,
            )

            st.session_state[
                COHORT_SAVED_STATE_KEY
            ] = True

            st.success(
                "Participant selection saved."
            )

        if (
            cohort_manifest_path.exists()
            or st.session_state.get(
                COHORT_SAVED_STATE_KEY,
                False,
            )
        ):
            st.caption(
                "A participant selection is "
                "saved for this dataset."
            )

    if not cohort_manifest_path.exists():
        if credentials is None:
            st.caption(
                "Participant review can be "
                "skipped now. You can review "
                "participants later under "
                "Analyze existing data."
            )

        else:
            st.caption(
                "Review and save the participant "
                "selection before starting "
                "participant-level data extraction."
            )


def _render_participant_extraction(
    *,
    dataset_dir: Path,
    credentials,
) -> None:
    if credentials is None:
        return

    cohort_manifest_path = (
        get_cohort_manifest_path(
            dataset_dir
        )
    )

    if not cohort_manifest_path.exists():
        return

    st.divider()

    st.subheader(
        "Participant-level data"
    )

    cohort_manifest = read_manifest(
        cohort_manifest_path
    )

    extraction_participants = [
        participant
        for participant in (
            cohort_manifest.get(
                "participants",
                [],
            )
        )
        if participant.get(
            "selected_for_participant_extraction",
            False,
        )
    ]

    already_extracted = [
        participant
        for participant in (
            extraction_participants
        )
        if participant.get(
            "participant_data_extracted",
            False,
        )
    ]

    remaining = [
        participant
        for participant in (
            extraction_participants
        )
        if not participant.get(
            "participant_data_extracted",
            False,
        )
    ]

    if not extraction_participants:
        st.warning(
            "No selected participants have "
            "matching credentials, so there is "
            "no participant-level data to extract."
        )

        return

    st.write(
        f"{len(extraction_participants)} "
        "selected participant(s) have "
        "credentials available."
    )

    if already_extracted:
        st.caption(
            "Participant-level data have already "
            "been extracted for "
            f"{len(already_extracted)} "
            "participant(s)."
        )

    if not remaining:
        st.success(
            "Participant-level data have already "
            "been extracted for all selected "
            "participants."
        )

        return

    st.caption(
        f"{len(remaining)} participant(s) "
        "remain to be extracted."
    )

    if not st.button(
        "Extract participant-level data",
        type="primary",
        use_container_width=True,
    ):
        return

    progress = st.progress(
        0,
        text=(
            "Preparing participant "
            "extraction..."
        ),
    )

    with st.status(
        "Extracting participant-level data",
        expanded=True,
    ) as status:

        def report_participant_progress(
            current: int,
            total: int,
            email: str,
            stage: str,
        ) -> None:
            stage_labels = {
                "authenticating": (
                    "Authenticating"
                ),
                "checking_player": (
                    "Checking GameBus player"
                ),
                "downloading": (
                    "Downloading data"
                ),
                "complete": (
                    "Complete"
                ),
            }

            stage_fraction = {
                "authenticating": 0.10,
                "checking_player": 0.25,
                "downloading": 0.45,
                "complete": 1.0,
            }.get(
                stage,
                0.0,
            )

            overall_fraction = (
                (
                    current
                    - 1
                    + stage_fraction
                )
                / total
            )

            percent = min(
                100,
                max(
                    0,
                    int(
                        overall_fraction
                        * 100
                    ),
                ),
            )

            label = (
                stage_labels.get(
                    stage,
                    stage,
                )
            )

            text = (
                f"Participant {current} "
                f"of {total}: "
                f"{label} — {email}"
            )

            progress.progress(
                percent,
                text=text,
            )

            if stage in {
                "authenticating",
                "downloading",
                "complete",
            }:
                status.write(
                    text
                )

        try:
            results = (
                extract_participant_data(
                    dataset_dir=(
                        dataset_dir
                    ),
                    participants=(
                        remaining
                    ),
                    credentials=(
                        credentials
                    ),
                    progress_callback=(
                        report_participant_progress
                    ),
                )
            )

            cohort_manifest = (
                _apply_extraction_results(
                    cohort_manifest,
                    results,
                )
            )

            write_manifest(
                cohort_manifest_path,
                cohort_manifest,
            )

            successful = [
                result
                for result in results
                if result.succeeded
            ]

            failed = [
                result
                for result in results
                if not result.succeeded
            ]

            progress.progress(
                100,
                text=(
                    "Participant-level "
                    "extraction complete."
                ),
            )

            if failed:
                status.update(
                    label=(
                        "Participant-level "
                        "extraction finished "
                        "with warnings"
                    ),
                    state="complete",
                    expanded=True,
                )

            else:
                status.update(
                    label=(
                        "Participant-level "
                        "data extracted"
                    ),
                    state="complete",
                    expanded=False,
                )

            st.success(
                "Participant-level data were "
                "successfully extracted for "
                f"{len(successful)} of "
                f"{len(results)} participant(s)."
            )

            if failed:
                st.warning(
                    "Extraction failed or was "
                    "skipped for "
                    f"{len(failed)} participant(s)."
                )

                failure_rows = [
                    {
                        "Email": (
                            result.email
                        ),
                        "PID": (
                            result.expected_pid
                        ),
                        "Status": (
                            result.status
                            .replace(
                                "_",
                                " ",
                            )
                            .title()
                        ),
                        "Reason": (
                            result.message
                            or ""
                        ),
                    }
                    for result in failed
                ]

                st.dataframe(
                    pd.DataFrame(
                        failure_rows
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

        except Exception as exc:
            status.update(
                label=(
                    "Participant-level "
                    "extraction failed"
                ),
                state="error",
                expanded=True,
            )

            st.error(
                "Participant-level "
                f"extraction failed: {exc}"
            )


def _show_campaign_summary() -> None:
    dataset_dir = Path(
        st.session_state[
            DATASET_STATE_KEY
        ]
    )

    users = st.session_state[
        USERS_STATE_KEY
    ]

    campaign_id = (
        st.session_state[
            CAMPAIGN_ID_STATE_KEY
        ]
    )

    credentials = (
        st.session_state.get(
            CREDENTIALS_STATE_KEY
        )
    )

    extraction_manifest = read_manifest(
        get_extraction_manifest_path(
            dataset_dir
        )
    )

    with st.sidebar:
        st.title(
            "GameBus Data Analyzer"
        )

        st.subheader(
            "Current dataset"
        )

        st.write(
            extraction_manifest[
                "campaign"
            ][
                "abbreviation"
            ]
        )

        st.caption(
            f"Campaign {campaign_id}"
        )

        if st.button(
            "Start another extraction",
            use_container_width=True,
        ):
            _clear_campaign_state()
            st.rerun()

    st.header(
        "Get data"
    )

    st.success(
        "Campaign data downloaded successfully."
    )

    credentials_metric = (
        "Not supplied"
        if credentials is None
        else (
            f"{credentials.count} supplied"
        )
    )

    col1, col2, col3 = st.columns(
        3
    )

    with col1:
        st.metric(
            "Campaign ID",
            campaign_id,
        )

    with col2:
        st.metric(
            "Campaign accounts",
            len(users),
        )

    with col3:
        st.metric(
            "Participant credentials",
            credentials_metric,
        )

    _show_dataset_location(
        dataset_dir,
        extraction_manifest,
    )

    if credentials is None:
        candidates = None

        st.info(
            "The campaign description and "
            "campaign data export have been "
            "downloaded from GameBus Studio. "
            "Participant credentials were not "
            "provided, so participant-level "
            "data were not extracted from the "
            "GameBus database."
        )

    else:
        candidates, summary = (
            build_initial_cohort_candidates(
                campaign_users=(
                    users
                ),
                credentials=(
                    credentials
                ),
            )
        )

        st.info(
            "The campaign description and "
            "campaign data export have been "
            "downloaded from GameBus Studio. "
            "Participant credentials were "
            "provided. Review the participants "
            "below before starting "
            "participant-level data extraction."
        )

        col1, col2, col3 = (
            st.columns(
                3
            )
        )

        with col1:
            st.metric(
                "Credentials",
                summary.credentials_accounts,
            )

        with col2:
            st.metric(
                "Matched to campaign",
                summary.matched_accounts,
            )

        with col3:
            st.metric(
                "Not in campaign",
                (
                    summary
                    .credentials_not_in_campaign
                ),
            )

        if (
            summary.matched_accounts
            == 0
        ):
            st.warning(
                "None of the accounts in the "
                "credentials file match this "
                "campaign. Matching is based "
                "on participant email, not on "
                "the filename."
            )

        elif (
            summary.credentials_not_in_campaign
            > 0
        ):
            st.warning(
                f"{summary.credentials_not_in_campaign} "
                "credential account(s) are not "
                "members of this campaign and "
                "will not be used for "
                "participant-level extraction."
            )

        else:
            st.success(
                "All credential accounts match "
                "campaign accounts."
            )

    st.divider()

    _render_participant_review(
        dataset_dir=(
            dataset_dir
        ),
        users=(
            users
        ),
        extraction_manifest=(
            extraction_manifest
        ),
        credentials=(
            credentials
        ),
        candidates=(
            candidates
        ),
    )

    _render_participant_extraction(
        dataset_dir=(
            dataset_dir
        ),
        credentials=(
            credentials
        ),
    )


def _render_sidebar_credentials_upload() -> None:
    st.subheader(
        "Participant credentials"
    )

    credentials_file = (
        st.file_uploader(
            "Participant credentials XLSX",
            type=[
                "xlsx",
            ],
            label_visibility=(
                "collapsed"
            ),
        )
    )

    st.caption(
        "Optional. Without participant "
        "credentials, only the campaign "
        "description and campaign analytics "
        "will be downloaded from GameBus "
        "Studio. Participant data from the "
        "GameBus database will not be extracted."
    )

    if credentials_file is None:
        st.session_state.pop(
            CREDENTIALS_STATE_KEY,
            None,
        )

        st.session_state.pop(
            CREDENTIALS_FILENAME_STATE_KEY,
            None,
        )

        return

    try:
        credentials = (
            load_participant_credentials_bytes(
                credentials_file.getvalue(),
                source_filename=(
                    credentials_file.name
                ),
            )
        )

        st.session_state[
            CREDENTIALS_STATE_KEY
        ] = credentials

        st.session_state[
            CREDENTIALS_FILENAME_STATE_KEY
        ] = (
            credentials_file.name
        )

        st.success(
            f"{credentials.count} participant "
            "credential account(s) loaded."
        )

    except Exception as exc:
        st.session_state.pop(
            CREDENTIALS_STATE_KEY,
            None,
        )

        st.session_state.pop(
            CREDENTIALS_FILENAME_STATE_KEY,
            None,
        )

        st.error(
            "Could not read credentials: "
            f"{exc}"
        )


def render_get_data_page() -> None:
    if (
        DATASET_STATE_KEY
        in st.session_state
    ):
        _show_campaign_summary()
        return

    settings = load_settings()

    saved_email = settings.get(
        "organizer_email",
        "",
    )

    remember_default = bool(
        settings.get(
            "remember_credentials",
            True,
        )
    )

    saved_password = (
        _load_saved_password(
            saved_email,
            remember_default,
        )
    )

    default_datasets_dir = (
        settings.get(
            "datasets_dir",
            "",
        )
        or str(
            DATASETS_DIR
        )
    )

    if (
        DATASETS_DIR_INPUT_KEY
        not in st.session_state
    ):
        st.session_state[
            DATASETS_DIR_INPUT_KEY
        ] = (
            default_datasets_dir
        )

    credentials_are_saved = bool(
        remember_default
        and saved_email
        and saved_password
    )

    with st.sidebar:
        st.title(
            "GameBus Data Analyzer"
        )

        with st.expander(
            "GameBus organizer access",
            expanded=(
                not credentials_are_saved
            ),
        ):
            organizer_email = (
                st.text_input(
                    "Organizer email",
                    value=(
                        saved_email
                    ),
                )
                .strip()
            )

            organizer_password = (
                st.text_input(
                    "Organizer password",
                    value=(
                        saved_password
                    ),
                    type="password",
                    help=(
                        "Needed only if the saved "
                        "GameBus session has expired."
                    ),
                )
            )

            remember_credentials = (
                st.checkbox(
                    "Remember organizer credentials",
                    value=(
                        remember_default
                    ),
                    help=(
                        "The password is stored "
                        "using the operating-system "
                        "keyring."
                    ),
                )
            )

        campaign_abbreviation = (
            _campaign_selector(
                settings
            )
        )

        st.subheader(
            "Dataset destination"
        )

        datasets_dir_text = (
            st.text_input(
                "Save datasets in",
                key=(
                    DATASETS_DIR_INPUT_KEY
                ),
            )
            .strip()
        )

        st.button(
            "Browse...",
            use_container_width=True,
            on_click=(
                _browse_datasets_directory
            ),
        )

        storage_error = (
            st.session_state.get(
                STORAGE_ERROR_KEY
            )
        )

        if storage_error:
            st.error(
                storage_error
            )

        st.caption(
            "Each extraction is saved in its "
            "own campaign + ID + timestamp "
            "subfolder."
        )

        _render_sidebar_credentials_upload()

        download_clicked = (
            st.button(
                "Download campaign data",
                type="primary",
                use_container_width=True,
                disabled=(
                    not campaign_abbreviation
                ),
            )
        )

    st.header(
        "Get data"
    )

    st.write(
        "Choose a campaign in the sidebar, "
        "then download its GameBus data."
    )

    st.info(
        "Participant credentials are optional. "
        "Without them, the app downloads only "
        "the campaign description and campaign "
        "analytics available through GameBus "
        "Studio. With participant credentials, "
        "you can additionally extract "
        "participant-level data from the "
        "GameBus database after reviewing the "
        "participant list."
    )

    if not download_clicked:
        return

    try:
        datasets_dir = Path(
            datasets_dir_text
        ).expanduser()

        progress = st.progress(
            0,
            text=(
                "Preparing download..."
            ),
        )

        with st.status(
            "Downloading campaign data",
            expanded=True,
        ) as status:

            def report_progress(
                stage: str,
            ) -> None:
                if (
                    stage
                    == "campaign_description"
                ):
                    status.write(
                        "Downloading campaign "
                        "description..."
                    )

                    progress.progress(
                        10,
                        text=(
                            "Downloading campaign "
                            "description..."
                        ),
                    )

                elif (
                    stage
                    == "campaign_analytics"
                ):
                    status.write(
                        "Campaign description ✓"
                    )

                    status.write(
                        "Downloading campaign "
                        "analytics..."
                    )

                    progress.progress(
                        35,
                        text=(
                            "Downloading campaign "
                            "analytics..."
                        ),
                    )

                elif (
                    stage
                    == "saving_dataset"
                ):
                    status.write(
                        "Campaign analytics ✓"
                    )

                    status.write(
                        "Saving campaign files..."
                    )

                    progress.progress(
                        70,
                        text=(
                            "Saving campaign files..."
                        ),
                    )

            dataset_dir = (
                bootstrap_campaign_dataset(
                    campaign_abbreviation=(
                        campaign_abbreviation
                    ),
                    organizer_email=(
                        organizer_email
                        or None
                    ),
                    organizer_password=(
                        organizer_password
                        or None
                    ),
                    cookie_file=(
                        get_cookie_file()
                    ),
                    datasets_dir=(
                        datasets_dir
                    ),
                    progress_callback=(
                        report_progress
                    ),
                )
            )

            extraction_manifest_path = (
                get_extraction_manifest_path(
                    dataset_dir
                )
            )

            extraction_manifest = (
                read_manifest(
                    extraction_manifest_path
                )
            )

            campaign_id = (
                extraction_manifest[
                    "campaign"
                ][
                    "id"
                ]
            )

            credentials = (
                st.session_state.get(
                    CREDENTIALS_STATE_KEY
                )
            )

            if credentials is not None:
                extraction_manifest[
                    "participant_credentials"
                ] = {
                    "supplied": True,
                    "source_filename": (
                        st.session_state.get(
                            CREDENTIALS_FILENAME_STATE_KEY
                        )
                    ),
                    "stored_in_dataset": False,
                }

                write_manifest(
                    extraction_manifest_path,
                    extraction_manifest,
                )

            status.write(
                "Campaign files saved ✓"
            )

            status.write(
                "Retrieving campaign accounts..."
            )

            progress.progress(
                85,
                text=(
                    "Retrieving campaign "
                    "accounts..."
                ),
            )

            users = list_campaign_users(
                campaign_id=(
                    campaign_id
                ),
                organizer_email=(
                    organizer_email
                    or None
                ),
                organizer_password=(
                    organizer_password
                    or None
                ),
                cookie_file=(
                    get_cookie_file()
                ),
            )

            status.write(
                "Campaign accounts ✓"
            )

            progress.progress(
                100,
                text=(
                    "Download complete."
                ),
            )

            status.update(
                label=(
                    "Campaign data downloaded"
                ),
                state="complete",
                expanded=False,
            )

        _remember_settings(
            email=(
                organizer_email
            ),
            password=(
                organizer_password
            ),
            remember_credentials=(
                remember_credentials
            ),
            campaign_abbreviation=(
                campaign_abbreviation
            ),
            datasets_dir=(
                datasets_dir
            ),
        )

        st.session_state[
            DATASET_STATE_KEY
        ] = str(
            dataset_dir
        )

        st.session_state[
            USERS_STATE_KEY
        ] = users

        st.session_state[
            CAMPAIGN_ID_STATE_KEY
        ] = campaign_id

        st.session_state[
            COHORT_SAVED_STATE_KEY
        ] = False

        st.rerun()

    except (
        CampaignDownloadError,
        OSError,
        ValueError,
    ) as exc:
        st.error(
            str(
                exc
            )
        )

    except Exception as exc:
        st.error(
            "Campaign retrieval failed: "
            f"{exc}"
        )