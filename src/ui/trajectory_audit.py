from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import pandas as pd
import streamlit as st

from src.acquisition.organizer_storage import (
    delete_organizer_password,
    get_cookie_file,
    load_organizer_password,
    load_settings,
    save_organizer_password,
    save_settings,
)
from src.trajectory.audit_run import (
    DEFAULT_TRAJECTORY_AUDITS_DIR,
    TrajectoryAuditRunError,
    run_trajectory_audit_snapshot,
)
from src.trajectory.audit_source import (
    TrajectoryAuditSource,
    TrajectorySourceError,
    prepare_gamebus_trajectory_source,
    prepare_uploaded_trajectory_source,
    remove_trajectory_source,
)


SOURCE_MODE_KEY = (
    "trajectory_source_mode"
)
SOURCE_STATE_KEY = (
    "trajectory_loaded_source"
)
COHORT_EDITOR_KEY = (
    "trajectory_cohort_editor"
)
OUTPUT_ROOT_KEY = (
    "trajectory_output_root"
)
STORAGE_ERROR_KEY = (
    "trajectory_storage_error"
)
LAST_RUN_STATE_KEY = (
    "trajectory_last_run"
)
CAMPAIGN_KEY = (
    "trajectory_campaign"
)
UPLOAD_DESCRIPTION_KEY = (
    "trajectory_upload_description"
)
UPLOAD_DATA_KEY = (
    "trajectory_upload_data"
)


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
                "Choose where trajectory audits "
                "should be saved"
            ),
        )

        return (
            selected
            or None
        )

    finally:
        root.destroy()


def _browse_output_root() -> None:
    current = st.session_state.get(
        OUTPUT_ROOT_KEY,
        str(
            DEFAULT_TRAJECTORY_AUDITS_DIR
        ),
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
            OUTPUT_ROOT_KEY
        ] = selected

        st.session_state.pop(
            STORAGE_ERROR_KEY,
            None,
        )


def _remove_loaded_source() -> None:
    source = st.session_state.get(
        SOURCE_STATE_KEY
    )

    if isinstance(
        source,
        TrajectoryAuditSource,
    ):
        remove_trajectory_source(
            source
        )

    st.session_state.pop(
        SOURCE_STATE_KEY,
        None,
    )

    st.session_state.pop(
        COHORT_EDITOR_KEY,
        None,
    )

    st.session_state.pop(
        LAST_RUN_STATE_KEY,
        None,
    )


def _source_mode_changed() -> None:
    _remove_loaded_source()


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



def _remember_gamebus_settings(
    *,
    email: str,
    password: str,
    remember_credentials: bool,
    campaign_abbreviation: str,
) -> None:
    settings = load_settings()

    campaigns = list(
        settings.get(
            "campaign_abbreviations",
            [],
        )
    )

    if (
        campaign_abbreviation
        and campaign_abbreviation
        not in campaigns
    ):
        campaigns.append(
            campaign_abbreviation
        )

    settings[
        "organizer_email"
    ] = email

    settings[
        "remember_credentials"
    ] = remember_credentials

    settings[
        "last_campaign_abbreviation"
    ] = campaign_abbreviation

    settings[
        "campaign_abbreviations"
    ] = campaigns

    save_settings(
        settings
    )

    if remember_credentials:
        if (
            email
            and password
        ):
            save_organizer_password(
                email,
                password,
            )

    else:
        delete_organizer_password(
            email
        )


def _initialize_state() -> dict:
    settings = load_settings()

    if (
        OUTPUT_ROOT_KEY
        not in st.session_state
    ):
        st.session_state[
            OUTPUT_ROOT_KEY
        ] = str(
            DEFAULT_TRAJECTORY_AUDITS_DIR
        )

    return settings



def _load_gamebus_source(
    *,
    campaign_abbreviation: str,
    email: str,
    password: str,
    remember_credentials: bool,
) -> None:
    _remove_loaded_source()

    try:
        source = (
            prepare_gamebus_trajectory_source(
                campaign_abbreviation=(
                    campaign_abbreviation
                ),
                organizer_email=(
                    email
                ),
                organizer_password=(
                    password
                ),
                cookie_file=(
                    get_cookie_file()
                ),
            )
        )

    except TrajectorySourceError as exc:
        st.error(
            str(
                exc
            )
        )

        return

    try:
        _remember_gamebus_settings(
            email=email,
            password=password,
            remember_credentials=(
                remember_credentials
            ),
            campaign_abbreviation=(
                campaign_abbreviation
            ),
        )

    except Exception as exc:
        st.warning(
            "Campaign loaded, but local organizer "
            "settings could not be updated: "
            f"{exc}"
        )

    st.session_state[
        SOURCE_STATE_KEY
    ] = source


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
        CAMPAIGN_KEY
        not in st.session_state
    ):
        st.session_state[
            CAMPAIGN_KEY
        ] = (
            last_campaign
            or None
        )

    selected = st.selectbox(
        "Campaign abbreviation",
        options=campaigns,
        placeholder=(
            "Select a saved campaign or "
            "enter a new abbreviation"
        ),
        accept_new_options=True,
        key=CAMPAIGN_KEY,
    )

    return (
        selected.strip()
        if selected
        else ""
    )


def _load_uploaded_source() -> None:
    description = st.session_state.get(
        UPLOAD_DESCRIPTION_KEY
    )

    data = st.session_state.get(
        UPLOAD_DATA_KEY
    )

    if (
        description is None
        or data is None
    ):
        return

    _remove_loaded_source()

    try:
        source = (
            prepare_uploaded_trajectory_source(
                campaign_description_filename=(
                    description.name
                ),
                campaign_description_bytes=(
                    description.getvalue()
                ),
                campaign_data_filename=(
                    data.name
                ),
                campaign_data_bytes=(
                    data.getvalue()
                ),
            )
        )

    except TrajectorySourceError as exc:
        st.error(
            str(
                exc
            )
        )

        return

    st.session_state[
        SOURCE_STATE_KEY
    ] = source


def _load_gamebus_source(
    *,
    campaign_abbreviation: str,
    email: str,
    password: str,
    remember_credentials: bool,
) -> None:
    _remove_loaded_source()

    try:
        source = (
            prepare_gamebus_trajectory_source(
                campaign_abbreviation=(
                    campaign_abbreviation
                ),
                organizer_email=(
                    email
                ),
                organizer_password=(
                    password
                ),
                cookie_file=(
                    get_cookie_file()
                ),
            )
        )

    except TrajectorySourceError as exc:
        st.error(
            str(
                exc
            )
        )

        return

    try:
        _remember_gamebus_settings(
            email=email,
            password=password,
            remember_credentials=(
                remember_credentials
            ),
            campaign_abbreviation=(
                campaign_abbreviation
            ),
        )

    except Exception as exc:
        st.warning(
            "Campaign loaded, but local organizer "
            "settings could not be updated: "
            f"{exc}"
        )

    st.session_state[
        SOURCE_STATE_KEY
    ] = source


def _render_upload_source() -> None:
    description = st.file_uploader(
        "Campaign description XLSX",
        type=[
            "xlsx",
        ],
        key=UPLOAD_DESCRIPTION_KEY,
    )

    data = st.file_uploader(
        "Campaign analytics ZIP",
        type=[
            "zip",
        ],
        key=UPLOAD_DATA_KEY,
    )

    st.button(
        "Load campaign files",
        type="primary",
        use_container_width=True,
        disabled=(
            description is None
            or data is None
        ),
        on_click=(
            _load_uploaded_source
        ),
    )


def _render_gamebus_source(
    settings: dict,
) -> None:
    saved_email = (
        settings.get(
            "organizer_email",
            "",
        )
        or ""
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

    credentials_are_saved = bool(
        remember_default
        and saved_email
        and saved_password
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

    load_clicked = st.button(
        "Load from GameBus",
        type="primary",
        use_container_width=True,
        disabled=(
            not campaign_abbreviation
        ),
    )

    if load_clicked:
        _load_gamebus_source(
            campaign_abbreviation=(
                campaign_abbreviation
            ),
            email=(
                organizer_email
            ),
            password=(
                organizer_password
            ),
            remember_credentials=(
                remember_credentials
            ),
        )


def _render_storage() -> Path:
    output_root_text = (
        st.text_input(
            "Folder to store trajectory audits",
            key=OUTPUT_ROOT_KEY,
        )
        .strip()
    )

    st.button(
        "Browse...",
        use_container_width=True,
        on_click=(
            _browse_output_root
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

    return Path(
        output_root_text
        or DEFAULT_TRAJECTORY_AUDITS_DIR
    ).expanduser()


def _show_source_summary(
    source: TrajectoryAuditSource,
) -> None:
    st.success(
        "Campaign source loaded."
    )

    col1, col2, col3, col4 = (
        st.columns(
            4
        )
    )

    with col1:
        st.metric(
            "Campaign",
            (
                source.campaign_abbreviation
                or "Unknown"
            ),
        )

    with col2:
        st.metric(
            "Campaign ID",
            (
                source.campaign_id
                or "Unknown"
            ),
        )

    with col3:
        source_label = (
            "GameBus"
            if (
                source.source_type
                == "gamebus"
            )
            else "Uploaded files"
        )

        st.metric(
            "Source",
            source_label,
        )

    with col4:
        st.metric(
            "Candidate participants",
            len(
                source.participants
            ),
        )

    if (
        source.source_type
        == "gamebus"
        and source.accounts_without_pid
    ):
        st.caption(
            f"{source.accounts_without_pid} "
            "GameBus account(s) had no player ID "
            "and cannot be included in trajectory "
            "analysis."
        )

    with st.expander(
        "Source files",
        expanded=False,
    ):
        st.write(
            "Campaign description:"
        )

        st.code(
            source
            .campaign_description_path
            .name
        )

        st.write(
            "Campaign analytics:"
        )

        st.code(
            source
            .campaign_data_path
            .name
        )

        st.caption(
            "When the audit runs, exact copies of "
            "these files are saved inside the new "
            "timestamped audit folder."
        )


def _render_cohort_editor(
    source: TrajectoryAuditSource,
) -> tuple[
    list[int],
    list[int],
]:
    st.subheader(
        "Review participants"
    )

    st.write(
        "Choose which participants should be "
        "included in this trajectory audit."
    )

    st.caption(
        "All available participants are included "
        "by default. Excluding an account affects "
        "only this audit run."
    )

    rows = []

    for participant in (
        source.participants
    ):
        row = {
            "Include": True,
            "PID": str(
                participant.pid
            ),
        }

        if (
            source.source_type
            == "gamebus"
        ):
            row[
                "Email"
            ] = (
                participant.email
                or ""
            )

            row[
                "Account ID"
            ] = (
                participant.account_id
                or ""
            )

        rows.append(
            row
        )

    table = pd.DataFrame(
        rows
    )

    if (
        source.source_type
        == "gamebus"
    ):
        columns = [
            "Email",
            "PID",
            "Account ID",
        ]

    else:
        columns = [
            "PID",
        ]

    edited = st.data_editor(
        table,
        key=COHORT_EDITOR_KEY,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        disabled=columns,
        column_config={
            "Include": (
                st.column_config.CheckboxColumn(
                    "Include"
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
            "Account ID": (
                st.column_config.TextColumn(
                    "Account ID"
                )
            ),
        },
    )

    candidate_ids = [
        participant.pid
        for participant in (
            source.participants
        )
    ]

    selected_ids = [
        int(
            edited.iloc[
                index
            ][
                "PID"
            ]
        )
        for index in range(
            len(
                edited
            )
        )
        if bool(
            edited.iloc[
                index
            ][
                "Include"
            ]
        )
    ]

    st.caption(
        f"{len(selected_ids)} of "
        f"{len(candidate_ids)} participants selected."
    )

    return (
        candidate_ids,
        selected_ids,
    )


def _show_run_summary(
    result,
) -> None:
    summary = (
        result.results.get(
            "audit_summary",
            {},
        )
    )

    cohort = summary.get(
        "cohort",
        {},
    )

    patterns = summary.get(
        "candidate_patterns",
        {},
    )

    cases = summary.get(
        "case_export",
        {},
    )

    st.success(
        "Trajectory audit completed."
    )

    col1, col2, col3, col4 = (
        st.columns(
            4
        )
    )

    with col1:
        st.metric(
            "Participants",
            cohort.get(
                "participants",
                0,
            ),
        )

    with col2:
        st.metric(
            "Observed participants",
            cohort.get(
                "participants_with_any_observed_event",
                0,
            ),
        )

    with col3:
        st.metric(
            "Candidate patterns",
            patterns.get(
                "rows",
                0,
            ),
        )

    with col4:
        st.metric(
            "Case exports",
            cases.get(
                "cases",
                0,
            ),
        )

    st.write(
        "Audit saved to:"
    )

    st.code(
        str(
            result.run_dir
        )
    )


def _trajectory_progress_value(
    stage: str,
) -> int:
    values = {
        "preparing": 3,
        "validating": 5,
        "data_profile": 10,
        "event_normalization": 18,
        "participation_episodes": 27,
        "observation_window": 34,
        "participant_state": 44,
        "inactivity_gaps": 54,
        "threshold_transitions": 62,
        "domain_tool_engagement": 70,
        "data_quality": 78,
        "candidate_patterns": 86,
        "case_export": 93,
        "summary": 96,
        "manifest": 98,
        "complete": 100,
    }

    return values.get(
        stage,
        0,
    )


def render_trajectory_audit_page() -> None:
    settings = (
        _initialize_state()
    )

    with st.sidebar:
        st.title(
            "GameBus Data Analyzer"
        )

        with st.expander(
            "Campaign source",
            expanded=True,
        ):
            source_mode = st.radio(
                "Choose input source",
                [
                    "Upload campaign files",
                    "Fetch campaign data from GameBus",
                ],
                key=SOURCE_MODE_KEY,
                on_change=(
                    _source_mode_changed
                ),
            )

            if (
                source_mode
                == "Upload campaign files"
            ):
                _render_upload_source()

            else:
                _render_gamebus_source(
                    settings
                )

        with st.expander(
            "Audit storage",
            expanded=True,
        ):
            output_root = (
                _render_storage()
            )

            st.caption(
                "Every run creates a new "
                "timestamped folder. Previous "
                "audits are not overwritten."
            )

    st.header(
        "Trajectory audit"
    )

    st.write(
        "Inspect longitudinal campaign trajectories "
        "and identify candidate changes in engagement "
        "that may warrant researcher review."
    )

    st.info(
        "The current trajectory audit uses the "
        "campaign description and campaign analytics "
        "export. It does not retrieve participant data "
        "from the GameBus database."
    )

    source = st.session_state.get(
        SOURCE_STATE_KEY
    )

    if not isinstance(
        source,
        TrajectoryAuditSource,
    ):
        st.info(
            "Choose a campaign source in the "
            "sidebar to continue."
        )

        return

    _show_source_summary(
        source
    )

    (
        candidate_ids,
        selected_ids,
    ) = _render_cohort_editor(
        source
    )

    st.divider()

    run_clicked = st.button(
        "Run trajectory audit",
        type="primary",
        use_container_width=True,
        disabled=(
            not selected_ids
        ),
    )

    if run_clicked:
        status = st.status(
            "Running trajectory audit...",
            expanded=True,
        )

        progress = st.progress(
            0,
            text=(
                "Preparing trajectory audit"
            ),
        )

        def update_progress(
                stage: str,
                message: str,
        ) -> None:
            value = (
                _trajectory_progress_value(
                    stage
                )
            )

            progress.progress(
                value,
                text=message,
            )

            status.write(
                message
            )

        try:
            result = (
                run_trajectory_audit_snapshot(
                    campaign_data_path=(
                        source.campaign_data_path
                    ),
                    campaign_description_path=(
                        source
                        .campaign_description_path
                    ),
                    candidate_participant_ids=(
                        candidate_ids
                    ),
                    selected_participant_ids=(
                        selected_ids
                    ),
                    source_type=(
                        source.source_type
                    ),
                    output_root=(
                        output_root
                    ),
                    progress_callback=(
                        update_progress
                    ),
                )
            )

        except Exception as exc:
            progress.empty()

            status.update(
                label=(
                    "Trajectory audit failed"
                ),
                state="error",
                expanded=True,
            )

            st.error(
                "Trajectory audit failed: "
                f"{exc}"
            )

        else:
            progress.progress(
                100,
                text=(
                    "Trajectory audit complete"
                ),
            )

            status.update(
                label=(
                    "Trajectory audit complete"
                ),
                state="complete",
                expanded=False,
            )

            st.session_state[
                LAST_RUN_STATE_KEY
            ] = result

    last_run = st.session_state.get(
        LAST_RUN_STATE_KEY
    )

    if last_run is not None:
        st.divider()

        _show_run_summary(
            last_run
        )