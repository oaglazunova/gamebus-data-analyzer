from __future__ import annotations

from pathlib import Path
import json
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
RESULT_VIEW_KEY = (
    "trajectory_result_view"
)
PARTICIPANT_VIEW_KEY = (
    "trajectory_result_participant"
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


def _clear_result_state() -> None:
    for key in (
        LAST_RUN_STATE_KEY,
        RESULT_VIEW_KEY,
        PARTICIPANT_VIEW_KEY,
    ):
        st.session_state.pop(
            key,
            None,
        )


def _source_mode_changed() -> None:
    _remove_loaded_source()

    source_mode = st.session_state.get(
        SOURCE_MODE_KEY
    )

    if source_mode not in {
        "Upload campaign files",
        "Fetch campaign data from GameBus",
    }:
        return

    try:
        settings = load_settings()

        settings[
            "trajectory_source_mode"
        ] = source_mode

        save_settings(
            settings
        )

    except Exception:
        pass


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
        SOURCE_MODE_KEY
        not in st.session_state
    ):
        saved_source_mode = settings.get(
            "trajectory_source_mode",
            "Upload campaign files",
        )

        if saved_source_mode not in {
            "Upload campaign files",
            "Fetch campaign data from GameBus",
        }:
            saved_source_mode = (
                "Upload campaign files"
            )

        st.session_state[
            SOURCE_MODE_KEY
        ] = saved_source_mode

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

    _clear_result_state()

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

    _clear_result_state()

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


def _build_participant_status_table(
    result,
) -> pd.DataFrame:
    participant_ids = (
        _selected_participant_ids(
            result
        )
    )

    columns = [
        "PID",
        "Current state",
        "Last explicit engagement",
        "Days since engagement",
        "Candidate patterns",
        "Latest detected",
        "Ongoing at audit cutoff",
        "Data quality",
    ]

    if not participant_ids:
        return pd.DataFrame(
            columns=columns
        )

    state = _read_result_table(
        result,
        "participant_state_daily.csv",
    )

    quality = _read_result_table(
        result,
        "data_quality.csv",
    )

    patterns = _read_result_table(
        result,
        "candidate_patterns.csv",
    )

    rows = []

    for participant_id in participant_ids:
        participant_state = (
            _participant_rows(
                state,
                participant_id,
            )
        )

        participant_quality = (
            _participant_rows(
                quality,
                participant_id,
            )
        )

        participant_patterns = (
            _participant_rows(
                patterns,
                participant_id,
            )
        )

        current_state = "Unknown"
        last_engagement = "—"
        days_since = "—"

        if not participant_state.empty:
            participant_state = (
                participant_state.copy()
            )

            participant_state[
                "date"
            ] = pd.to_datetime(
                participant_state[
                    "date"
                ],
                errors="coerce",
            )

            participant_state = (
                participant_state.sort_values(
                    "date"
                )
            )

            current = (
                participant_state.iloc[-1]
            )

            raw_state = current.get(
                "engagement_state"
            )

            if pd.notna(
                raw_state
            ):
                current_state = str(
                    raw_state
                ).replace(
                    "_",
                    " ",
                )

            raw_last = current.get(
                "last_explicit_engagement_date"
            )

            if pd.notna(
                raw_last
            ):
                parsed_last = pd.to_datetime(
                    raw_last,
                    errors="coerce",
                )

                if pd.notna(
                    parsed_last
                ):
                    last_engagement = (
                        _format_date(
                            parsed_last
                        )
                    )

                else:
                    last_engagement = str(
                        raw_last
                    )

            raw_days = current.get(
                "days_since_last_explicit_engagement"
            )

            if pd.notna(
                raw_days
            ):
                try:
                    days_since = int(
                        float(
                            raw_days
                        )
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    days_since = str(
                        raw_days
                    )

        quality_state = "Unknown"

        if not participant_quality.empty:
            raw_quality = (
                participant_quality
                .iloc[0]
                .get(
                    "core_quality_state"
                )
            )

            if pd.notna(
                raw_quality
            ):
                quality_state = (
                    str(
                        raw_quality
                    )
                    .replace(
                        "_",
                        " ",
                    )
                )

        pattern_text = "—"
        latest_detected = "—"
        ongoing_at_cutoff = "No"

        if not participant_patterns.empty:
            pattern_names = []

            if (
                "pattern_type"
                in participant_patterns.columns
            ):
                for value in (
                    participant_patterns[
                        "pattern_type"
                    ]
                    .dropna()
                    .astype(str)
                    .tolist()
                ):
                    friendly = (
                        value.replace(
                            "_",
                            " ",
                        )
                    )

                    if friendly not in pattern_names:
                        pattern_names.append(
                            friendly
                        )

            if pattern_names:
                pattern_text = (
                    f"{len(participant_patterns)} — "
                    + "; ".join(
                        pattern_names
                    )
                )

            if (
                "detected_at"
                in participant_patterns.columns
            ):
                detected = pd.to_datetime(
                    participant_patterns[
                        "detected_at"
                    ],
                    errors="coerce",
                ).dropna()

                if not detected.empty:
                    latest_detected = (
                        _format_date(
                            detected.max()
                        )
                    )

            if (
                "right_censored"
                in participant_patterns.columns
            ):
                right_censored = (
                    participant_patterns[
                        "right_censored"
                    ]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .isin(
                        [
                            "true",
                            "1",
                            "yes",
                        ]
                    )
                )

                if right_censored.any():
                    ongoing_at_cutoff = "Yes"

        rows.append(
            {
                "PID": participant_id,
                "Current state": (
                    current_state
                ),
                "Last explicit engagement": (
                    last_engagement
                ),
                "Days since engagement": (
                    days_since
                ),
                "Candidate patterns": (
                    pattern_text
                ),
                "Latest detected": (
                    latest_detected
                ),
                "Ongoing at audit cutoff": (
                    ongoing_at_cutoff
                ),
                "Data quality": (
                    quality_state
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=columns,
    )



def _show_campaign_overview(
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

    st.subheader(
        "Campaign overview"
    )

    # -------------------------------------------------
    # Main audit metrics
    # -------------------------------------------------

    col1, col2, col3, col4, col5 = (
        st.columns(
            5
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
            "Observed",
            cohort.get(
                "participants_with_any_observed_event",
                0,
            ),
        )

    with col3:
        st.metric(
            "Explicit engagement",
            cohort.get(
                "participants_with_explicit_engagement",
                0,
            ),
        )

    with col4:
        st.metric(
            "Candidate patterns",
            patterns.get(
                "rows",
                0,
            ),
        )

    with col5:
        st.metric(
            "Case exports",
            cases.get(
                "cases",
                0,
            ),
        )


    # -------------------------------------------------
    # Participant status
    # -------------------------------------------------

    st.subheader(
        "Participant status"
    )

    participant_status = (
        _build_participant_status_table(
            result
        )
    )

    if participant_status.empty:
        st.info(
            "No participant status information "
            "is available for this audit."
        )

    else:
        st.dataframe(
            participant_status,
            hide_index=True,
            use_container_width=True,
        )


    # -------------------------------------------------
    # Campaign engagement over time
    # -------------------------------------------------

    st.subheader(
        "Engagement over time"
    )

    engagement_plot = (
        result.results_dir
        / "plots"
        / "cohort_engagement_over_time.png"
    )

    if engagement_plot.is_file():
        st.image(
            str(
                engagement_plot
            ),
            use_container_width=True,
        )

    else:
        st.info(
            "No cohort engagement plot was produced "
            "for this audit."
        )

    # -------------------------------------------------
    # Domain / tool engagement
    # -------------------------------------------------

    st.subheader(
        "Domain and tool engagement"
    )

    domain_tool_plot = (
        result.results_dir
        / "plots"
        / "cohort_domain_tool_engagement.png"
    )

    if domain_tool_plot.is_file():
        st.image(
            str(
                domain_tool_plot
            ),
            use_container_width=True,
        )

        st.caption(
            "The first two panels show explicit "
            "intervention engagement. The final panel "
            "shows other observed activity, such as "
            "navigation or passive sensor evidence; it "
            "does not count as explicit engagement."
        )

    else:
        st.info(
            "No cohort domain/tool plot was produced "
            "for this audit."
        )


    # -------------------------------------------------
    # Saved audit
    # -------------------------------------------------

    with st.expander(
        "Audit files",
        expanded=False,
    ):
        st.write(
            "Audit saved to:"
        )

        st.code(
            str(
                result.run_dir
            )
        )

        st.caption(
            "The tables and figures shown above are "
            "also saved inside the audit results folder."
        )



def _selected_participant_ids(
    result,
) -> list[int]:
    """
    Read the participants included in this audit run.
    """

    if not result.cohort_path.is_file():
        return []

    try:
        cohort = json.loads(
            result.cohort_path.read_text(
                encoding="utf-8"
            )
        )

    except Exception:
        return []

    participant_ids = []

    for participant in cohort.get(
        "participants",
        [],
    ):
        if not participant.get(
            "included",
            False,
        ):
            continue

        try:
            participant_ids.append(
                int(
                    participant[
                        "pid"
                    ]
                )
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

    return sorted(
        set(
            participant_ids
        )
    )


def _read_result_table(
    result,
    filename: str,
) -> pd.DataFrame:
    path = (
        result.results_dir
        / filename
    )

    if not path.is_file():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            path
        )

    except Exception:
        return pd.DataFrame()


def _participant_rows(
    table: pd.DataFrame,
    participant_id: int,
) -> pd.DataFrame:
    if (
        table.empty
        or "participant_id"
        not in table.columns
    ):
        return table.iloc[
            0:0
        ].copy()

    ids = pd.to_numeric(
        table[
            "participant_id"
        ],
        errors="coerce",
    )

    return (
        table.loc[
            ids
            == participant_id
        ]
        .copy()
    )


def _friendly_label(
    value,
) -> str:
    if pd.isna(
        value
    ):
        return "—"

    return (
        str(
            value
        )
        .replace(
            "_",
            " ",
        )
    )


def _format_date(
    value,
) -> str:
    if pd.isna(
        value
    ):
        return "—"

    parsed = pd.to_datetime(
        value,
        errors="coerce",
    )

    if pd.isna(
        parsed
    ):
        return str(
            value
        )

    return parsed.strftime(
        "%d-%m-%Y"
    )


def _yes_no(
    value,
) -> str:
    if pd.isna(
        value
    ):
        return "No"

    if isinstance(
        value,
        bool,
    ):
        return (
            "Yes"
            if value
            else "No"
        )

    return (
        "Yes"
        if str(
            value
        ).strip().lower()
        in {
            "true",
            "1",
            "yes",
        }
        else "No"
    )



def _format_flags(
    value,
) -> str:
    if pd.isna(
        value
    ):
        return "None"

    if isinstance(
        value,
        list,
    ):
        flags = value

    else:
        text = str(
            value
        ).strip()

        if (
            not text
            or text == "[]"
        ):
            return "None"

        try:
            parsed = json.loads(
                text
            )

            flags = (
                parsed
                if isinstance(
                    parsed,
                    list,
                )
                else [
                    parsed
                ]
            )

        except (
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            flags = [
                text
            ]

    if not flags:
        return "None"

    return "; ".join(
        _friendly_label(
            flag
        )
        for flag in flags
    )



def _format_pattern_evidence(
    value,
) -> str:
    if pd.isna(
        value
    ):
        return "—"

    try:
        evidence = json.loads(
            str(
                value
            )
        )

    except (
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return str(
            value
        )

    if not isinstance(
        evidence,
        dict,
    ):
        return str(
            value
        )

    parts = []

    label_map = {
        "days_since_last_explicit_engagement": (
            "days since last engagement"
        ),
        "last_explicit_engagement_date": (
            "last engagement"
        ),
        "inactive_days": (
            "inactive days"
        ),
        "previous_engagement_date": (
            "previous engagement"
        ),
        "reengagement_date": (
            "re-engagement"
        ),
        "reference_events": (
            "reference events"
        ),
        "recent_events": (
            "recent events"
        ),
        "recent_to_reference_ratio": (
            "recent/reference ratio"
        ),
        "decline_ratio_threshold": (
            "decline threshold"
        ),
    }

    preferred_keys = [
        "days_since_last_explicit_engagement",
        "last_explicit_engagement_date",
        "inactive_days",
        "previous_engagement_date",
        "reengagement_date",
        "reference_events",
        "recent_events",
        "recent_to_reference_ratio",
        "decline_ratio_threshold",
    ]

    for key in preferred_keys:
        if key not in evidence:
            continue

        raw_value = evidence[
            key
        ]

        if "date" in key:
            display_value = (
                _format_date(
                    raw_value
                )
            )

        else:
            display_value = str(
                raw_value
            )

        parts.append(
            (
                f"{label_map[key]}: "
                f"{display_value}"
            )
        )

    if parts:
        return "; ".join(
            parts
        )

    return "; ".join(
        (
            f"{_friendly_label(key)}: "
            f"{value}"
        )
        for key, value
        in evidence.items()
    )



def _show_participant_inspector(
    result,
) -> None:
    st.subheader(
        "Participant"
    )

    participant_ids = (
        _selected_participant_ids(
            result
        )
    )

    if not participant_ids:
        st.info(
            "No included participants were found "
            "for this audit."
        )
        return

    email_by_pid = {}

    source = st.session_state.get(
        SOURCE_STATE_KEY
    )

    if isinstance(
            source,
            TrajectoryAuditSource,
    ):
        for participant in (
                source.participants
        ):
            try:
                pid = int(
                    participant.pid
                )

            except (
                    TypeError,
                    ValueError,
            ):
                continue

            email = (
                    participant.email
                    or ""
            ).strip()

            if email:
                email_by_pid[
                    pid
                ] = email

    def participant_label(
            pid: int,
    ) -> str:
        email = email_by_pid.get(
            pid
        )

        if email:
            return (
                f"{email} · PID {pid}"
            )

        return f"PID {pid}"

    participant_id = st.selectbox(
        "Participant",
        options=participant_ids,
        key=PARTICIPANT_VIEW_KEY,
        format_func=participant_label,
    )

    state = _participant_rows(
        _read_result_table(
            result,
            "participant_state_daily.csv",
        ),
        participant_id,
    )

    quality = _participant_rows(
        _read_result_table(
            result,
            "data_quality.csv",
        ),
        participant_id,
    )

    patterns = _participant_rows(
        _read_result_table(
            result,
            "candidate_patterns.csv",
        ),
        participant_id,
    )

    # -------------------------------------------------
    # Current trajectory state
    # -------------------------------------------------

    current_state = "Unknown"
    last_engagement = "—"
    days_since = "—"

    if not state.empty:
        state[
            "date"
        ] = pd.to_datetime(
            state[
                "date"
            ],
            errors="coerce",
        )

        state = (
            state.sort_values(
                "date"
            )
        )

        current = (
            state.iloc[-1]
        )

        current_state = (
            str(
                current.get(
                    "engagement_state",
                    "Unknown",
                )
            )
        )

        raw_last_engagement = (
            current.get(
                "last_explicit_engagement_date"
            )
        )

        if pd.notna(
            raw_last_engagement
        ):
            last_engagement = (
                _format_date(
                    raw_last_engagement
                )
            )

        raw_days_since = (
            current.get(
                "days_since_last_explicit_engagement"
            )
        )

        if pd.notna(
            raw_days_since
        ):
            try:
                days_since = str(
                    int(
                        float(
                            raw_days_since
                        )
                    )
                )

            except (
                TypeError,
                ValueError,
            ):
                days_since = str(
                    raw_days_since
                )

    # -------------------------------------------------
    # Data quality
    # -------------------------------------------------

    quality_state = "Unknown"

    if not quality.empty:
        raw_quality = (
            quality.iloc[0].get(
                "core_quality_state"
            )
        )

        if pd.notna(
            raw_quality
        ):
            quality_state = str(
                raw_quality
            )

    # -------------------------------------------------
    # Summary metrics
    # -------------------------------------------------

    col1, col2, col3, col4 = (
        st.columns(
            4
        )
    )

    with col1:
        st.metric(
            "Current state",
            current_state,
        )

    with col2:
        st.metric(
            "Last explicit engagement",
            last_engagement,
        )

    with col3:
        st.metric(
            "Days since engagement",
            days_since,
        )

    with col4:
        st.metric(
            "Candidate patterns",
            len(
                patterns
            ),
        )

    st.caption(
        f"Data quality: {quality_state}"
    )

    # -------------------------------------------------
    # Trajectory
    # -------------------------------------------------

    st.subheader(
        "Trajectory"
    )

    trajectory_plot = (
            result.results_dir
            / "plots"
            / "participants"
            / (
                f"participant_"
                f"{participant_id}_trajectory.png"
            )
    )

    if trajectory_plot.is_file():
        st.image(
            str(
                trajectory_plot
            ),
            use_container_width=True,
        )

    else:
        st.info(
            "No trajectory plot was produced "
            "for this participant."
        )

    # -------------------------------------------------
    # Behavioral domains and tools
    # -------------------------------------------------

    st.subheader(
        "Domains and tools"
    )

    domain_tool_plot = (
            result.results_dir
            / "plots"
            / "participants"
            / (
                f"participant_"
                f"{participant_id}_domain_tool.png"
            )
    )

    if domain_tool_plot.is_file():
        st.image(
            str(
                domain_tool_plot
            ),
            use_container_width=True,
        )

        st.caption(
            "The first two panels show explicit "
            "intervention engagement. The final panel "
            "shows other observed activity, such as "
            "navigation or passive sensor evidence; it "
            "does not count as explicit engagement."
        )

    else:
        st.info(
            "No domain/tool trajectory plot was "
            "produced for this participant."
        )

    # -------------------------------------------------
    # Candidate changes
    # -------------------------------------------------

    st.subheader(
        "Candidate changes"
    )

    if patterns.empty:
        st.info(
            "No candidate trajectory changes were "
            "detected for this participant."
        )

    else:
        candidate_table = (
            patterns.copy()
        )

        candidate_table[
            "Pattern"
        ] = candidate_table[
            "pattern_type"
        ].apply(
            _friendly_label
        )

        candidate_table[
            "Detected at"
        ] = candidate_table[
            "detected_at"
        ].apply(
            _format_date
        )

        candidate_table[
            "Ongoing at audit cutoff"
        ] = candidate_table[
            "right_censored"
        ].apply(
            _yes_no
        )

        candidate_table[
            "Evidence"
        ] = candidate_table[
            "evidence"
        ].apply(
            _format_pattern_evidence
        )

        display_columns = [
            "Pattern",
            "Detected at",
            "Ongoing at audit cutoff",
            "Evidence",
        ]

        if (
                "subject"
                in candidate_table.columns
                and candidate_table[
            "subject"
        ].notna().any()
        ):
            candidate_table[
                "Subject"
            ] = candidate_table[
                "subject"
            ].apply(
                _friendly_label
            )

            display_columns.insert(
                1,
                "Subject",
            )

        st.dataframe(
            candidate_table[
                display_columns
            ],
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "\"Ongoing at audit cutoff\" means that "
            "the observed pattern had not reached an "
            "observed ending before the available "
            "observation period ended."
        )

    # -------------------------------------------------
    # Participation episodes
    # -------------------------------------------------

    st.subheader(
        "Participation episodes"
    )

    episodes = _participant_rows(
        _read_result_table(
            result,
            "participation_episodes.csv",
        ),
        participant_id,
    )

    if episodes.empty:
        st.info(
            "No participation episodes were identified "
            "for this participant."
        )

    else:
        episodes = (
            episodes.copy()
        )

        episodes[
            "Episode"
        ] = episodes[
            "episode_number"
        ]

        episodes[
            "Start"
        ] = episodes[
            "episode_start"
        ].apply(
            _format_date
        )

        episodes[
            "End"
        ] = episodes[
            "episode_end"
        ].apply(
            _format_date
        )

        episodes[
            "Duration (days)"
        ] = episodes[
            "duration_days"
        ]

        episodes[
            "Active days"
        ] = episodes[
            "active_days"
        ]

        episodes[
            "Explicit events"
        ] = episodes[
            "event_count"
        ]

        episodes[
            "Points"
        ] = episodes[
            "points"
        ]

        st.dataframe(
            episodes[
                [
                    "Episode",
                    "Start",
                    "End",
                    "Duration (days)",
                    "Active days",
                    "Explicit events",
                    "Points",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "A new participation episode begins after "
            "a gap of more than 14 days between explicit "
            "engagement dates."
        )

    # -------------------------------------------------
    # Meaningful inactivity gaps
    # -------------------------------------------------

    st.subheader(
        "Meaningful inactivity gaps"
    )

    gaps = _participant_rows(
        _read_result_table(
            result,
            "inactivity_gaps.csv",
        ),
        participant_id,
    )

    if not gaps.empty:
        gaps[
            "inactive_days"
        ] = pd.to_numeric(
            gaps[
                "inactive_days"
            ],
            errors="coerce",
        )

        gaps = (
            gaps.loc[
                gaps[
                    "inactive_days"
                ]
                >= 7
                ]
            .copy()
        )

    if gaps.empty:
        st.info(
            "No inactivity gaps of 7 days or longer "
            "were observed for this participant."
        )

    else:
        gaps[
            "Start"
        ] = gaps[
            "gap_start"
        ].apply(
            _format_date
        )

        gaps[
            "End"
        ] = gaps[
            "gap_end"
        ].apply(
            _format_date
        )

        gaps[
            "Inactive days"
        ] = gaps[
            "inactive_days"
        ]

        gaps[
            "Re-engaged"
        ] = gaps[
            "reengaged"
        ].apply(
            _yes_no
        )

        gaps[
            "Ongoing at audit cutoff"
        ] = gaps[
            "right_censored"
        ].apply(
            _yes_no
        )

        gaps[
            "Threshold reached"
        ] = gaps.apply(
            lambda row: (
                "21 days"
                if _yes_no(
                    row.get(
                        "reached_21d"
                    )
                )
                   == "Yes"
                else (
                    "14 days"
                    if _yes_no(
                        row.get(
                            "reached_14d"
                        )
                    )
                       == "Yes"
                    else "7 days"
                )
            ),
            axis=1,
        )

        st.dataframe(
            gaps[
                [
                    "Start",
                    "End",
                    "Inactive days",
                    "Threshold reached",
                    "Re-engaged",
                    "Ongoing at audit cutoff",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )

    # -------------------------------------------------
    # Observation and data quality
    # -------------------------------------------------

    st.subheader(
        "Observation and data quality"
    )

    audit_summary = (
        result.results.get(
            "audit_summary",
            {},
        )
    )

    audit_quality = (
        audit_summary.get(
            "data_quality",
            {},
        )
    )

    observation_window = (
        audit_quality.get(
            "observation_window",
            {},
        )
    )

    if quality.empty:
        st.info(
            "No participant data-quality information "
            "is available."
        )

    else:
        quality_row = quality.iloc[0]

        col1, col2, col3, col4 = (
            st.columns(
                4
            )
        )

        with col1:
            st.metric(
                "First observed",
                _format_date(
                    quality_row.get(
                        "first_observed_at"
                    )
                ),
            )

        with col2:
            st.metric(
                "Last observed",
                _format_date(
                    quality_row.get(
                        "last_observed_at"
                    )
                ),
            )

        with col3:
            total_events = pd.to_numeric(
                quality_row.get(
                    "total_events"
                ),
                errors="coerce",
            )

            st.metric(
                "Observed events",
                (
                    int(
                        total_events
                    )
                    if pd.notna(
                        total_events
                    )
                    else "—"
                ),
            )

        with col4:
            explicit_events = pd.to_numeric(
                quality_row.get(
                    "explicit_engagement_events"
                ),
                errors="coerce",
            )

            st.metric(
                "Explicit engagement events",
                (
                    int(
                        explicit_events
                    )
                    if pd.notna(
                        explicit_events
                    )
                    else "—"
                ),
            )

        # ---------------------------------------------
        # Audit observation window
        # ---------------------------------------------

        st.markdown(
            "**Audit observation window**"
        )

        window_col1, window_col2 = (
            st.columns(
                2
            )
        )

        with window_col1:
            st.metric(
                "Observation start",
                _format_date(
                    observation_window.get(
                        "effective_start"
                    )
                ),
            )

        with window_col2:
            st.metric(
                "Audit cutoff",
                _format_date(
                    observation_window.get(
                        "analysis_cutoff"
                    )
                ),
            )

        start_source = (
            observation_window.get(
                "start_source"
            )
        )

        cutoff_source = (
            observation_window.get(
                "cutoff_source"
            )
        )

        if (
                start_source
                or cutoff_source
        ):
            st.caption(
                "Observation window: "
                f"start = "
                f"{_friendly_label(start_source)}; "
                f"cutoff = "
                f"{_friendly_label(cutoff_source)}."
            )

        # ---------------------------------------------
        # Stream availability
        # ---------------------------------------------

        st.markdown(
            "**Data streams**"
        )

        stream_columns = [
            (
                "Activity",
                "activity_stream_state",
            ),
            (
                "Navigation",
                "navigation_stream_state",
            ),
            (
                "Notifications",
                "notification_stream_state",
            ),
            (
                "Sensor",
                "sensor_stream_state",
            ),
            (
                "Garmin",
                "garmin_stream_state",
            ),
            (
                "Nutrida",
                "nutrida_stream_state",
            ),
        ]

        stream_rows = []

        for label, column in stream_columns:
            if column not in quality.columns:
                continue

            raw_state = quality_row.get(
                column
            )

            stream_rows.append(
                {
                    "Data stream": label,
                    "Availability": (
                        _friendly_label(
                            raw_state
                        )
                    ),
                }
            )

        if stream_rows:
            st.dataframe(
                pd.DataFrame(
                    stream_rows
                ),
                hide_index=True,
                use_container_width=True,
            )

        st.caption(
            "\"Unavailable\" means that the audit did "
            "not have that data stream. It must not be "
            "interpreted as zero participant behavior. "
            "\"Available empty\" means that the stream "
            "was present but contained no observed events."
        )

        # ---------------------------------------------
        # Quality / observation flags
        # ---------------------------------------------

        quality_flags = _format_flags(
            quality_row.get(
                "quality_flags"
            )
        )

        observation_flags = _format_flags(
            quality_row.get(
                "observation_flags"
            )
        )

        if (
                quality_flags != "None"
                or observation_flags != "None"
        ):
            st.markdown(
                "**Cautions**"
            )

            if quality_flags != "None":
                st.write(
                    "Data quality: "
                    f"{quality_flags}"
                )

            if observation_flags != "None":
                st.write(
                    "Observation: "
                    f"{observation_flags}"
                )

        else:
            st.caption(
                "No participant-specific data-quality "
                "or observation cautions were recorded."
            )



def _show_run_summary(
    result,
) -> None:
    st.success(
        "Trajectory audit completed."
    )

    if (
        RESULT_VIEW_KEY
        not in st.session_state
    ):
        st.session_state[
            RESULT_VIEW_KEY
        ] = "Campaign"

    st.divider()

    view = st.segmented_control(
        "Results view",
        options=[
            "Campaign",
            "Participant",
        ],
        selection_mode="single",
        key=RESULT_VIEW_KEY,
        width="content",
    )

    if view == "Participant":
        _show_participant_inspector(
            result
        )

    else:
        _show_campaign_overview(
            result
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

    last_run = st.session_state.get(
        LAST_RUN_STATE_KEY
    )

    if not isinstance(
            source,
            TrajectoryAuditSource,
    ):
        if last_run is not None:
            _show_run_summary(
                last_run
            )

        else:
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

            st.session_state[
                RESULT_VIEW_KEY
            ] = "Campaign"

            st.session_state.pop(
                PARTICIPANT_VIEW_KEY,
                None,
            )

    last_run = st.session_state.get(
        LAST_RUN_STATE_KEY
    )

    if last_run is not None:
        st.divider()

        _show_run_summary(
            last_run
        )