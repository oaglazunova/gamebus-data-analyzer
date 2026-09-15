from __future__ import annotations

from pathlib import Path
from tkinter import filedialog

import pandas as pd
import streamlit as st
import tkinter as tk

from src.datasets.inspection import (
    DatasetInspection,
    DatasetValidationError,
    inspect_dataset,
)
from src.datasets.layout import (
    DATASETS_DIR,
    get_cohort_manifest_path,
)
from src.datasets.manifests import (
    build_cohort_manifest,
    write_manifest,
)
from src.analysis.dataset_analysis import (
    DatasetAnalysisError,
    run_dataset_analysis,
)


DATASET_PATH_KEY = (
    "analyze_dataset_path"
)

SELECTED_DATASET_KEY = (
    "analyze_selected_dataset"
)

COHORT_EDITOR_KEY = (
    "analyze_cohort_editor"
)

FOLDER_ERROR_KEY = (
    "analyze_folder_error"
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
                "Choose a GameBus dataset folder"
            ),
        )

        return (
            selected
            or None
        )

    finally:
        root.destroy()


def _browse_dataset() -> None:
    current = st.session_state.get(
        DATASET_PATH_KEY,
        str(DATASETS_DIR),
    )

    try:
        selected = _choose_directory(
            current
        )

    except Exception as exc:
        st.session_state[
            FOLDER_ERROR_KEY
        ] = (
            "Could not open the folder "
            f"selector: {exc}"
        )

        return

    if selected:
        st.session_state[
            DATASET_PATH_KEY
        ] = selected

        st.session_state.pop(
            FOLDER_ERROR_KEY,
            None,
        )

        st.session_state.pop(
            SELECTED_DATASET_KEY,
            None,
        )

        st.session_state.pop(
            COHORT_EDITOR_KEY,
            None,
        )


def _load_dataset(
    dataset_path: str,
) -> DatasetInspection:
    path = Path(
        dataset_path
    ).expanduser()

    inspection = inspect_dataset(
        path
    )

    st.session_state[
        SELECTED_DATASET_KEY
    ] = str(
        inspection.dataset_dir
    )

    return inspection


def _get_current_inspection(
) -> DatasetInspection | None:
    selected = st.session_state.get(
        SELECTED_DATASET_KEY
    )

    if not selected:
        return None

    try:
        return inspect_dataset(
            Path(
                selected
            )
        )

    except DatasetValidationError:
        st.session_state.pop(
            SELECTED_DATASET_KEY,
            None,
        )

        return None


def _show_dataset_summary(
    inspection: DatasetInspection,
) -> None:
    st.success(
        "Dataset loaded successfully."
    )

    campaign_name = (
        inspection.campaign_abbreviation
        or "Unknown"
    )

    campaign_id = (
        inspection.campaign_id
        or "Unknown"
    )

    col1, col2, col3, col4 = (
        st.columns(
            4
        )
    )

    with col1:
        st.metric(
            "Campaign",
            campaign_name,
        )

    with col2:
        st.metric(
            "Campaign ID",
            campaign_id,
        )

    with col3:
        if inspection.has_cohort_manifest:
            st.metric(
                "Selected participants",
                (
                    inspection
                    .selected_for_analysis_count
                ),
            )

        else:
            st.metric(
                "Selected participants",
                "Not reviewed",
            )

    with col4:
        st.metric(
            "Raw JSON files",
            inspection.raw_json_count,
        )

    st.caption(
        str(
            inspection.dataset_dir
        )
    )

    with st.expander(
        "Dataset files",
        expanded=False,
    ):
        st.write(
            "Campaign analytics:"
        )

        st.code(
            inspection
            .campaign_data_path
            .name
        )

        st.write(
            "Campaign description:"
        )

        st.code(
            inspection
            .campaign_description_path
            .name
        )

        if (
            inspection.raw_data_dir.exists()
        ):
            st.write(
                "Participant-level data:"
            )

            st.code(
                str(
                    inspection.raw_data_dir
                )
            )

        st.write(
            "Analysis output:"
        )

        st.code(
            str(
                inspection.analysis_dir
            )
        )


def _render_existing_cohort(
    inspection: DatasetInspection,
) -> None:
    manifest = (
        inspection.cohort_manifest
    )

    if manifest is None:
        return

    participants = manifest.get(
        "participants",
        [],
    )

    if not participants:
        st.warning(
            "The cohort manifest does not "
            "contain any participants."
        )

        return

    with st.expander(
        "View / change analysis cohort",
        expanded=False,
    ):
        st.write(
            "Choose which participants should "
            "be included in analysis."
        )

        st.caption(
            "Changing this selection does not "
            "delete downloaded data and does not "
            "trigger participant-level extraction."
        )

        rows = []

        for participant in participants:
            rows.append(
                {
                    "Include": bool(
                        participant.get(
                            "selected_for_analysis",
                            False,
                        )
                    ),
                    "Email": (
                        participant.get(
                            "email"
                        )
                        or ""
                    ),
                    "PID": (
                        participant.get(
                            "pid"
                        )
                        or ""
                    ),
                    "Participant data": bool(
                        participant.get(
                            "participant_data_extracted",
                            False,
                        )
                    ),
                }
            )

        table = pd.DataFrame(
            rows
        )

        edited = st.data_editor(
            table,
            key=COHORT_EDITOR_KEY,
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            disabled=[
                "Email",
                "PID",
                "Participant data",
            ],
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
                "Participant data": (
                    st.column_config.CheckboxColumn(
                        "Participant data",
                        help=(
                            "Participant-level data "
                            "were previously extracted."
                        ),
                    )
                ),
            },
        )

        selected_count = int(
            edited[
                "Include"
            ].sum()
        )

        st.caption(
            f"{selected_count} of "
            f"{len(participants)} participants "
            "selected for analysis."
        )

        if st.button(
            "Save analysis cohort",
            type="primary",
            use_container_width=True,
        ):
            updated_participants = []

            for index, participant in enumerate(
                participants
            ):
                updated = dict(
                    participant
                )

                updated[
                    "selected_for_analysis"
                ] = bool(
                    edited.iloc[
                        index
                    ][
                        "Include"
                    ]
                )

                updated_participants.append(
                    updated
                )

            campaign = manifest.get(
                "campaign",
                {},
            )

            verification = manifest.get(
                "verification",
                {},
            )

            updated_manifest = (
                build_cohort_manifest(
                    campaign_abbreviation=(
                        campaign.get(
                            "abbreviation"
                        )
                        or (
                            inspection
                            .campaign_abbreviation
                        )
                        or "UNKNOWN"
                    ),
                    campaign_id=(
                        campaign.get(
                            "id"
                        )
                        or inspection.campaign_id
                        or "UNKNOWN"
                    ),
                    participants=(
                        updated_participants
                    ),
                    candidate_source=(
                        verification.get(
                            "candidate_source"
                        )
                        or "gamebus_studio_users"
                    ),
                )
            )

            write_manifest(
                get_cohort_manifest_path(
                    inspection.dataset_dir
                ),
                updated_manifest,
            )

            st.success(
                "Analysis cohort saved."
            )

            st.rerun()



def _analysis_progress_value(
    stage: str,
) -> int:
    values = {
        "validating": 5,
        "preparing": 10,
        "loading_campaign": 20,
        "loading_participant_data": 30,
        "filtering": 40,
        "activities": 55,
        "geofence": 70,
        "steps": 78,
        "campaign_configuration": 86,
        "report": 94,
        "complete": 100,
    }

    return values.get(
        stage,
        0,
    )


def _render_analysis_status(
    inspection: DatasetInspection,
) -> None:
    st.divider()

    if not inspection.has_cohort_manifest:
        st.warning(
            "This dataset does not yet have a "
            "saved participant cohort."
        )

        st.info(
            "Participant review is required "
            "before analysis."
        )

        return

    selected_count = (
        inspection
        .selected_for_analysis_count
    )

    if selected_count == 0:
        st.warning(
            "No participants are currently "
            "selected for analysis."
        )

        return

    st.subheader(
        "Analysis"
    )

    st.write(
        f"{selected_count} participant(s) "
        "will be included in the analysis."
    )

    if inspection.analysis_dir.exists():
        st.caption(
            "Running the analysis again will "
            "replace the existing contents of "
            "data_analysis/."
        )

    run_clicked = st.button(
        "Run analysis",
        type="primary",
        use_container_width=True,
    )

    if not run_clicked:
        return

    status = st.status(
        "Running analysis...",
        expanded=True,
    )

    progress = st.progress(
        0,
        text="Starting analysis",
    )

    def update_progress(
        stage: str,
        message: str,
    ) -> None:
        value = (
            _analysis_progress_value(
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
        result = run_dataset_analysis(
            inspection.dataset_dir,
            progress_callback=(
                update_progress
            ),
        )

    except DatasetAnalysisError as exc:
        progress.empty()

        status.update(
            label="Analysis failed",
            state="error",
            expanded=True,
        )

        st.error(
            str(
                exc
            )
        )

        return

    except Exception as exc:
        progress.empty()

        status.update(
            label="Analysis failed",
            state="error",
            expanded=True,
        )

        st.error(
            "Unexpected error while running "
            f"analysis: {exc}"
        )

        return

    progress.progress(
        100,
        text="Analysis complete",
    )

    status.update(
        label="Analysis complete",
        state="complete",
        expanded=False,
    )

    st.success(
        "Analysis completed successfully."
    )

    col1, col2, col3 = st.columns(
        3
    )

    with col1:
        st.metric(
            "Participants analyzed",
            result.selected_participants,
        )

    with col2:
        st.metric(
            "Campaign tables loaded",
            result.campaign_tables_loaded,
        )

    with col3:
        st.metric(
            "Active participants",
            (
                result.unique_active_users
                if (
                    result.unique_active_users
                    is not None
                )
                else "—"
            ),
        )

    st.write(
        "Results were written to:"
    )

    st.code(
        str(
            result.output_dir
        )
    )

    if result.report_path is not None:
        st.write(
            "Report:"
        )

        st.code(
            str(
                result.report_path
            )
        )


def render_analyze_data_page() -> None:
    if (
        DATASET_PATH_KEY
        not in st.session_state
    ):
        st.session_state[
            DATASET_PATH_KEY
        ] = str(
            DATASETS_DIR
        )

    with st.sidebar:
        st.title(
            "GameBus Data Analyzer"
        )

        st.subheader(
            "Dataset"
        )

        dataset_path = (
            st.text_input(
                "Dataset folder",
                key=DATASET_PATH_KEY,
            )
            .strip()
        )

        st.button(
            "Browse...",
            use_container_width=True,
            on_click=_browse_dataset,
        )

        folder_error = (
            st.session_state.get(
                FOLDER_ERROR_KEY
            )
        )

        if folder_error:
            st.error(
                folder_error
            )

        load_clicked = st.button(
            "Load dataset",
            type="primary",
            use_container_width=True,
            disabled=(
                not dataset_path
            ),
        )

    st.header(
        "Analyze existing data"
    )

    st.write(
        "Choose an existing GameBus dataset "
        "folder. Organizer login and participant "
        "credentials are not required."
    )

    inspection = (
        _get_current_inspection()
    )

    if load_clicked:
        try:
            inspection = _load_dataset(
                dataset_path
            )

        except DatasetValidationError as exc:
            st.session_state.pop(
                SELECTED_DATASET_KEY,
                None,
            )

            st.error(
                str(
                    exc
                )
            )

            return

        except Exception as exc:
            st.session_state.pop(
                SELECTED_DATASET_KEY,
                None,
            )

            st.error(
                "Could not load dataset: "
                f"{exc}"
            )

            return

    if inspection is None:
        st.info(
            "Select a dataset folder in the "
            "sidebar to continue."
        )

        return

    _show_dataset_summary(
        inspection
    )

    if inspection.has_cohort_manifest:
        _render_existing_cohort(
            inspection
        )

    _render_analysis_status(
        inspection
    )