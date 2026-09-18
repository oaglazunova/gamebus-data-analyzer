from __future__ import annotations

import streamlit as st

from src.ui.analyze_data import (
    render_analyze_data_page,
)
from src.ui.get_data import (
    render_get_data_page,
)
from src.ui.trajectory_audit import (
    render_trajectory_audit_page,
)


st.set_page_config(
    page_title="GameBus Data Analyzer",
    page_icon="📊",
    layout="wide",
)


def main() -> None:
    action = st.radio(
        "Action",
        [
            "Get data",
            "Analyze",
            "Trajectory audit",
        ],
        horizontal=True,
        label_visibility="collapsed",
        key="main_action",
    )

    st.divider()

    if action == "Get data":
        render_get_data_page()

    elif action == "Analyze":
        render_analyze_data_page()

    else:
        render_trajectory_audit_page()


if __name__ == "__main__":
    main()