from __future__ import annotations

import streamlit as st

from src.ui.analyze_data import (
    render_analyze_data_page,
)
from src.ui.get_data import (
    render_get_data_page,
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
            "Analyze existing data",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.divider()

    if action == "Get data":
        render_get_data_page()

    else:
        render_analyze_data_page()


if __name__ == "__main__":
    main()