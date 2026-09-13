from __future__ import annotations

import platform
import sys

import streamlit as st


st.set_page_config(
    page_title="GameBus Data Analyzer",
    page_icon="📊",
    layout="wide",
)


def main() -> None:
    st.title("GameBus Data Analyzer")

    st.caption(
        "Extract and analyze participant data "
        "from GameBus campaigns."
    )

    st.success(
        "The application is installed and ready."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Python",
            platform.python_version(),
        )

    with col2:
        st.metric(
            "Streamlit",
            st.__version__,
        )

    st.divider()

    st.subheader("Analysis")

    st.write(
        "The campaign analysis workflow will be "
        "connected here next."
    )

    st.subheader("Data extraction")

    st.write(
        "The GameBus API extraction workflow will "
        "be connected here after the analysis "
        "workflow."
    )

    with st.sidebar:
        st.header("GameBus Data Analyzer")

        st.write(
            "User interface under development."
        )

        st.caption(
            f"Python {sys.version_info.major}."
            f"{sys.version_info.minor}"
        )


if __name__ == "__main__":
    main()