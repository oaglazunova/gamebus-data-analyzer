from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from src.analysis import activity_plots
from src.analysis.common import (
    OUTPUT_VISUALIZATIONS_DIR,
    ensure_output_dirs,
    use_analysis_output_directory,
)
from src.analysis.reporting import (
    generate_analysis_report,
)



class TestRuntimeAnalysisOutput(
    unittest.TestCase
):

    def test_output_directory_changes_temporarily(
        self,
    ) -> None:
        original = os.fspath(
            OUTPUT_VISUALIZATIONS_DIR
        )

        with tempfile.TemporaryDirectory() as tmp:
            target = (
                Path(
                    tmp
                )
                / "data_analysis"
            ).resolve()

            with use_analysis_output_directory(
                target
            ):
                self.assertEqual(
                    os.fspath(
                        OUTPUT_VISUALIZATIONS_DIR
                    ),
                    os.fspath(
                        target
                    ),
                )

                # activity_plots imported the same object,
                # so it must see the changed path too.
                self.assertEqual(
                    os.fspath(
                        activity_plots
                        .OUTPUT_VISUALIZATIONS_DIR
                    ),
                    os.fspath(
                        target
                    ),
                )

            self.assertEqual(
                os.fspath(
                    OUTPUT_VISUALIZATIONS_DIR
                ),
                original,
            )

    def test_ensure_output_dirs_uses_runtime_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = (
                Path(
                    tmp
                )
                / "data_analysis"
            )

            with use_analysis_output_directory(
                target
            ):
                ensure_output_dirs()

                self.assertTrue(
                    target.is_dir()
                )

                self.assertTrue(
                    (
                        target
                        / "statistics"
                    ).is_dir()
                )

    def test_analysis_report_uses_runtime_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = (
                Path(tmp)
                / "data_analysis"
            )

            with use_analysis_output_directory(
                target
            ):
                report_path = (
                    generate_analysis_report(
                        csv_data={},
                        json_data={},
                        activities=None,
                        campaign_metrics=None,
                        dropout_metrics=None,
                        joining_metrics=None,
                    )
                )

            self.assertEqual(
                Path(
                    report_path
                ).resolve(),
                (
                    target
                    / "analysis_report.txt"
                ).resolve(),
            )

            self.assertTrue(
                (
                    target
                    / "analysis_report.txt"
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()