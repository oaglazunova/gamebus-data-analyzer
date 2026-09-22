from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.trajectory.cross_campaign_evaluation import (
    run_cross_campaign_dynamic_evaluation,
)


class TestCrossCampaignEvaluation(
    unittest.TestCase
):

    def _write_audit(
        self,
        root: Path,
        *,
        campaign_id: int,
        shift: float,
    ) -> Path:

        audit_dir = (
            root
            / f"campaign_{campaign_id}"
            / "audits"
            / "2026-09-19_120000"
        )

        results_dir = (
            audit_dir
            / "results"
        )

        results_dir.mkdir(
            parents=True
        )

        manifest = {
            "schema_version": 1,
            "source_type": (
                "uploaded_files"
            ),
            "campaign": {
                "id": str(
                    campaign_id
                ),
                "abbreviation": (
                    f"C{campaign_id}"
                ),
            },
        }

        (
            audit_dir
            / "run_manifest.json"
        ).write_text(
            json.dumps(
                manifest
            ),
            encoding="utf-8",
        )

        rows = []

        for participant in range(
            1,
            6,
        ):

            for gap in range(
                1,
                9,
            ):

                rows.append(
                    {
                        "participant_id": (
                            participant
                        ),

                        "date": (
                            pd.Timestamp(
                                "2026-01-01"
                            )
                            + pd.Timedelta(
                                days=gap
                            )
                        ).date(),

                        (
                            "days_since_last_explicit_"
                            "engagement"
                        ): gap,

                        (
                            "next_day_outcome_"
                            "available"
                        ): True,

                        (
                            "explicit_engagement_"
                            "next_day"
                        ): (
                            gap
                            <= (
                                2
                                + shift
                            )
                        ),

                        "active_days_28d": (
                            max(
                                0,
                                8 - gap
                            )
                        ),

                        "active_day_ratio_28d": (
                            max(
                                0.0,
                                1.0
                                - (
                                    gap
                                    / 10
                                )
                            )
                        ),

                        "reengagements_to_date": (
                            participant
                            % 2
                        ),

                        (
                            "explicit_domain_"
                            "diversity_28d"
                        ): (
                            2
                            if gap <= 4
                            else 1
                        ),

                        (
                            "explicit_tool_"
                            "diversity_28d"
                        ): 1,
                    }
                )

        pd.DataFrame(
            rows
        ).to_csv(
            results_dir
            / "participant_day_modeling.csv",
            index=False,
        )

        return audit_dir

    def test_evaluation_campaign_does_not_affect_training_preprocessing(
        self,
    ) -> None:

        with tempfile.TemporaryDirectory() as tmp:

            root = Path(
                tmp
            )

            development = self._write_audit(
                root,
                campaign_id=101,
                shift=0,
            )

            evaluation = self._write_audit(
                root,
                campaign_id=202,
                shift=1,
            )

            first = (
                run_cross_campaign_dynamic_evaluation(
                    development_audit_dirs=[
                        development
                    ],
                    evaluation_audit_dirs=[
                        evaluation
                    ],
                    output_dir=(
                        root
                        / "experiment_1"
                    ),
                )
            )

            first_means = (
                first[
                    "model"
                ].means
            )

            # Change evaluation predictors drastically.
            evaluation_csv = (
                evaluation
                / "results"
                / "participant_day_modeling.csv"
            )

            evaluation_frame = pd.read_csv(
                evaluation_csv
            )

            evaluation_frame[
                "active_days_28d"
            ] = 100000

            evaluation_frame.to_csv(
                evaluation_csv,
                index=False,
            )

            second = (
                run_cross_campaign_dynamic_evaluation(
                    development_audit_dirs=[
                        development
                    ],
                    evaluation_audit_dirs=[
                        evaluation
                    ],
                    output_dir=(
                        root
                        / "experiment_2"
                    ),
                )
            )

            self.assertEqual(
                first_means,
                second[
                    "model"
                ].means,
            )

    def test_same_audit_cannot_be_development_and_evaluation(
        self,
    ) -> None:

        with tempfile.TemporaryDirectory() as tmp:

            root = Path(
                tmp
            )

            audit = self._write_audit(
                root,
                campaign_id=101,
                shift=0,
            )

            with self.assertRaisesRegex(
                ValueError,
                "cannot be used",
            ):
                run_cross_campaign_dynamic_evaluation(
                    development_audit_dirs=[
                        audit
                    ],
                    evaluation_audit_dirs=[
                        audit
                    ],
                    output_dir=(
                        root
                        / "experiment"
                    ),
                )

    def test_outputs_are_written(
        self,
    ) -> None:

        with tempfile.TemporaryDirectory() as tmp:

            root = Path(
                tmp
            )

            development = self._write_audit(
                root,
                campaign_id=101,
                shift=0,
            )

            evaluation = self._write_audit(
                root,
                campaign_id=202,
                shift=1,
            )

            output = (
                root
                / "experiment"
            )

            result = (
                run_cross_campaign_dynamic_evaluation(
                    development_audit_dirs=[
                        development
                    ],
                    evaluation_audit_dirs=[
                        evaluation
                    ],
                    output_dir=output,
                )
            )

            for filename in (
                (
                    "dynamic_event_history_"
                    "model.json"
                ),
                (
                    "dynamic_event_history_"
                    "coefficients.csv"
                ),
                (
                    "dynamic_event_history_"
                    "predictions.csv"
                ),
                (
                    "dynamic_event_history_"
                    "evaluation.csv"
                ),
                (
                    "cross_campaign_"
                    "experiment.json"
                ),
            ):

                self.assertTrue(
                    (
                        output
                        / filename
                    ).is_file()
                )

            evaluation_table = (
                result[
                    "evaluation"
                ]
            )

            self.assertEqual(
                set(
                    evaluation_table[
                        "evaluation_scope"
                    ]
                ),
                {
                    "campaign",
                    "pooled_evaluation",
                },
            )

            probabilities = result[
                "predictions"
            ][
                (
                    "predicted_next_day_"
                    "reengagement_probability"
                )
            ]

            self.assertTrue(
                probabilities.between(
                    0,
                    1,
                ).all()
            )


if __name__ == "__main__":
    unittest.main()