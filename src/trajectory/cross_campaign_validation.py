from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_CAMPAIGNS = [
    283,
    294,
    379,
    456,
    462,
]


def _read_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


def _core_quality_counts(
    audit_dir: Path,
) -> dict[str, int | None]:
    """
    Read participant quality states.

    If the participant CSV is unavailable,
    return NA-compatible values rather than
    fabricating counts.
    """

    path = (
            audit_dir
            / "data_quality.csv"
    )

    states = {
        "sufficient_for_core_trajectory": None,
        "core_trajectory_with_cautions": None,
        "no_observed_trajectory": None,
        "insufficient_for_core_trajectory": None,
    }

    if not path.exists():
        return states

    quality = pd.read_csv(
        path
    )

    if (
        quality.empty
        or "core_quality_state"
        not in quality.columns
    ):
        return states

    counts = (
        quality[
            "core_quality_state"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    for state in states:
        states[state] = int(
            counts.get(
                state,
                0,
            )
        )

    return states


def _campaign_row(
    campaign_id: int,
    audit_dir: Path,
) -> dict[str, Any]:

    summary_path = (
        audit_dir
        / "audit_summary.json"
    )

    quality_path = (
        audit_dir
        / "data_quality_summary.json"
    )

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing audit summary: "
            f"{summary_path}"
        )

    if not quality_path.exists():
        raise FileNotFoundError(
            f"Missing data-quality summary: "
            f"{quality_path}"
        )

    summary = _read_json(
        summary_path
    )

    quality = _read_json(
        quality_path
    )

    cohort = summary.get(
        "cohort",
        {},
    )

    events = summary.get(
        "events",
        {},
    )

    episodes = summary.get(
        "episodes",
        {},
    )

    gaps = summary.get(
        "inactivity_gaps",
        {},
    )

    threshold_14 = (
        gaps
        .get(
            "thresholds",
            {},
        )
        .get(
            "14",
            {},
        )
    )

    domain_mapping = summary.get(
        "domain_mapping",
        {},
    )

    patterns = summary.get(
        "candidate_patterns",
        {},
    )

    observation_window = quality.get(
        "observation_window",
        {},
    )

    streams = quality.get(
        "streams",
        {},
    )

    event_quality = quality.get(
        "event_quality",
        {},
    )

    quality_flags = quality.get(
        "campaign_quality_flags",
        [],
    )

    if not isinstance(
        quality_flags,
        list,
    ):
        quality_flags = []

    quality_counts = (
        _core_quality_counts(
            audit_dir
        )
    )

    return {
        "campaign": campaign_id,

        # -----------------------------------------
        # Cohort evidence
        # -----------------------------------------

        "cohort_size": (
            cohort.get(
                "participants"
            )
        ),

        "participants_with_any_observed_event": (
            cohort.get(
                "participants_with_any_observed_event"
            )
        ),

        "participants_without_observed_events": (
            cohort.get(
                "participants_without_observed_events"
            )
        ),

        "observed_participant_coverage": (
            cohort.get(
                "observed_participant_coverage"
            )
        ),

        "participants_with_explicit_engagement": (
            cohort.get(
                "participants_with_explicit_engagement"
            )
        ),

        "participants_without_explicit_engagement": (
            cohort.get(
                "participants_without_explicit_engagement"
            )
        ),

        "cohort_membership_evidence": (
            cohort.get(
                "cohort_membership_evidence"
            )
        ),

        # -----------------------------------------
        # Event / trajectory volume
        # -----------------------------------------

        "normalized_events": (
            events.get(
                "normalized_events"
            )
        ),

        "explicit_engagement_events": (
            events.get(
                "explicit_engagement_events"
            )
        ),

        "participation_episodes": (
            episodes.get(
                "participation_episodes"
            )
        ),

        # -----------------------------------------
        # 14-day inactivity sensitivity scale
        # -----------------------------------------

        "gaps_14d": (
            threshold_14.get(
                "gaps"
            )
        ),

        "gaps_14d_observed_reengagements": (
            threshold_14.get(
                "observed_reengagements"
            )
        ),

        "gaps_14d_right_censored": (
            threshold_14.get(
                "right_censored"
            )
        ),

        # -----------------------------------------
        # Candidate trajectory patterns
        # -----------------------------------------

        "candidate_pattern_rows": (
            patterns.get(
                "rows"
            )
        ),

        "participants_with_candidate_patterns": (
            patterns.get(
                "participants"
            )
        ),

        # -----------------------------------------
        # Mapping / event quality
        # -----------------------------------------

        "unmapped_domain_events": (
            domain_mapping.get(
                "unmapped_activity_events",
                event_quality.get(
                    "unmapped_domain_events"
                ),
            )
        ),

        "dedup_conflict_events": (
            event_quality.get(
                "dedup_conflict_events"
            )
        ),

        "events_missing_participant_id": (
            event_quality.get(
                "events_missing_participant_id"
            )
        ),

        "events_missing_timestamp": (
            event_quality.get(
                "events_missing_timestamp"
            )
        ),

        # -----------------------------------------
        # Observation window
        # -----------------------------------------

        "configured_start": (
            observation_window.get(
                "configured_start"
            )
        ),

        "effective_start": (
            observation_window.get(
                "effective_start"
            )
        ),

        "analysis_cutoff": (
            observation_window.get(
                "analysis_cutoff"
            )
        ),

        "start_source": (
            observation_window.get(
                "start_source"
            )
        ),

        "configured_start_adjusted": (
            "configured_start_adjusted"
            in quality_flags
        ),

        # -----------------------------------------
        # Participant core quality
        # -----------------------------------------

        "quality_sufficient": (
            quality_counts[
                "sufficient_for_core_trajectory"
            ]
        ),

        "quality_with_cautions": (
            quality_counts[
                "core_trajectory_with_cautions"
            ]
        ),

        "quality_no_observed_trajectory": (
            quality_counts[
                "no_observed_trajectory"
            ]
        ),

        "quality_insufficient": (
            quality_counts[
                "insufficient_for_core_trajectory"
            ]
        ),

        # -----------------------------------------
        # Optional streams
        # -----------------------------------------

        "activity_stream": (
            streams.get(
                "activity"
            )
        ),

        "navigation_stream": (
            streams.get(
                "navigation"
            )
        ),

        "garmin_stream": (
            streams.get(
                "garmin"
            )
        ),

        "nutrida_stream": (
            streams.get(
                "nutrida"
            )
        ),

        "credential_extraction": (
            streams.get(
                "credential_extraction"
            )
        ),

        "campaign_quality_flags": (
            json.dumps(
                quality_flags,
                ensure_ascii=False,
            )
        ),
    }


def build_cross_campaign_validation(
    campaign_ids: list[int],
    data_analysis_dir: Path,
) -> pd.DataFrame:

    rows = []

    for campaign_id in campaign_ids:

        audit_dir = (
            data_analysis_dir
            / f"trajectory_audit_{campaign_id}"
        )

        rows.append(
            _campaign_row(
                campaign_id,
                audit_dir,
            )
        )

    result = pd.DataFrame(
        rows
    )

    if (
        "observed_participant_coverage"
        in result.columns
    ):
        result[
            "observed_participant_coverage_pct"
        ] = (
            result[
                "observed_participant_coverage"
            ]
            * 100
        ).round(
            1
        )

    return result


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Build a cross-campaign validation "
            "table from trajectory audit outputs."
        )
    )

    parser.add_argument(
        "--campaigns",
        nargs="+",
        type=int,
        default=DEFAULT_CAMPAIGNS,
    )

    parser.add_argument(
        "--data-analysis-dir",
        default="data_analysis",
    )

    parser.add_argument(
        "--output",
        default=(
            "data_analysis/"
            "cross_campaign_validation.csv"
        ),
    )

    args = parser.parse_args()

    table = (
        build_cross_campaign_validation(
            campaign_ids=args.campaigns,
            data_analysis_dir=Path(
                args.data_analysis_dir
            ),
        )
    )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        "Cross-campaign validation"
    )
    print()

    display_columns = [
        "campaign",
        "cohort_size",
        "participants_with_any_observed_event",
        "observed_participant_coverage_pct",
        "participants_with_explicit_engagement",
        "normalized_events",
        "explicit_engagement_events",
        "participation_episodes",
        "gaps_14d",
        "gaps_14d_observed_reengagements",
        "gaps_14d_right_censored",
        "candidate_pattern_rows",
        "unmapped_domain_events",
        "quality_no_observed_trajectory",
        "configured_start_adjusted",
    ]

    print(
        table[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Written to: {output_path}"
    )


if __name__ == "__main__":
    main()