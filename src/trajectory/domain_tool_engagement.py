from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

from src.trajectory.common import (
    TrajectoryAuditConfig,
    load_campaign_desc,
)

from src.trajectory.engagement import (
    behavioral_sensor_mask,
    explicit_engagement_mask,
)

from src.trajectory.event_normalization import (
    build_normalized_events,
)


OUTPUT_COLUMNS = [
    "event_id",
    "participant_id",
    "occurred_at",
    "date",

    "event_channel",

    "tool",
    "provider",
    "event_type",

    "points",

    "challenge_ids",

    "domain",
    "domain_mapping_source",

    "domain_membership_count",
    "event_weight",
    "points_equal_share",
]


# ---------------------------------------------------------
# Controlled vocabulary
# ---------------------------------------------------------
#
# These aliases are used only when translating explicit
# campaign-configuration text into three common domains.
#
# We do NOT classify an arbitrary activity description
# based on these words.
#
# Primary mapping still comes from campaign structure:
#
# challenge -> label -> category
#
DOMAIN_KEYWORDS = {
    "nutrition": [
        "ernährung",
        "nutrition",
        "nutrizione",
        "alimentazione",
        "food",
        "cibo",
        "essen",
        "kochen",
        "cooking",
        "meal",
    ],

    "physical_activity": [
        "körperliche aktivität",
        "physische aktivität",
        "physical activity",
        "attività fisica",
        "bewegung",
        "movimento",
        "walking",
        "camminare",
        "exercise",
        "fitness",
        "sport",
        "schritt",
        "steps",
        "radel",
    ],

    "mental_wellbeing": [
        "achtsamkeit",
        "mindfulness",
        "mental wellbeing",
        "mental well-being",
        "benessere mentale",
        "psychisches wohlbefinden",
        "psyche",
        "sleep",
        "schlaf",
        "sonno",
        "equilibrio mentale",
        "wellness",
        "gelassenheit",
        "serenity",
    ],
}


def _canonical_domain(
    value: Any,
) -> str | None:
    """
    Translate an explicit campaign configuration label
    into our common three-domain vocabulary.

    Returns:
        nutrition
        physical_activity
        mental_wellbeing

    or None when the configuration text is not
    sufficiently informative.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    text = str(
        value
    ).strip().lower()

    if not text:
        return None

    for domain, keywords in (
        DOMAIN_KEYWORDS.items()
    ):

        for keyword in keywords:

            if keyword in text:
                return domain

    return None


def _extract_integer_ids(
    value: Any,
) -> List[int]:
    """
    Extract IDs from campaign configuration cells.

    Handles values such as:

        513
        513.0
        "513"
        "513, 514"
        "3472, 4388, 4392"
    """

    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    matches = re.findall(
        r"\d+",
        str(value),
    )

    result: List[int] = []

    for match in matches:

        number = int(match)

        if number not in result:
            result.append(number)

    return result


def _parse_json_ids(
    value: Any,
) -> List[int]:
    """
    Parse challenge_ids from normalized_events.csv.

    The normalized value looks like:

        [14375]

    or:

        [14375, 15001]
    """

    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    if isinstance(
        value,
        list,
    ):
        raw = value

    else:

        try:
            raw = json.loads(
                str(value)
            )

        except (
            json.JSONDecodeError,
            TypeError,
        ):
            return []

    if not isinstance(
        raw,
        list,
    ):
        return []

    result: List[int] = []

    for item in raw:

        try:
            number = int(
                float(item)
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        if number not in result:
            result.append(
                number
            )

    return result


def _tool_name(
    provider: Any,
) -> str:
    """
    Normalize known providers into stable tool names.
    """

    if provider is None:
        return "unknown"

    try:
        if pd.isna(provider):
            return "unknown"
    except Exception:
        pass

    text = (
        str(provider)
        .strip()
        .lower()
    )

    if "garmin" in text:
        return "garmin"

    if "nutrida" in text:
        return "nutrida"

    if "gamebus" in text:
        return "gamebus_studio"

    # Generic fallback for future providers.
    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    ).strip("_")

    return (
        text
        if text
        else "unknown"
    )


def _build_challenge_domain_map(
    config: TrajectoryAuditConfig,
) -> Dict[
    int,
    Tuple[
        Set[str],
        Set[str],
    ],
]:
    """
    Build:

        challenge_id
            ->
        (
            {domains},
            {mapping_sources}
        )

    Mapping priority:

        1. challenge label -> category
        2. challenge visualization
        3. challenge reward
        4. challenge name

    We stop at the first level that produces a
    meaningful domain.

    This gives us provenance for every mapping.
    """

    desc, _ = load_campaign_desc(
        config.campaign_desc_path
    )

    challenges = desc.get(
        "challenges",
        pd.DataFrame(),
    )

    if challenges.empty:
        return {}

    categories = desc.get(
        "categories",
        pd.DataFrame(),
    )

    labels = desc.get(
        "labels",
        pd.DataFrame(),
    )

    visualizations = desc.get(
        "visualizations",
        pd.DataFrame(),
    )

    challenge_rewards = desc.get(
        "challengerewards",
        pd.DataFrame(),
    )

    # -------------------------------------------------
    # Category ID -> canonical domain
    # -------------------------------------------------

    category_domains: Dict[
        int,
        str,
    ] = {}

    if (
        not categories.empty
        and "id" in categories.columns
        and "name" in categories.columns
    ):

        for _, row in (
            categories.iterrows()
        ):

            try:
                category_id = int(
                    row["id"]
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            domain = _canonical_domain(
                row["name"]
            )

            if domain is not None:

                category_domains[
                    category_id
                ] = domain

    # -------------------------------------------------
    # Label ID -> category ID
    # -------------------------------------------------

    label_categories: Dict[
        int,
        int,
    ] = {}

    if (
        not labels.empty
        and "id" in labels.columns
        and "category" in labels.columns
    ):

        for _, row in (
            labels.iterrows()
        ):

            try:
                label_id = int(
                    row["id"]
                )

                category_id = int(
                    row["category"]
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

            label_categories[
                label_id
            ] = category_id

    # -------------------------------------------------
    # Visualization ID -> canonical domain
    # -------------------------------------------------

    visualization_domains: Dict[
        int,
        str,
    ] = {}

    if (
        not visualizations.empty
        and "id" in visualizations.columns
    ):

        for _, row in (
            visualizations.iterrows()
        ):

            try:
                visualization_id = int(
                    row["id"]
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

            possible_text = []

            for column in (
                "label",
                "description",
            ):

                if column in (
                    visualizations.columns
                ):

                    possible_text.append(
                        row.get(column)
                    )

            domain = None

            for value in possible_text:

                domain = (
                    _canonical_domain(
                        value
                    )
                )

                if domain is not None:
                    break

            if domain is not None:

                visualization_domains[
                    visualization_id
                ] = domain

    # -------------------------------------------------
    # Challenge ID -> reward names
    # -------------------------------------------------

    rewards_by_challenge: Dict[
        int,
        List[str],
    ] = {}

    if (
        not challenge_rewards.empty
        and "challenge"
        in challenge_rewards.columns
        and "name"
        in challenge_rewards.columns
    ):

        for _, row in (
            challenge_rewards.iterrows()
        ):

            try:
                challenge_id = int(
                    row["challenge"]
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

            reward_name = row.get(
                "name"
            )

            if pd.isna(
                reward_name
            ):
                continue

            rewards_by_challenge.setdefault(
                challenge_id,
                [],
            ).append(
                str(reward_name)
            )

    # -------------------------------------------------
    # Build final mapping
    # -------------------------------------------------

    result: Dict[
        int,
        Tuple[
            Set[str],
            Set[str],
        ],
    ] = {}

    for _, challenge in (
        challenges.iterrows()
    ):

        try:
            challenge_id = int(
                challenge["id"]
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        domains: Set[str] = set()
        sources: Set[str] = set()

        # =================================================
        # 1. LABEL -> CATEGORY
        # =================================================

        if "labels" in challenges.columns:

            label_ids = (
                _extract_integer_ids(
                    challenge.get(
                        "labels"
                    )
                )
            )

            for label_id in label_ids:

                category_id = (
                    label_categories.get(
                        label_id
                    )
                )

                if category_id is None:
                    continue

                domain = (
                    category_domains.get(
                        category_id
                    )
                )

                if domain is not None:

                    domains.add(
                        domain
                    )

            if domains:

                sources.add(
                    "challenge_category"
                )

        # =================================================
        # 2. VISUALIZATION
        # =================================================

        if (
            not domains
            and "visualizations"
            in challenges.columns
        ):

            visualization_ids = (
                _extract_integer_ids(
                    challenge.get(
                        "visualizations"
                    )
                )
            )

            for visualization_id in (
                visualization_ids
            ):

                domain = (
                    visualization_domains.get(
                        visualization_id
                    )
                )

                if domain is not None:

                    domains.add(
                        domain
                    )

            if domains:

                sources.add(
                    "challenge_visualization"
                )

        # =================================================
        # 3. CHALLENGE REWARD
        # =================================================

        if not domains:

            reward_names = (
                rewards_by_challenge.get(
                    challenge_id,
                    [],
                )
            )

            for reward_name in (
                reward_names
            ):

                domain = (
                    _canonical_domain(
                        reward_name
                    )
                )

                if domain is not None:

                    domains.add(
                        domain
                    )

            if domains:

                sources.add(
                    "challenge_reward"
                )

        # =================================================
        # 4. CHALLENGE NAME
        # =================================================

        if not domains:

            domain = _canonical_domain(
                challenge.get(
                    "name"
                )
            )

            if domain is not None:

                domains.add(
                    domain
                )

                sources.add(
                    "challenge_name"
                )

        result[
            challenge_id
        ] = (
            domains,
            sources,
        )

    return result


def _domains_for_event(
    challenge_ids: List[int],
    challenge_map: Dict[
        int,
        Tuple[
            Set[str],
            Set[str],
        ],
    ],
) -> Tuple[
    List[str],
    List[str],
]:
    """
    Resolve all domains touched by one activity.

    One activity may reward challenges in more
    than one domain.
    """

    domains: Set[str] = set()
    sources: Set[str] = set()

    for challenge_id in (
        challenge_ids
    ):

        mapping = (
            challenge_map.get(
                challenge_id
            )
        )

        if mapping is None:
            continue

        challenge_domains, (
            challenge_sources
        ) = mapping

        domains.update(
            challenge_domains
        )

        sources.update(
            challenge_sources
        )

    return (
        sorted(domains),
        sorted(sources),
    )


def build_domain_tool_engagement(
    config: TrajectoryAuditConfig,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Enrich normalized activity events with:

        behavioral domain
        tool/provider
        engagement channel

    Output is LONG FORMAT.

    A multi-domain event appears once per domain.

    Example:

        activity A
            nutrition
            physical_activity

    becomes two rows, each with:

        event_weight = 0.5

    so event_weight remains additive.
    """

    if events.empty:

        return pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

    prepared = events.copy()

    prepared["occurred_at"] = (
        pd.to_datetime(
            prepared["occurred_at"],
            utc=True,
            errors="coerce",
        )
    )

    prepared["participant_id"] = (
        pd.to_numeric(
            prepared["participant_id"],
            errors="coerce",
        )
        .astype("Int64")
    )

    prepared["points"] = (
        pd.to_numeric(
            prepared["points"],
            errors="coerce",
        )
        .fillna(0)
    )

    event_kind = (
        prepared["event_kind"]
        .astype("string")
        .fillna("")
        .str.lower()
    )

    # Domain attribution only makes sense for
    # activities linked to campaign challenges.
    activities = prepared.loc[
        event_kind.eq("activity")
    ].copy()

    if activities.empty:

        return pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

    # -------------------------------------------------
    # Engagement channel
    # -------------------------------------------------

    explicit = (
        explicit_engagement_mask(
            activities
        )
    )

    behavioral = (
        behavioral_sensor_mask(
            activities
        )
    )

    activities[
        "event_channel"
    ] = "other_activity"

    activities.loc[
        explicit,
        "event_channel",
    ] = "explicit_engagement"

    activities.loc[
        behavioral,
        "event_channel",
    ] = "behavioral_sensor"

    # -------------------------------------------------
    # Challenge -> domain map
    # -------------------------------------------------

    challenge_map = (
        _build_challenge_domain_map(
            config
        )
    )

    rows: List[
        Dict[str, Any]
    ] = []

    for _, event in (
        activities.iterrows()
    ):

        challenge_ids = (
            _parse_json_ids(
                event.get(
                    "challenge_ids"
                )
            )
        )

        domains, sources = (
            _domains_for_event(
                challenge_ids,
                challenge_map,
            )
        )

        # We retain unmapped events rather than
        # silently discarding them.
        if not domains:

            domains = [
                "unmapped"
            ]

            sources = [
                "unmapped"
            ]

        membership_count = len(
            domains
        )

        event_weight = (
            1.0
            / membership_count
        )

        points = float(
            event.get(
                "points",
                0,
            )
            or 0
        )

        points_equal_share = (
            points
            / membership_count
        )

        provider = event.get(
            "provider"
        )

        for domain in domains:

            rows.append(
                {
                    "event_id": (
                        event["event_id"]
                    ),

                    "participant_id": (
                        event[
                            "participant_id"
                        ]
                    ),

                    "occurred_at": (
                        event[
                            "occurred_at"
                        ]
                    ),

                    "date": (
                        event[
                            "occurred_at"
                        ].date()
                        if pd.notna(
                            event[
                                "occurred_at"
                            ]
                        )
                        else pd.NaT
                    ),

                    "event_channel": (
                        event[
                            "event_channel"
                        ]
                    ),

                    "tool": (
                        _tool_name(
                            provider
                        )
                    ),

                    "provider": (
                        provider
                    ),

                    "event_type": (
                        event.get(
                            "event_type"
                        )
                    ),

                    "points": (
                        points
                    ),

                    "challenge_ids": (
                        json.dumps(
                            challenge_ids
                        )
                    ),

                    "domain": (
                        domain
                    ),

                    "domain_mapping_source": (
                        "|".join(
                            sources
                        )
                    ),

                    "domain_membership_count": (
                        membership_count
                    ),

                    "event_weight": (
                        event_weight
                    ),

                    "points_equal_share": (
                        points_equal_share
                    ),
                }
            )

    if not rows:

        return pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

    result = pd.DataFrame(
        rows
    )

    result = result.sort_values(
        [
            "participant_id",
            "occurred_at",
            "event_id",
            "domain",
        ]
    ).reset_index(
        drop=True
    )

    for column in OUTPUT_COLUMNS:

        if column not in result.columns:
            result[column] = pd.NA

    return result[
        OUTPUT_COLUMNS
    ]


def run_domain_tool_engagement(
    config: TrajectoryAuditConfig | None = None,
) -> pd.DataFrame:
    """
    Build and save domain_tool_engagement.csv.
    """

    config = (
        config
        or TrajectoryAuditConfig()
    )

    events = build_normalized_events(
        config
    )

    result = (
        build_domain_tool_engagement(
            config,
            events,
        )
    )

    os.makedirs(
        config.output_dir,
        exist_ok=True,
    )

    output_path = os.path.join(
        config.output_dir,
        "domain_tool_engagement.csv",
    )

    result.to_csv(
        output_path,
        index=False,
    )

    return result


if __name__ == "__main__":

    result = (
        run_domain_tool_engagement()
    )

    print(
        "Domain/tool engagement "
        "written successfully."
    )

    print()

    print(
        "Domain/tool rows:",
        len(result),
    )

    if not result.empty:

        unique_events = (
            result[
                "event_id"
            ]
            .nunique()
        )

        multi_domain_events = (
            result.loc[
                result[
                    "domain_membership_count"
                ]
                > 1,
                "event_id",
            ]
            .nunique()
        )

        unmapped_events = (
            result.loc[
                result["domain"]
                == "unmapped",
                "event_id",
            ]
            .nunique()
        )

        print(
            "Unique activity events:",
            unique_events,
        )

        print(
            "Multi-domain activity events:",
            multi_domain_events,
        )

        print(
            "Unmapped activity events:",
            unmapped_events,
        )

        print()

        print(
            "Unique events by channel:"
        )

        channel_counts = (
            result[
                [
                    "event_id",
                    "event_channel",
                ]
            ]
            .drop_duplicates()
            ["event_channel"]
            .value_counts()
        )

        for (
            channel,
            count,
        ) in channel_counts.items():

            print(
                f"  {channel}: {count}"
            )

        print()

        print(
            "Domain memberships:"
        )

        domain_counts = (
            result[
                "domain"
            ]
            .value_counts()
        )

        for (
            domain,
            count,
        ) in domain_counts.items():

            print(
                f"  {domain}: {count}"
            )

        print()

        print(
            "Unique events by tool:"
        )

        tool_counts = (
            result[
                [
                    "event_id",
                    "tool",
                ]
            ]
            .drop_duplicates()
            ["tool"]
            .value_counts()
        )

        for (
            tool,
            count,
        ) in tool_counts.items():

            print(
                f"  {tool}: {count}"
            )

        print()

        print(
            "Domain mapping sources:"
        )

        source_counts = (
            result[
                "domain_mapping_source"
            ]
            .value_counts()
        )

        for (
            source,
            count,
        ) in source_counts.items():

            print(
                f"  {source}: {count}"
            )