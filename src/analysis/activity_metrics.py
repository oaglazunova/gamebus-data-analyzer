from __future__ import annotations

from typing import Dict, Optional, Any, Union
import os
import re

import pandas as pd

from config.paths import RAW_DATA_DIR, USERS_FILE_PATH
from src.analysis.loaders import extract_points, extract_detailed_rewards
from src.analysis.common import WEEKDAY_ORDER, HOUR_BUCKET_LABELS, HOUR_BUCKET_BINS, logger



PLAYER_ID_COLUMNS = (
    "pid",
    "playerId",
    "player_id",
    "playerID",
    "X_PLAYER_ID",
)

DROPOUT_INACTIVITY_DAYS = 14

USER_EMAIL_MAPPING_FILE = os.path.join(RAW_DATA_DIR, "user_email_mapping.txt")


def _normalize_user_id(value: Any) -> Optional[str]:
    """
    Convert a player identifier to a stable string representation.

    This prevents mismatches such as 454, 454.0, and "454" being treated
    as three different users.
    """
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if value is None:
        return None

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        if pd.isna(value):
            return None
        if value.is_integer():
            return str(int(value))
        return str(value).strip()

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None

    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]

    return text


def _normalize_email(value: Any) -> Optional[str]:
    """
    Normalize an email address for comparison between users.xlsx and
    user_email_mapping.txt.
    """
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if value is None:
        return None

    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "null"}:
        return None

    return text


def _find_column_case_insensitive(df: pd.DataFrame, wanted_name: str) -> Optional[str]:
    wanted = wanted_name.strip().lower()
    for col in df.columns:
        if isinstance(col, str) and col.strip().lower() == wanted:
            return col
    return None


def _find_player_id_column(df: pd.DataFrame) -> Optional[str]:
    """
    Return the first known player-id column in a table, if present.

    Intentionally does not use generic columns such as "id" or "userId",
    because those may refer to activities, accounts, tasks, or other entities.
    """
    for candidate in PLAYER_ID_COLUMNS:
        if candidate in df.columns:
            return candidate

    lowered = {
        col.strip().lower(): col
        for col in df.columns
        if isinstance(col, str)
    }
    for candidate in PLAYER_ID_COLUMNS:
        match = lowered.get(candidate.lower())
        if match is not None:
            return match

    return None


def read_users_xlsx_emails(users_file: str = USERS_FILE_PATH) -> set[str]:
    """
    Read the authoritative participant roster from config/users.xlsx.

    users.xlsx is the source of truth for the analysis cohort. It is expected
    to contain emails and passwords, but not player IDs.
    """
    if not os.path.exists(users_file):
        raise FileNotFoundError(f"users.xlsx not found: {users_file}")

    users_df = pd.read_excel(users_file)
    email_col = _find_column_case_insensitive(users_df, "email")
    if email_col is None:
        raise ValueError(f"users.xlsx must contain an 'email' column: {users_file}")

    emails = {
        email
        for email in users_df[email_col].map(_normalize_email)
        if email is not None
    }

    if not emails:
        raise ValueError(f"No valid participant emails found in users.xlsx: {users_file}")

    return emails



def ensure_user_email_mapping_exists(
    mapping_file: Optional[str] = None,
    *,
    force: bool = False,
) -> str:
    """
    Ensure data_raw/user_email_mapping.txt exists before analysis.

    If the mapping file is missing or empty, generate it automatically by
    logging in users from config/users.xlsx and resolving their GameBus
    player_id values.

    If force=True, regenerate the file even if it already exists. This is used
    once when the existing mapping does not cover all emails in users.xlsx.
    """
    if mapping_file is None:
        mapping_file = USER_EMAIL_MAPPING_FILE

    mapping_exists = os.path.exists(mapping_file) and os.path.getsize(mapping_file) > 0

    if mapping_exists and not force:
        return mapping_file

    if not os.path.exists(USERS_FILE_PATH):
        raise FileNotFoundError(
            f"Cannot generate user-email mapping because users.xlsx was not found: "
            f"{USERS_FILE_PATH}"
        )

    logger.info(
        "Generating user-email mapping file for analysis: "
        f"{mapping_file}"
    )

    try:
        from src.scripts.create_user_email_mapping import main as build_user_email_mapping
    except Exception as e:
        raise RuntimeError(
            "Could not import src.scripts.create_user_email_mapping. "
            "Cannot generate user-email mapping automatically."
        ) from e

    try:
        build_user_email_mapping()
    except Exception as e:
        raise RuntimeError(
            "Failed to generate user_email_mapping.txt automatically."
        ) from e

    if not os.path.exists(mapping_file) or os.path.getsize(mapping_file) == 0:
        raise FileNotFoundError(
            "user_email_mapping.txt is required for participant-filtered analysis, "
            "but automatic generation did not produce a valid file. "
            f"Expected file: {mapping_file}"
        )

    return mapping_file




def read_user_email_mapping(mapping_file: Optional[str] = None) -> dict[str, str]:
    """
    Read data_raw/user_email_mapping.txt and return {normalized_email: player_id}.

    Expected line format generated by src.scripts.create_user_email_mapping:
        player_id=454 user_id=123: user@example.org

    If the mapping file is missing, it is generated automatically.
    """
    mapping_file = ensure_user_email_mapping_exists(mapping_file)

    email_to_pid: dict[str, str] = {}
    line_re = re.compile(r"player_id\s*=\s*([^\s:]+).*?:\s*(\S+@\S+)", re.IGNORECASE)

    with open(mapping_file, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            match = line_re.search(line)
            if not match:
                logger.warning(f"Could not parse user-email mapping line: {line}")
                continue

            pid = _normalize_user_id(match.group(1))
            email = _normalize_email(match.group(2))

            if pid is not None and email is not None:
                email_to_pid[email] = pid

    if not email_to_pid:
        raise ValueError(f"No valid entries found in user-email mapping: {mapping_file}")

    return email_to_pid


def resolve_analysis_user_ids(
    users_file: str = USERS_FILE_PATH,
    mapping_file: Optional[str] = None,
) -> set[str]:
    """
    Resolve the exact player-id cohort to analyze.

    Logic:
    1. Read participant emails from config/users.xlsx.
    2. Ensure data_raw/user_email_mapping.txt exists.
    3. Read player_id <-> email mappings.
    4. Keep only player IDs whose email is present in users.xlsx.
    5. If the mapping is stale/incomplete, regenerate it once and retry.

    This prevents campaign/test accounts that appear in campaign_data.zip from
    being included in statistics.
    """
    roster_emails = read_users_xlsx_emails(users_file)

    resolved_mapping_file = ensure_user_email_mapping_exists(mapping_file)
    email_to_pid = read_user_email_mapping(resolved_mapping_file)

    missing_emails = sorted(email for email in roster_emails if email not in email_to_pid)

    if missing_emails:
        logger.warning(
            "Existing user_email_mapping.txt does not cover all emails in users.xlsx. "
            "Regenerating mapping once before failing."
        )

        resolved_mapping_file = ensure_user_email_mapping_exists(
            resolved_mapping_file,
            force=True,
        )
        email_to_pid = read_user_email_mapping(resolved_mapping_file)
        missing_emails = sorted(email for email in roster_emails if email not in email_to_pid)

    if missing_emails:
        preview = ", ".join(missing_emails[:10])
        suffix = "" if len(missing_emails) <= 10 else f" ... and {len(missing_emails) - 10} more"

        raise ValueError(
            "Could not resolve player_id for every email in users.xlsx. "
            f"Missing {len(missing_emails)} email(s): {preview}{suffix}. "
            "Analysis stopped to avoid including test accounts."
        )

    participant_ids = {email_to_pid[email] for email in roster_emails}

    if not participant_ids:
        raise ValueError("No participant player IDs could be resolved from users.xlsx")

    return participant_ids


def filter_dataframe_to_user_ids(df: pd.DataFrame, participant_ids: set[str]) -> pd.DataFrame:
    """
    Filter one DataFrame to the authoritative participant cohort if it has a
    player-id column. Tables without a player-id column are returned unchanged.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df

    player_col = _find_player_id_column(df)
    if player_col is None:
        return df.copy()

    out = df.copy()
    normalized_ids = out[player_col].map(_normalize_user_id)
    out = out[normalized_ids.isin(participant_ids)].copy()
    out[player_col] = out[player_col].map(_normalize_user_id)
    return out


def filter_tabular_data_to_user_ids(
    data: Dict[str, pd.DataFrame],
    participant_ids: set[str],
) -> Dict[str, pd.DataFrame]:
    """
    Filter all loaded Excel/CSV tables to the authoritative participant cohort.

    This affects activities, aggregation, navigation events, notification events,
    sensor events, and any other table with a known player-id column.
    """
    return {
        key: filter_dataframe_to_user_ids(df, participant_ids)
        if isinstance(df, pd.DataFrame)
        else df
        for key, df in data.items()
    }


def filter_json_data_to_user_ids(
    data: Dict[str, Union[pd.DataFrame, Dict]],
    participant_ids: set[str],
) -> Dict[str, Union[pd.DataFrame, Dict]]:
    """
    Filter loaded JSON-derived data to the authoritative participant cohort.

    Handles:
    - DataFrames with known player-id columns, e.g. X_PLAYER_ID
    - per-user dictionary payloads keyed as player_<pid>_* or user_<pid>_*
    - dictionary payloads that contain nested DataFrames

    Other dictionary payloads are kept unchanged.
    """
    normalized_participant_ids = {
        normalized
        for normalized in (_normalize_user_id(pid) for pid in participant_ids)
        if normalized is not None
    }

    filtered: Dict[str, Union[pd.DataFrame, Dict]] = {}

    for key, value in data.items():
        key_str = str(key)

        # Case 1: regular DataFrame payload.
        if isinstance(value, pd.DataFrame):
            filtered[key] = filter_dataframe_to_user_ids(value, normalized_participant_ids)
            continue

        # Case 2: per-user JSON payload stored under names such as:
        # player_454_all_data, user_454_all_data, player_454_day_aggregate, etc.
        match = re.match(r"^(?:player|user)_(\d+)(?:_|$)", key_str, flags=re.IGNORECASE)
        if match:
            pid = _normalize_user_id(match.group(1))
            if pid in normalized_participant_ids:
                filtered[key] = value
            continue

        # Case 3: dictionary containing nested DataFrames.
        if isinstance(value, dict):
            nested = {}
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, pd.DataFrame):
                    nested[nested_key] = filter_dataframe_to_user_ids(
                        nested_value,
                        normalized_participant_ids,
                    )
                else:
                    nested[nested_key] = nested_value
            filtered[key] = nested
            continue

        filtered[key] = value

    return filtered




GAMEBUS_ACTIVE_TYPES = [
    "DRINKING_DIARY",
    "GENERAL_ACTIVITY",
    "NUTRITION_DIARY",
    "PHYSICAL_ACTIVITY",
    "WALK",
    "BIKE",
]

GAMEBUS_METRIC_TYPES = GAMEBUS_ACTIVE_TYPES + ["DAY_AGGREGATE"]

GAMEBUS_USAGE_TIME_TYPES = GAMEBUS_ACTIVE_TYPES  # DAY_AGGREGATE excluded

NUTRIDA_TYPES = [
    "PLAN_MEAL",
    "NUTRITION_DIARY_VID",
]

COMBINED_ACTIVE_TYPES = GAMEBUS_ACTIVE_TYPES + NUTRIDA_TYPES
COMBINED_METRIC_TYPES = GAMEBUS_METRIC_TYPES + NUTRIDA_TYPES
COMBINED_USAGE_TIME_TYPES = GAMEBUS_USAGE_TIME_TYPES + NUTRIDA_TYPES


APP_SCOPES = {
    "gamebus": {
        "label": "GameBus",
        "active_types": GAMEBUS_ACTIVE_TYPES,
        "metric_types": GAMEBUS_METRIC_TYPES,
        "usage_time_types": GAMEBUS_USAGE_TIME_TYPES,
    },
    "nutrida": {
        "label": "Nutrida",
        "active_types": NUTRIDA_TYPES,
        "metric_types": NUTRIDA_TYPES,
        "usage_time_types": NUTRIDA_TYPES,
    },
    "combined": {
        "label": "Combined",
        "active_types": COMBINED_ACTIVE_TYPES,
        "metric_types": COMBINED_METRIC_TYPES,
        "usage_time_types": COMBINED_USAGE_TIME_TYPES,
    },
}


def build_activities_frame(csv_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Validate and return a copy of the activities sheet.
    Raises ValueError if the required input is missing or invalid.
    """
    if "activities" not in csv_data:
        raise ValueError("Activities data not found in the provided data (missing key 'activities')")

    activities = csv_data["activities"].copy()
    if activities.empty:
        raise ValueError("Activities DataFrame is empty")

    required_columns = ["createdAt", "pid", "type", "rewardedParticipations"]
    missing_columns = [c for c in required_columns if c not in activities.columns]
    if missing_columns:
        raise ValueError(f"Activities DataFrame missing required columns: {missing_columns}")

    return activities


def normalize_activity_columns(activities: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize timestamps and derive commonly used activity columns.
    """
    activities = activities.copy()

    activities["pid"] = activities["pid"].map(_normalize_user_id)
    activities = activities[activities["pid"].notna()].copy()

    activities["createdAt"] = pd.to_datetime(activities["createdAt"], errors="coerce", utc=True)
    activities["date"] = activities["createdAt"].dt.floor("D")
    activities["hour"] = activities["createdAt"].dt.hour

    activities["points"] = activities["rewardedParticipations"].apply(extract_points)
    activities["detailed_rewards"] = activities["rewardedParticipations"].apply(extract_detailed_rewards)
    activities["num_rewards"] = activities["detailed_rewards"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )

    return activities


def get_campaign_waves(csv_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Return normalized campaign waves from desc_waves, if available.
    """
    waves = csv_data.get("desc_waves")
    if not isinstance(waves, pd.DataFrame) or waves.empty:
        return pd.DataFrame()

    required = {"start", "end"}
    if not required.issubset(waves.columns):
        return pd.DataFrame()

    out = waves.copy()
    out["start"] = pd.to_datetime(out["start"], errors="coerce", utc=True)
    out["end"] = pd.to_datetime(out["end"], errors="coerce", utc=True)
    out = out.dropna(subset=["start", "end"]).sort_values("start").reset_index(drop=True)
    return out


def _get_campaign_bounds(
    activities: pd.DataFrame,
    csv_data: Dict[str, pd.DataFrame],
) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Return (campaign_start, campaign_end), preferring desc_waves when available.
    """
    waves = get_campaign_waves(csv_data)
    if not waves.empty:
        return waves["start"].min(), waves["end"].max()

    return activities["createdAt"].min(), activities["createdAt"].max()


def assign_campaign_wave(activities: pd.DataFrame, csv_data: Dict[str, pd.DataFrame]) -> pd.Series:
    """
    Assign each activity to a campaign wave.
    Uses desc_waves when available; otherwise falls back to 7-day buckets
    from the earliest activity date.
    """
    waves = get_campaign_waves(csv_data)

    if waves.empty:
        start = activities["date"].min()
        if pd.isna(start):
            return pd.Series([pd.NA] * len(activities), index=activities.index)
        return ((activities["date"] - start).dt.days // 7) + 1

    result = pd.Series([pd.NA] * len(activities), index=activities.index, dtype="object")

    for i, row in waves.iterrows():
        mask = (
            (activities["createdAt"] >= row["start"])
            & (activities["createdAt"] <= row["end"])
        )
        result.loc[mask] = i + 1

    return result


def _extract_campaign_identity(csv_data: Dict[str, pd.DataFrame]) -> tuple[str, str]:
    """
    Extract campaign name and abbreviation from desc_* sheets if available.
    """
    campaign_name = "GameBus Campaign"
    campaign_abbr = ""

    for key, df in csv_data.items():
        if (
            isinstance(key, str)
            and key.startswith("desc_")
            and isinstance(df, pd.DataFrame)
            and not df.empty
            and "name" in df.columns
        ):
            try:
                names = df["name"].dropna()
                if names.empty:
                    continue

                campaign_name = names.iloc[0]

                if "abbreviation" in df.columns:
                    row_idx = names.index[0]
                    if row_idx in df.index and not pd.isna(df.loc[row_idx, "abbreviation"]):
                        campaign_abbr = df.loc[row_idx, "abbreviation"]

                break
            except Exception:
                continue

    return campaign_name, campaign_abbr


def compute_campaign_metrics(
    activities: pd.DataFrame,
    csv_data: Dict[str, pd.DataFrame],
) -> dict:
    """
    Compute campaign-level metrics that do not depend on plotting.
    """
    unique_users_count = int(activities["pid"].nunique())

    campaign_start, campaign_end = _get_campaign_bounds(activities, csv_data)

    campaign_length_days = 0
    if pd.notna(campaign_start) and pd.notna(campaign_end):
        campaign_length_days = max(0, int((campaign_end - campaign_start).days))

    campaign_name, campaign_abbr = _extract_campaign_identity(csv_data)

    return {
        "unique_users": unique_users_count,
        "length_days": campaign_length_days,
        "start_date": campaign_start.date() if pd.notna(campaign_start) else None,
        "end_date": campaign_end.date() if pd.notna(campaign_end) else None,
        "name": campaign_name,
        "abbreviation": campaign_abbr,
    }


def compute_dropout_metrics(activities: pd.DataFrame) -> dict:
    """
    Compute dropout metrics and return both the metrics and the per-user table.
    """
    user_dropout = (
        activities.groupby("pid")
        .agg(first_activity=("createdAt", "min"), last_activity=("createdAt", "max"))
        .dropna()
    )

    if user_dropout.empty:
        return {
            "metrics": None,
            "user_dropout": user_dropout,
        }

    user_dropout["dropout_days"] = (
        user_dropout["last_activity"] - user_dropout["first_activity"]
    ).dt.days
    user_dropout.loc[user_dropout["dropout_days"] < 0, "dropout_days"] = 0

    dropout_stats = user_dropout["dropout_days"].describe(
        percentiles=[0.25, 0.5, 0.75, 0.9]
    ).round(2)

    metrics = {
        "avg_dropout_days": _safe_float(dropout_stats.get("mean", 0.0)),
        "median_dropout_days": _safe_float(dropout_stats.get("50%", 0.0)),
        "min_dropout_days": _safe_float(dropout_stats.get("min", 0.0)),
        "max_dropout_days": _safe_float(dropout_stats.get("max", 0.0)),
        "std_dropout_days": _safe_float(dropout_stats.get("std", 0.0)),
    }

    return {
        "metrics": metrics,
        "user_dropout": user_dropout,
    }


def compute_joining_metrics(
    activities: pd.DataFrame,
    csv_data: Dict[str, pd.DataFrame],
) -> dict:
    """
    Compute joining metrics and return both the metrics and the per-user table.
    """
    campaign_start, _ = _get_campaign_bounds(activities, csv_data)

    user_dropout = (
        activities.groupby("pid")
        .agg(first_activity=("createdAt", "min"), last_activity=("createdAt", "max"))
        .dropna()
    )

    if user_dropout.empty or pd.isna(campaign_start):
        return {
            "metrics": None,
            "user_dropout": user_dropout,
        }

    user_dropout["dropout_days"] = (
        user_dropout["last_activity"] - user_dropout["first_activity"]
    ).dt.days
    user_dropout.loc[user_dropout["dropout_days"] < 0, "dropout_days"] = 0

    user_dropout["joining_days"] = (
        user_dropout["first_activity"] - campaign_start
    ).dt.days

    joining_stats = user_dropout["joining_days"].describe(
        percentiles=[0.25, 0.5, 0.75, 0.9]
    ).round(2)

    metrics = {
        "avg_joining_days": _safe_float(joining_stats.get("mean", 0.0)),
        "median_joining_days": _safe_float(joining_stats.get("50%", 0.0)),
        "min_joining_days": _safe_float(joining_stats.get("min", 0.0)),
        "max_joining_days": _safe_float(joining_stats.get("max", 0.0)),
        "std_joining_days": _safe_float(joining_stats.get("std", 0.0)),
    }

    return {
        "metrics": metrics,
        "user_dropout": user_dropout,
    }



def compute_real_dropout_and_weekly_retention(
    activities: pd.DataFrame,
    csv_data: Dict[str, pd.DataFrame],
    enrolled_user_ids: set,
    *,
    active_types: list[str],
    inactivity_days: int = DROPOUT_INACTIVITY_DAYS,
) -> dict:
    """
    Compute real dropout and weekly retention.

    Definitions:
    - enrolled user: participant from users.xlsx / resolved participant cohort
    - qualifying activity: activity whose type is in active_types
    - never-active: enrolled participant with zero qualifying activities
    - dropped out: participant with at least one qualifying activity whose last
      qualifying activity is at least inactivity_days before campaign end
    - retained/censored: participant with at least one qualifying activity whose
      dropout cannot be confirmed by campaign end

    Weekly retention:
    - active_users_count: participants active in that week
    - retained_users_count: participants who have started and have not crossed
      the inactivity threshold by the end of that week
    """
    if not enrolled_user_ids:
        enrolled_user_ids = _series_to_id_set(activities["pid"]) if "pid" in activities.columns else set()

    normalized_enrolled_user_ids = set()
    for pid in enrolled_user_ids:
        normalized = _normalize_user_id(pid)
        if normalized is not None:
            normalized_enrolled_user_ids.add(normalized)

    enrolled_user_ids = normalized_enrolled_user_ids

    total_enrolled = len(enrolled_user_ids)

    empty_result = {
        "metrics": {
            "total_enrolled_users": total_enrolled,
            "joined_users_count": 0,
            "never_active_users_count": total_enrolled,
            "dropout_users_count": 0,
            "retained_or_censored_users_count": 0,
            "dropout_rate_among_enrolled": 0.0,
            "dropout_rate_among_joined": 0.0,
            "retention_rate_among_enrolled": 0.0,
            "retention_rate_among_joined": 0.0,
            "inactivity_days_threshold": inactivity_days,
            "campaign_end_source": "unknown",
        },
        "user_dropout": pd.DataFrame(),
        "weekly_retention": pd.DataFrame(),
    }

    if activities is None or activities.empty or total_enrolled <= 0:
        return empty_result

    scoped = filter_activities_by_types(activities, active_types)
    if scoped.empty:
        user_dropout = pd.DataFrame(index=pd.Index(sorted(enrolled_user_ids), name="pid"))
        user_dropout["dropout_status"] = "never_active"

        empty_result["user_dropout"] = user_dropout
        return empty_result

    scoped = scoped.copy()
    scoped["pid"] = scoped["pid"].map(_normalize_user_id)
    scoped = scoped[scoped["pid"].isin(enrolled_user_ids)].copy()

    if scoped.empty:
        user_dropout = pd.DataFrame(index=pd.Index(sorted(enrolled_user_ids), name="pid"))
        user_dropout["dropout_status"] = "never_active"

        empty_result["user_dropout"] = user_dropout
        return empty_result

    scoped["createdAt"] = pd.to_datetime(scoped["createdAt"], errors="coerce", utc=True)
    scoped = scoped.dropna(subset=["pid", "createdAt"]).copy()

    if scoped.empty:
        user_dropout = pd.DataFrame(index=pd.Index(sorted(enrolled_user_ids), name="pid"))
        user_dropout["dropout_status"] = "never_active"

        empty_result["user_dropout"] = user_dropout
        return empty_result

    scoped["activity_day"] = scoped["createdAt"].dt.tz_convert(None).dt.normalize()

    waves = get_campaign_waves(csv_data)
    campaign_end_source = "desc_waves" if not waves.empty else "last_activity_fallback"

    campaign_start, campaign_end = _get_campaign_bounds(scoped, csv_data)

    campaign_start = pd.to_datetime(campaign_start, errors="coerce", utc=True)
    campaign_end = pd.to_datetime(campaign_end, errors="coerce", utc=True)

    if pd.isna(campaign_start):
        campaign_start = scoped["createdAt"].min()
    if pd.isna(campaign_end):
        campaign_end = scoped["createdAt"].max()

    if pd.isna(campaign_start) or pd.isna(campaign_end):
        return empty_result

    campaign_start_day = campaign_start.tz_convert(None).normalize()
    campaign_end_day = campaign_end.tz_convert(None).normalize()

    all_users_index = pd.Index(sorted(enrolled_user_ids), name="pid")

    user_activity = (
        scoped.groupby("pid")
        .agg(
            first_activity=("createdAt", "min"),
            last_activity=("createdAt", "max"),
            active_days_count=("activity_day", "nunique"),
            activity_records_count=("createdAt", "size"),
        )
    )

    user_dropout = pd.DataFrame(index=all_users_index).join(user_activity)

    user_dropout["has_started"] = user_dropout["first_activity"].notna()

    user_dropout["first_activity_day"] = (
        pd.to_datetime(user_dropout["first_activity"], errors="coerce", utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )
    user_dropout["last_activity_day"] = (
        pd.to_datetime(user_dropout["last_activity"], errors="coerce", utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )

    user_dropout["days_from_campaign_start_to_first_activity"] = (
        user_dropout["first_activity_day"] - campaign_start_day
    ).dt.days

    user_dropout["days_from_first_to_last_activity"] = (
        user_dropout["last_activity_day"] - user_dropout["first_activity_day"]
    ).dt.days

    user_dropout["days_since_last_activity_at_campaign_end"] = (
        campaign_end_day - user_dropout["last_activity_day"]
    ).dt.days

    user_dropout["dropout_date"] = (
        user_dropout["last_activity_day"] + pd.to_timedelta(inactivity_days, unit="D")
    )

    user_dropout["days_until_dropout"] = (
        user_dropout["dropout_date"] - user_dropout["first_activity_day"]
    ).dt.days

    user_dropout["is_dropout"] = (
        user_dropout["has_started"]
        & (user_dropout["days_since_last_activity_at_campaign_end"] >= inactivity_days)
    )

    user_dropout["dropout_status"] = "never_active"
    user_dropout.loc[
        user_dropout["has_started"] & user_dropout["is_dropout"],
        "dropout_status",
    ] = "dropped_out"
    user_dropout.loc[
        user_dropout["has_started"] & ~user_dropout["is_dropout"],
        "dropout_status",
    ] = "retained_or_censored"

    joined_users_count = int(user_dropout["has_started"].sum())
    never_active_users_count = int((~user_dropout["has_started"]).sum())
    dropout_users_count = int(user_dropout["is_dropout"].sum())
    retained_or_censored_users_count = int(
        (user_dropout["has_started"] & ~user_dropout["is_dropout"]).sum()
    )

    dropout_rate_among_enrolled = (
        dropout_users_count / total_enrolled * 100.0
        if total_enrolled > 0
        else 0.0
    )
    dropout_rate_among_joined = (
        dropout_users_count / joined_users_count * 100.0
        if joined_users_count > 0
        else 0.0
    )
    retention_rate_among_enrolled = (
        retained_or_censored_users_count / total_enrolled * 100.0
        if total_enrolled > 0
        else 0.0
    )
    retention_rate_among_joined = (
        retained_or_censored_users_count / joined_users_count * 100.0
        if joined_users_count > 0
        else 0.0
    )

    # ------------------------------------------------------------------
    # Weekly retention table
    # ------------------------------------------------------------------
    weekly_rows = []

    week_start = campaign_start_day
    week_number = 1

    while week_start <= campaign_end_day:
        week_end = min(week_start + pd.Timedelta(days=6), campaign_end_day)

        active_mask = (
            (scoped["activity_day"] >= week_start)
            & (scoped["activity_day"] <= week_end)
        )
        active_users = set(scoped.loc[active_mask, "pid"].dropna().unique())
        active_users_count = len(active_users)

        joined_by_week = (
            user_dropout["has_started"]
            & user_dropout["first_activity_day"].notna()
            & (user_dropout["first_activity_day"] <= week_end)
        )

        retained_by_week = (
            joined_by_week
            & user_dropout["dropout_date"].notna()
            & (user_dropout["dropout_date"] > week_end)
        )

        joined_by_week_count = int(joined_by_week.sum())
        retained_users_count = int(retained_by_week.sum())
        cumulative_dropout_count = int(joined_by_week_count - retained_users_count)

        weekly_rows.append(
            {
                "week": week_number,
                "week_start": week_start.date(),
                "week_end": week_end.date(),
                "active_users_count": active_users_count,
                "active_pct_of_enrolled": (
                    active_users_count / total_enrolled * 100.0
                    if total_enrolled > 0
                    else 0.0
                ),
                "joined_by_week_count": joined_by_week_count,
                "retained_users_count": retained_users_count,
                "retention_pct_of_enrolled": (
                    retained_users_count / total_enrolled * 100.0
                    if total_enrolled > 0
                    else 0.0
                ),
                "retention_pct_of_joined_by_week": (
                    retained_users_count / joined_by_week_count * 100.0
                    if joined_by_week_count > 0
                    else 0.0
                ),
                "cumulative_dropout_count": cumulative_dropout_count,
            }
        )

        week_start = week_start + pd.Timedelta(days=7)
        week_number += 1

    weekly_retention = pd.DataFrame(weekly_rows)

    metrics = {
        "total_enrolled_users": total_enrolled,
        "joined_users_count": joined_users_count,
        "never_active_users_count": never_active_users_count,
        "dropout_users_count": dropout_users_count,
        "retained_or_censored_users_count": retained_or_censored_users_count,
        "dropout_rate_among_enrolled": round(dropout_rate_among_enrolled, 2),
        "dropout_rate_among_joined": round(dropout_rate_among_joined, 2),
        "retention_rate_among_enrolled": round(retention_rate_among_enrolled, 2),
        "retention_rate_among_joined": round(retention_rate_among_joined, 2),
        "inactivity_days_threshold": inactivity_days,
        "campaign_start": campaign_start_day.date(),
        "campaign_end": campaign_end_day.date(),
        "campaign_end_source": campaign_end_source,
    }

    return {
        "metrics": metrics,
        "user_dropout": user_dropout,
        "weekly_retention": weekly_retention,
    }



def _series_to_id_set(series: pd.Series) -> set[str]:
    """
    Convert a Series of identifiers into a de-duplicated normalized string set,
    dropping NaN and blank-string values.
    """
    if series.empty:
        return set()

    return {
        normalized
        for normalized in series.map(_normalize_user_id)
        if normalized is not None
    }


def get_enrolled_user_ids(
    csv_data: Dict[str, pd.DataFrame],
    activities: pd.DataFrame,
) -> set[str]:
    """
    Determine the enrolled user base.

    In normal project runs, the enrolled base is the authoritative users.xlsx
    roster resolved through user_email_mapping.txt. This deliberately avoids
    using aggregation/activity campaign exports as the cohort source, because
    those exports can contain test accounts.

    A fallback to activities is allowed only when users.xlsx is absent, which
    keeps small developer/unit-test scenarios usable without reintroducing the
    production bug.
    """
    try:
        return resolve_analysis_user_ids()
    except FileNotFoundError as e:
        if os.path.exists(USERS_FILE_PATH):
            raise
        logger.warning(f"No users.xlsx available; falling back to users seen in activities: {e}")
        return _series_to_id_set(activities["pid"])


def filter_activities_by_types(
    activities: pd.DataFrame,
    allowed_types: list[str],
) -> pd.DataFrame:
    """
    Return a copy of activities filtered to the provided descriptor types.
    """
    if activities.empty or "type" not in activities.columns:
        return activities.iloc[0:0].copy()

    return activities[activities["type"].isin(allowed_types)].copy()


def compute_reward_based_active_user_ids(activities: pd.DataFrame) -> set:
    """
    Legacy comparison metric:
    active user = any user who earned points or has one or more rewards.

    This preserves the current implementation logic so you can compare
    descriptor-based active users vs reward-based active users.
    """
    frame = activities.copy()

    if "points" not in frame.columns:
        frame["points"] = frame["rewardedParticipations"].apply(extract_points)

    if "num_rewards" not in frame.columns:
        if "detailed_rewards" in frame.columns:
            frame["num_rewards"] = frame["detailed_rewards"].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
        else:
            frame["detailed_rewards"] = frame["rewardedParticipations"].apply(extract_detailed_rewards)
            frame["num_rewards"] = frame["detailed_rewards"].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )

    rewarded = frame[(frame["points"] > 0) | (frame["num_rewards"] > 0)]

    return _series_to_id_set(rewarded["pid"])


def compute_active_passive_by_types(
    activities: pd.DataFrame,
    enrolled_user_ids: set,
    active_types: list[str],
) -> dict:
    """
    Descriptor-based active/passive classification.

    active = enrolled user with at least one activity record whose type is in active_types
    passive = enrolled user with zero activity records whose type is in active_types
    """
    scoped_activities = filter_activities_by_types(activities, active_types)
    active_user_ids = _series_to_id_set(scoped_activities["pid"])

    if enrolled_user_ids:
        passive_user_ids = set(enrolled_user_ids) - set(active_user_ids)
    else:
        passive_user_ids = set()

    return {
        "active_user_ids": active_user_ids,
        "passive_user_ids": passive_user_ids,
        "active_users_count": len(active_user_ids),
        "passive_users_count": len(passive_user_ids),
        "active_records_count": int(len(scoped_activities)),
    }


def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.mean())


def _safe_median(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.median())


def _safe_min(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.min())


def _safe_max(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.max())

def _safe_std(series: pd.Series) -> float:
    if series.empty or len(series.dropna()) <= 1:
        return 0.0

    value = series.std()
    if pd.isna(value):
        return 0.0

    return float(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def compute_scope_usage_metrics(
    activities: pd.DataFrame,
    enrolled_user_ids: set,
    metric_types: list[str],
    usage_time_types: list[str],
) -> dict:
    """
    Compute per-scope usage metrics.

    Rules:
    - Average active days / participant, active players per day, and average time to first inactivity
      use metric_types (GameBus includes DAY_AGGREGATE here).
    - Usage time / day of week, usage time / hour of day buckets, and peak activity hour
      use usage_time_types (GameBus excludes DAY_AGGREGATE here).
    """
    metric_activities = filter_activities_by_types(activities, metric_types)
    usage_activities = filter_activities_by_types(activities, usage_time_types)

    # ------------------------------------------------------------------
    # 1) Average active days / participant
    # Includes passive enrolled users as 0 if enrolled_user_ids is available
    # ------------------------------------------------------------------
    user_active_days = (
        metric_activities.groupby("pid")["date"]
        .nunique()
        if not metric_activities.empty
        else pd.Series(dtype="int64")
    )

    if enrolled_user_ids:
        active_days_per_participant = user_active_days.reindex(
            pd.Index(list(enrolled_user_ids)),
            fill_value=0,
        )
    else:
        active_days_per_participant = user_active_days

    # ------------------------------------------------------------------
    # 2) Active players per day
    # ------------------------------------------------------------------
    active_players_per_day = (
        metric_activities.groupby("date")["pid"]
        .nunique()
        .sort_index()
        if not metric_activities.empty
        else pd.Series(dtype="int64")
    )

    # ------------------------------------------------------------------
    # 3) Average time to first inactivity (keep current proxy)
    # Current proxy = last_activity - first_activity
    # Only for users with at least one scoped activity
    # ------------------------------------------------------------------
    user_inactivity = (
        metric_activities.groupby("pid")
        .agg(first_activity=("createdAt", "min"), last_activity=("createdAt", "max"))
        .dropna()
    )

    if not user_inactivity.empty:
        user_inactivity["time_to_first_inactivity_days"] = (
            user_inactivity["last_activity"] - user_inactivity["first_activity"]
        ).dt.days
        user_inactivity.loc[
            user_inactivity["time_to_first_inactivity_days"] < 0,
            "time_to_first_inactivity_days",
        ] = 0

        time_to_first_inactivity = user_inactivity["time_to_first_inactivity_days"]
    else:
        time_to_first_inactivity = pd.Series(dtype="float64")

    # ------------------------------------------------------------------
    # 4) Usage time / day of week
    # Uses usage_time_types (e.g., excludes DAY_AGGREGATE for GameBus)
    # ------------------------------------------------------------------
    if not usage_activities.empty:
        day_names = usage_activities["createdAt"].dt.day_name()
        usage_by_day_of_week = day_names.value_counts().reindex(WEEKDAY_ORDER, fill_value=0)
    else:
        usage_by_day_of_week = pd.Series(0, index=WEEKDAY_ORDER, dtype="int64")

    # ------------------------------------------------------------------
    # 5) Usage time / hour of day
    # ------------------------------------------------------------------
    if not usage_activities.empty:
        usage_by_hour = (
            usage_activities["hour"]
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
        )
    else:
        usage_by_hour = pd.Series(dtype="int64")

    # ------------------------------------------------------------------
    # 6) Usage time / hour buckets
    # ------------------------------------------------------------------
    if not usage_by_hour.empty:
        bucket_series = pd.cut(
            usage_activities["hour"].dropna().astype(int),
            bins=HOUR_BUCKET_BINS,
            labels=HOUR_BUCKET_LABELS,
        )
        usage_by_hour_buckets = bucket_series.value_counts().reindex(HOUR_BUCKET_LABELS, fill_value=0)
    else:
        usage_by_hour_buckets = pd.Series(0, index=HOUR_BUCKET_LABELS, dtype="int64")

    # ------------------------------------------------------------------
    # 7) Peak activity hour
    # ------------------------------------------------------------------
    peak_activity_hour = None
    if not usage_by_hour.empty:
        peak_activity_hour = int(usage_by_hour.idxmax())

    return {
        "metric_activities_count": int(len(metric_activities)),
        "usage_time_activities_count": int(len(usage_activities)),

        "active_days_per_participant": active_days_per_participant,
        "avg_active_days_per_participant": _safe_mean(active_days_per_participant),
        "median_active_days_per_participant": _safe_median(active_days_per_participant),
        "std_active_days_per_participant": _safe_std(active_days_per_participant),
        "min_active_days_per_participant": _safe_min(active_days_per_participant),
        "max_active_days_per_participant": _safe_max(active_days_per_participant),

        "active_players_per_day": active_players_per_day,
        "avg_active_players_per_day": _safe_mean(active_players_per_day),
        "median_active_players_per_day": _safe_median(active_players_per_day),
        "std_active_players_per_day": _safe_std(active_players_per_day),
        "min_active_players_per_day": _safe_min(active_players_per_day),
        "max_active_players_per_day": _safe_max(active_players_per_day),

        "time_to_first_inactivity_by_user": time_to_first_inactivity,
        "avg_time_to_first_inactivity_days": _safe_mean(time_to_first_inactivity),
        "median_time_to_first_inactivity_days": _safe_median(time_to_first_inactivity),
        "std_time_to_first_inactivity_days": _safe_std(time_to_first_inactivity),
        "min_time_to_first_inactivity_days": _safe_min(time_to_first_inactivity),
        "max_time_to_first_inactivity_days": _safe_max(time_to_first_inactivity),

        "usage_by_day_of_week": usage_by_day_of_week,
        "usage_by_hour": usage_by_hour,
        "usage_by_hour_buckets": usage_by_hour_buckets,
        "peak_activity_hour": peak_activity_hour,
    }