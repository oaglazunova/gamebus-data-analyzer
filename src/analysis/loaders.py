from __future__ import annotations

import ast
import glob
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd

from src.analysis.common import (
    CONFIG_DIR,
    RAW_DATA_DIR,
    ensure_dir,
    logger,
)


CAMPAIGN_DATA_FILENAME = (
    "campaign_data.zip"
)

CAMPAIGN_DESCRIPTION_FILENAME = (
    "campaign_desc.xlsx"
)


def _load_excel_sheets(
    path: str | Path,
    prefix: str = "",
) -> Dict[str, pd.DataFrame]:
    sheets: Dict[
        str,
        pd.DataFrame,
    ] = {}

    path = os.fspath(
        path
    )

    if not os.path.exists(
        path
    ):
        logger.warning(
            f"Excel file not found: {path}"
        )

        return sheets

    try:
        with pd.ExcelFile(
            path
        ) as workbook:
            for sheet_name in (
                workbook.sheet_names
            ):
                try:
                    df = workbook.parse(
                        sheet_name
                    )

                    if df.empty:
                        logger.warning(
                            "Sheet "
                            f"'{sheet_name}' "
                            f"in {path} is empty; "
                            "skipping"
                        )

                        continue

                    key = (
                        f"{prefix}"
                        f"{sheet_name}"
                    )

                    sheets[
                        key
                    ] = df

                    logger.info(
                        "Loaded sheet "
                        f"'{sheet_name}' "
                        f"from {path} "
                        f"({len(df)} rows, "
                        f"{len(df.columns)} cols)"
                    )

                except Exception as exc:
                    logger.error(
                        "Error loading sheet "
                        f"'{sheet_name}' "
                        f"from {path}: {exc}"
                    )

    except Exception as exc:
        logger.error(
            f"Error opening {path}: "
            f"{exc}"
        )

    return sheets


def _resolve_campaign_paths(
    *,
    campaign_data_path: (
        str | Path | None
    ),
    campaign_description_path: (
        str | Path | None
    ),
) -> tuple[
    Path,
    Path,
]:
    """
    Resolve campaign inputs.

    When explicit paths are supplied, they are used directly.

    When omitted, retain the legacy repository-level layout:

        config/campaign_data.zip
        config/campaign_desc.xlsx
    """
    if campaign_data_path is None:
        data_path = Path(
            CONFIG_DIR
        ) / CAMPAIGN_DATA_FILENAME

    else:
        data_path = Path(
            campaign_data_path
        ).expanduser()

    if (
        campaign_description_path
        is None
    ):
        description_path = (
            Path(
                CONFIG_DIR
            )
            / CAMPAIGN_DESCRIPTION_FILENAME
        )

    else:
        description_path = Path(
            campaign_description_path
        ).expanduser()

    return (
        data_path,
        description_path,
    )


def load_excel_files(
    campaign_data_path: (
        str | Path | None
    ) = None,
    campaign_description_path: (
        str | Path | None
    ) = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load campaign-level GameBus data.

    Dataset-aware usage:

        load_excel_files(
            campaign_data_path=...,
            campaign_description_path=...,
        )

    Legacy usage remains supported:

        load_excel_files()

    The campaign analytics ZIP is mapped to:

        aggregation
        activities
        navigation
        notification_events
        sensor_events

    Campaign-description workbook sheets are prefixed
    with ``desc_``.
    """
    (
        data_path,
        description_path,
    ) = _resolve_campaign_paths(
        campaign_data_path=(
            campaign_data_path
        ),
        campaign_description_path=(
            campaign_description_path
        ),
    )

    data_dict: Dict[
        str,
        pd.DataFrame,
    ] = {}

    if (
        campaign_data_path
        is None
    ):
        ensure_dir(
            CONFIG_DIR
        )

    if data_path.exists():
        try:
            with zipfile.ZipFile(
                data_path,
                "r",
            ) as archive:
                name_map = {
                    (
                        "1-aggregated-data.csv"
                    ): "aggregation",
                    (
                        "2-activities.csv"
                    ): "activities",
                    (
                        "3-navigation-events.csv"
                    ): "navigation",
                    (
                        "4-notification-events.csv"
                    ): "notification_events",
                    (
                        "5-sensor-events.csv"
                    ): "sensor_events",
                }

                members = {
                    Path(
                        name
                    ).name.casefold(): name
                    for name
                    in archive.namelist()
                    if not name.endswith(
                        "/"
                    )
                }

                for (
                    filename,
                    key,
                ) in name_map.items():
                    lookup_name = (
                        filename.casefold()
                    )

                    if (
                        lookup_name
                        not in members
                    ):
                        logger.warning(
                            "Expected file "
                            f"'{filename}' "
                            "not found in "
                            f"{data_path}"
                        )

                        continue

                    try:
                        with archive.open(
                            members[
                                lookup_name
                            ]
                        ) as file:
                            df = pd.read_csv(
                                file
                            )

                        if df.empty:
                            logger.warning(
                                "CSV "
                                f"'{filename}' "
                                f"in {data_path} "
                                "is empty; skipping"
                            )

                            continue

                        data_dict[
                            key
                        ] = df

                        logger.info(
                            "Loaded "
                            f"'{filename}' "
                            "from ZIP into key "
                            f"'{key}' "
                            f"({len(df)} rows)"
                        )

                    except Exception as exc:
                        logger.error(
                            "Error reading "
                            f"'{filename}' "
                            f"from {data_path}: "
                            f"{exc}"
                        )

        except zipfile.BadZipFile as exc:
            logger.error(
                "Invalid ZIP file "
                f"{data_path}: {exc}"
            )

        except Exception as exc:
            logger.error(
                "Error opening ZIP "
                f"{data_path}: {exc}"
            )

    else:
        logger.warning(
            "Campaign data ZIP not found: "
            f"{data_path}"
        )

    try:
        data_dict.update(
            _load_excel_sheets(
                description_path,
                prefix="desc_",
            )
        )

    except Exception as exc:
        logger.error(
            "Error loading campaign "
            f"description {description_path}: "
            f"{exc}"
        )

    if not data_dict:
        logger.warning(
            "No campaign-level data loaded "
            f"from {data_path} or "
            f"{description_path}"
        )

    return data_dict


def load_json_files(
    raw_data_dir: (
        str | Path | None
    ) = None,
) -> Dict[
    str,
    Union[
        pd.DataFrame,
        Dict,
    ],
]:
    """
    Load participant JSON data.

    Dataset-aware usage:

        load_json_files(
            raw_data_dir=dataset_dir / "data_raw"
        )

    Legacy usage remains supported:

        load_json_files()
    """
    if raw_data_dir is None:
        resolved_raw_dir = Path(
            RAW_DATA_DIR
        )

        ensure_dir(
            os.fspath(
                resolved_raw_dir
            )
        )

    else:
        resolved_raw_dir = Path(
            raw_data_dir
        ).expanduser()

    data_dict: Dict[
        str,
        Union[
            pd.DataFrame,
            Dict,
        ],
    ] = {}

    if not resolved_raw_dir.exists():
        logger.warning(
            "Raw-data directory not found: "
            f"{resolved_raw_dir}"
        )

        return data_dict

    if not resolved_raw_dir.is_dir():
        logger.warning(
            "Raw-data path is not a "
            "directory: "
            f"{resolved_raw_dir}"
        )

        return data_dict

    json_files = sorted(
        glob.glob(
            os.path.join(
                os.fspath(
                    resolved_raw_dir
                ),
                "*.json",
            )
        )
    )

    if not json_files:
        logger.warning(
            "No JSON files found in "
            f"{resolved_raw_dir}"
        )

        return data_dict

    for file in json_files:
        filename = os.path.basename(
            file
        )

        key = filename.rsplit(
            ".",
            1,
        )[0]

        try:
            with open(
                file,
                "r",
                encoding="utf-8",
            ) as handle:
                try:
                    data = json.load(
                        handle
                    )

                except json.JSONDecodeError as exc:
                    logger.error(
                        "Invalid JSON in "
                        f"{file}: {exc}"
                    )

                    continue

            if isinstance(
                data,
                list,
            ):
                if not data:
                    logger.warning(
                        "Empty list in "
                        f"{filename}; skipping"
                    )

                    continue

                if isinstance(
                    data[0],
                    dict,
                ):
                    chunk_size = 5000
                    count = len(
                        data
                    )

                    if count > chunk_size:
                        frames = []

                        for start in range(
                            0,
                            count,
                            chunk_size,
                        ):
                            chunk = data[
                                start:
                                start
                                + chunk_size
                            ]

                            frames.append(
                                pd.DataFrame.from_records(
                                    chunk
                                )
                            )

                        df = pd.concat(
                            frames,
                            ignore_index=True,
                            sort=True,
                        )

                    else:
                        df = (
                            pd.DataFrame
                            .from_records(
                                data
                            )
                        )

                else:
                    df = pd.DataFrame(
                        {
                            "value": data,
                        }
                    )

                if df.empty:
                    logger.warning(
                        "Empty DataFrame created "
                        f"from {filename}; "
                        "skipping"
                    )

                    continue

                data_dict[
                    key
                ] = df

            else:
                if not data:
                    logger.warning(
                        "Empty dict in "
                        f"{filename}; skipping"
                    )

                    continue

                data_dict[
                    key
                ] = data

        except FileNotFoundError:
            logger.error(
                f"File not found: {file}"
            )

        except PermissionError:
            logger.error(
                "Permission denied: "
                f"{file}"
            )

        except Exception as exc:
            logger.error(
                "Error loading "
                f"{file}: {exc}"
            )

    if not data_dict:
        logger.warning(
            "No data loaded from JSON "
            f"files in {resolved_raw_dir}"
        )

    return data_dict


def _parse_rewards_jsonlike(
    rewards_value: Any,
) -> List[Dict]:
    """
    Parse reward payloads that may come as:

    - JSON string
    - Python-literal-like string
    - already-materialized list[dict]
    - single dict
    - empty / NaN / None
    """
    if rewards_value is None:
        return []

    if isinstance(
        rewards_value,
        list,
    ):
        return [
            reward
            for reward in rewards_value
            if isinstance(
                reward,
                dict,
            )
        ]

    if isinstance(
        rewards_value,
        dict,
    ):
        return [
            rewards_value
        ]

    try:
        if pd.isna(
            rewards_value
        ):
            return []

    except Exception:
        pass

    if not isinstance(
        rewards_value,
        str,
    ):
        return []

    value = rewards_value.strip()

    if not value:
        return []

    try:
        obj = json.loads(
            value
        )

        if isinstance(
            obj,
            list,
        ):
            return [
                reward
                for reward in obj
                if isinstance(
                    reward,
                    dict,
                )
            ]

        if isinstance(
            obj,
            dict,
        ):
            return [
                obj
            ]

        return []

    except Exception:
        pass

    try:
        obj = json.loads(
            value.replace(
                "'",
                '"',
            )
        )

        if isinstance(
            obj,
            list,
        ):
            return [
                reward
                for reward in obj
                if isinstance(
                    reward,
                    dict,
                )
            ]

        if isinstance(
            obj,
            dict,
        ):
            return [
                obj
            ]

        return []

    except Exception:
        pass

    try:
        obj = ast.literal_eval(
            value
        )

        if isinstance(
            obj,
            list,
        ):
            return [
                reward
                for reward in obj
                if isinstance(
                    reward,
                    dict,
                )
            ]

        if isinstance(
            obj,
            dict,
        ):
            return [
                obj
            ]

        return []

    except Exception:
        return []


def extract_points(
    rewards_value: Any,
) -> int:
    rewards = (
        _parse_rewards_jsonlike(
            rewards_value
        )
    )

    total = 0

    for reward in rewards:
        if isinstance(
            reward,
            dict,
        ):
            try:
                total += int(
                    reward.get(
                        "points",
                        0,
                    )
                    or 0
                )

            except Exception:
                pass

    return total


def extract_detailed_rewards(
    rewards_value: Any,
) -> List[Dict]:
    return _parse_rewards_jsonlike(
        rewards_value
    )