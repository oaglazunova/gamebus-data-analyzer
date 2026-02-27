from __future__ import annotations

import os
import json
import glob
import ast
from typing import Dict, Union, List
import pandas as pd

from src.analysis.common import ensure_dir, CONFIG_DIR, RAW_DATA_DIR, logger


# -----------------------------------------------------------------------------
# Excel loading (campaign_data.zip + campaign_desc.xlsx)
# -----------------------------------------------------------------------------
def _load_excel_sheets(path: str, prefix: str = "") -> Dict[str, pd.DataFrame]:
    sheets: Dict[str, pd.DataFrame] = {}
    if not os.path.exists(path):
        logger.warning(f"Excel file not found: {path}")
        return sheets

    try:
        xl = pd.ExcelFile(path)
        for sheet_name in xl.sheet_names:
            try:
                df = pd.read_excel(path, sheet_name=sheet_name)
                if df.empty:
                    logger.warning(f"Sheet '{sheet_name}' in {path} is empty; skipping")
                    continue
                key = f"{prefix}{sheet_name}"
                sheets[key] = df
                logger.info(f"Loaded sheet '{sheet_name}' from {path} ({len(df)} rows, {len(df.columns)} cols)")
            except Exception as e:
                logger.error(f"Error loading sheet '{sheet_name}' from {path}: {e}")
    except Exception as e:
        logger.error(f"Error opening {path}: {e}")

    return sheets


def load_excel_files() -> Dict[str, pd.DataFrame]:
    """
    Loads:
      - CSVs from config/campaign_data.zip (mapped to keys: aggregation, activities, navigation, notification_events, sensor_events)
      - Sheets from config/campaign_desc.xlsx prefixed with desc_ (e.g., desc_tasks, desc_challenges, ...)
    """
    import zipfile

    ensure_dir(CONFIG_DIR)

    data_dict: Dict[str, pd.DataFrame] = {}

    zip_path = os.path.join(CONFIG_DIR, "campaign_data.zip")
    if os.path.exists(zip_path):
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                name_map = {
                    "1-aggregated-data.csv": "aggregation",
                    "2-activities.csv": "activities",
                    "3-navigation-events.csv": "navigation",
                    "4-notification-events.csv": "notification_events",
                    "5-sensor-events.csv": "sensor_events",
                }
                members = {
                    os.path.basename(n).lower(): n
                    for n in zf.namelist()
                    if not n.endswith("/")
                }
                for fname, key in name_map.items():
                    lf = fname.lower()
                    if lf not in members:
                        logger.warning(f"Expected file '{fname}' not found in {zip_path}")
                        continue
                    try:
                        with zf.open(members[lf]) as f:
                            df = pd.read_csv(f)
                        if df.empty:
                            logger.warning(f"CSV '{fname}' in {zip_path} is empty; skipping")
                            continue
                        data_dict[key] = df
                        logger.info(f"Loaded '{fname}' from ZIP into key '{key}' ({len(df)} rows)")
                    except Exception as e:
                        logger.error(f"Error reading '{fname}' from {zip_path}: {e}")
        except zipfile.BadZipFile as e:
            logger.error(f"Invalid ZIP file {zip_path}: {e}")
        except Exception as e:
            logger.error(f"Error opening ZIP {zip_path}: {e}")
    else:
        logger.warning(f"campaign_data.zip not found in {CONFIG_DIR}")

    # campaign_desc.xlsx sheets (prefixed with desc_)
    try:
        data_dict.update(_load_excel_sheets(os.path.join(CONFIG_DIR, "campaign_desc.xlsx"), prefix="desc_"))
    except Exception as e:
        logger.error(f"Error loading campaign_desc.xlsx: {e}")

    if not data_dict:
        logger.warning("No data loaded from campaign_data.zip or campaign_desc.xlsx")

    return data_dict


# -----------------------------------------------------------------------------
# JSON loading
# -----------------------------------------------------------------------------
def load_json_files() -> Dict[str, Union[pd.DataFrame, Dict]]:
    ensure_dir(RAW_DATA_DIR)

    json_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.json"))
    data_dict: Dict[str, Union[pd.DataFrame, Dict]] = {}

    if not json_files:
        logger.warning(f"No JSON files found in {RAW_DATA_DIR}")
        return data_dict

    for file in json_files:
        filename = os.path.basename(file)
        key = filename.rsplit(".", 1)[0]

        try:
            with open(file, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in {file}: {e}")
                    continue

            if isinstance(data, list):
                if not data:
                    logger.warning(f"Empty list in {filename}; skipping")
                    continue
                if isinstance(data[0], dict):
                    # chunked read for very large lists
                    CHUNK_SIZE = 5000
                    n = len(data)
                    if n > CHUNK_SIZE:
                        frames = []
                        for start in range(0, n, CHUNK_SIZE):
                            chunk = data[start : start + CHUNK_SIZE]
                            frames.append(pd.DataFrame.from_records(chunk))
                        df = pd.concat(frames, ignore_index=True, sort=True)
                    else:
                        df = pd.DataFrame.from_records(data)
                else:
                    df = pd.DataFrame({"value": data})

                if df.empty:
                    logger.warning(f"Empty DataFrame created from {filename}; skipping")
                    continue
                data_dict[key] = df
            else:
                if not data:
                    logger.warning(f"Empty dict in {filename}; skipping")
                    continue
                data_dict[key] = data

        except FileNotFoundError:
            logger.error(f"File not found: {file}")
        except PermissionError:
            logger.error(f"Permission denied: {file}")
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")

    if not data_dict:
        logger.warning("No data loaded from JSON files")

    return data_dict


# -----------------------------------------------------------------------------
# Rewards parsing
# -----------------------------------------------------------------------------
def _parse_rewards_jsonlike(rewards_str: str) -> List[Dict]:
    if not isinstance(rewards_str, str) or not rewards_str.strip():
        return []
    s = rewards_str.strip()

    # strict json
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, list) else []
    except Exception:
        pass

    # json with single quotes
    try:
        obj = json.loads(s.replace("'", '"'))
        return obj if isinstance(obj, list) else []
    except Exception:
        pass

    # python literal
    try:
        obj = ast.literal_eval(s)
        return obj if isinstance(obj, list) else []
    except Exception:
        return []


def extract_points(rewards_str: str) -> int:
    rewards = _parse_rewards_jsonlike(rewards_str)
    total = 0
    for r in rewards:
        if isinstance(r, dict):
            try:
                total += int(r.get("points", 0) or 0)
            except Exception:
                pass
    return total


def extract_detailed_rewards(rewards_str: str) -> List[Dict]:
    return _parse_rewards_jsonlike(rewards_str)