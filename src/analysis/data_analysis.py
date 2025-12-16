"""
GameBus Health Behavior Mining Data Analysis Script

This script analyzes data from the GameBus Health Behavior Mining project, including:
1. Excel files in the config directory containing campaign and activity data
2. JSON files in the data_raw directory containing user-specific data like activity types, mood logs, geofence logs

The script generates:
1. Descriptive statistics tables for each variable
2. Visualizations for activity patterns, engagement, and time-series trends
3. Bivariate analyses (selected heatmaps and pivots)
4. A text report summarizing what was loaded and key campaign metrics

All outputs are saved under: <PROJECT_ROOT>/data_analysis

Usage:
    python -m src.analysis.data_analysis

Requirements:
    - pandas
    - numpy
    - matplotlib
    - seaborn
    - scipy
"""

from __future__ import annotations

import os
import sys
import re
import json
import glob
import ast
import logging
from typing import Callable, Tuple, Dict, Optional, Union, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# -----------------------------------------------------------------------------
# Path setup (project root + config import)
# -----------------------------------------------------------------------------
# Add the project root to the Python path to find the config module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from config.paths import PROJECT_ROOT  # noqa: E402

# -----------------------------------------------------------------------------
# Output directories
# -----------------------------------------------------------------------------
OUTPUT_VISUALIZATIONS_DIR = os.path.join(PROJECT_ROOT, "data_analysis")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data_raw")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# -----------------------------------------------------------------------------
# Plot configuration
# -----------------------------------------------------------------------------
plt.style.use("ggplot")
sns.set(style="whitegrid")
sns.set_palette("colorblind")

BAR_COLORMAP = "tab20"
PIE_COLORMAP = "tab10"
CORRELATION_HEATMAP_COLORMAP = "coolwarm"
SEQUENTIAL_HEATMAP_COLORMAP = "viridis"
BOX_COLORMAP = "Set2"

LABEL_MAX_CHARS = 30
ELLIPSIS = "..."

# Wider plot for daily stacked bars and daily date tick labeling
PLOT_FIGSIZE_ACTIVITY_TYPES_STACKED = (30, 8)

# Heatmap guardrails (prevents unreadable huge plots)
MAX_USERS_HEATMAP = 60
MAX_DAYS_HEATMAP = 60
MAX_TYPES_HEATMAP = 15

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
os.makedirs(LOGS_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Clear any existing handlers to avoid duplicate logs if re-imported
if logger.hasHandlers():
    logger.handlers.clear()

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(os.path.join(LOGS_DIR, "data_analysis.log"))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.propagate = False

# -----------------------------------------------------------------------------
# Filesystem utilities
# -----------------------------------------------------------------------------
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def ensure_output_dirs() -> None:
    ensure_dir(OUTPUT_VISUALIZATIONS_DIR)
    ensure_dir(os.path.join(OUTPUT_VISUALIZATIONS_DIR, "statistics"))


def safe_filename(text: Optional[str], max_len: int = 120) -> str:
    if not text:
        return "unknown"
    try:
        s = str(text)
    except Exception:
        s = "unknown"
    s = s.replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9._-]", "", s)
    return (s or "unknown")[:max_len]


def compute_barh_fig_height(
    n_bars: int,
    row_height: float = 0.35,
    min_height: float = 4.0,
    max_height: float = 30.0,
) -> float:
    try:
        n = int(n_bars)
    except Exception:
        n = 10
    h = n * row_height + 2.0
    return max(min_height, min(max_height, h))


# -----------------------------------------------------------------------------
# Label truncation utilities (safe: edits existing Text objects; does not reset ticks)
# -----------------------------------------------------------------------------
def _truncate_text(text: Optional[str], max_chars: int = LABEL_MAX_CHARS) -> str:
    if not text:
        return ""
    try:
        s = str(text)
    except Exception:
        return ""
    if len(s) <= max_chars:
        return s
    cut = max(0, max_chars - len(ELLIPSIS))
    return (s[:cut] + ELLIPSIS) if cut > 0 else ELLIPSIS


def _apply_label_truncation(fig, max_chars: int = LABEL_MAX_CHARS) -> None:
    try:
        try:
            fig.canvas.draw()
        except Exception:
            pass

        for ax in fig.get_axes():
            # Tick labels
            for txt in ax.get_xticklabels():
                try:
                    txt.set_text(_truncate_text(txt.get_text(), max_chars))
                except Exception:
                    pass
            for txt in ax.get_yticklabels():
                try:
                    txt.set_text(_truncate_text(txt.get_text(), max_chars))
                except Exception:
                    pass

            # Legend labels
            leg = ax.get_legend()
            if leg:
                for t in leg.get_texts():
                    try:
                        t.set_text(_truncate_text(t.get_text(), max_chars))
                    except Exception:
                        pass

        try:
            fig.canvas.draw()
        except Exception:
            pass
    except Exception:
        # Never fail plotting because of label truncation
        pass


def create_and_save_figure(
    plot_function: Callable[[], None],
    filename: str,
    figsize: Tuple[float, float] = (10, 6),
) -> None:
    output_dir = os.path.dirname(filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Let plot function create the figure (some plots call plt.figure themselves)
    plot_function()
    fig = plt.gcf()

    try:
        fig.set_size_inches(figsize[0], figsize[1])
    except Exception:
        pass

    _apply_label_truncation(fig, LABEL_MAX_CHARS)

    try:
        fig.tight_layout()
    except Exception:
        pass

    try:
        fig.savefig(filename, bbox_inches="tight")
        logger.info(f"Saved figure -> {filename}")
    except Exception as e:
        logger.error(f"Failed saving figure {filename}: {e}")
    finally:
        try:
            plt.close(fig)
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Small stats formatting helpers
# -----------------------------------------------------------------------------
def _mean_sd(x: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    s = pd.to_numeric(x, errors="coerce").dropna()
    if s.empty:
        return None, None
    mean = float(s.mean())
    sd = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    return mean, sd


def _fmt_mean_sd(mean: Optional[float], sd: Optional[float], digits: int = 2) -> str:
    if mean is None:
        return "N/A"
    if sd is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def _fmt_pct(n: int, d: int, digits: int = 1) -> str:
    if d <= 0:
        return "0.0%"
    return f"{(n / d * 100.0):.{digits}f}%"


def _bucket_hour(h: int) -> str:
    if 6 <= h <= 11:
        return "morning (6AM-11AM)"
    if 12 <= h <= 17:
        return "afternoon (12PM-5PM)"
    if 18 <= h <= 23:
        return "evening (6PM-11PM)"
    return "night (12AM-5AM)"


# -----------------------------------------------------------------------------
# Descriptive statistics tables
# -----------------------------------------------------------------------------
def generate_descriptive_stats(
    df: pd.DataFrame, title: str, filename: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Robust descriptive stats generation.
    Handles object columns that may contain unhashable values (e.g., lists/dicts).
    """
    stats_dir = ensure_dir(os.path.join(OUTPUT_VISUALIZATIONS_DIR, "statistics"))
    base = safe_filename(filename or title)

    out: Dict[str, pd.DataFrame] = {}

    # Numerical
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if num_cols:
        num_stats = df[num_cols].describe(percentiles=[0.25, 0.5, 0.75, 0.9]).T
        num_stats["median"] = df[num_cols].median(numeric_only=True)
        num_stats["variance"] = df[num_cols].var(numeric_only=True)
        num_stats["skew"] = df[num_cols].skew(numeric_only=True)
        num_stats["kurtosis"] = df[num_cols].kurtosis(numeric_only=True)
        num_stats = num_stats.round(2)
        out["numerical"] = num_stats
        num_stats.to_csv(os.path.join(stats_dir, f"{base}_numerical.csv"), index=True)

    # Helper: stringify values to avoid "unhashable type: list" in nunique/value_counts
    def _stable_str(v) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        try:
            # JSON stringify dict/list in a stable manner
            if isinstance(v, (dict, list)):
                return json.dumps(v, sort_keys=True, ensure_ascii=False)
            # sets -> sorted list
            if isinstance(v, set):
                return json.dumps(sorted(list(v)), ensure_ascii=False)
            return str(v)
        except Exception:
            return str(v)

    # Categorical / other
    cat_cols = [c for c in df.columns if c not in num_cols]
    if cat_cols:
        rows = []
        for c in cat_cols:
            s = df[c]
            missing = int(s.isna().sum())
            n = int(len(s))

            # Convert to a safe, hashable representation for nunique + top
            s_safe = s.dropna().map(_stable_str)

            nunique = int(s_safe.nunique(dropna=True))

            top = s_safe.value_counts().head(1)
            top_val = top.index[0] if len(top) else None
            top_cnt = int(top.iloc[0]) if len(top) else 0

            rows.append(
                {
                    "column": c,
                    "n": n,
                    "missing": missing,
                    "missing_pct": round((missing / n * 100.0), 2) if n else 0.0,
                    "nunique": nunique,
                    "top": top_val,
                    "top_count": top_cnt,
                }
            )

        cat_stats = pd.DataFrame(rows).set_index("column")
        out["categorical"] = cat_stats
        cat_stats.to_csv(os.path.join(stats_dir, f"{base}_categorical.csv"), index=True)

    logger.info(f"Saved descriptive stats ({list(out.keys())}) -> {stats_dir}")
    return out


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


# -----------------------------------------------------------------------------
# Descriptive summary text (campaign)
# -----------------------------------------------------------------------------
def generate_descriptive_summary_text(
    activities: pd.DataFrame,
    csv_data: Dict[str, pd.DataFrame],
    campaign_metrics: Optional[Dict] = None,
    out_path: Optional[str] = None,
) -> str:
    out_dir = ensure_dir(OUTPUT_VISUALIZATIONS_DIR)
    if out_path is None:
        out_path = os.path.join(out_dir, "descriptive_summary.txt")

    if activities is None or not isinstance(activities, pd.DataFrame) or activities.empty:
        txt = "No activities data available; cannot generate descriptive summary.\n"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(txt)
        return out_path

    df = activities.copy()

    # Ensure datetime + derived fields
    if "createdAt" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["createdAt"]):
        df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce", utc=True)
    if "date" not in df.columns and "createdAt" in df.columns:
        df["date"] = df["createdAt"].dt.floor("D")
    if "hour" not in df.columns and "createdAt" in df.columns:
        df["hour"] = df["createdAt"].dt.hour

    # Total participants: prefer aggregation sheet pid count if available
    total_users = None
    try:
        agg_key = next((k for k in csv_data.keys() if isinstance(k, str) and k.lower() == "aggregation"), None)
        if agg_key:
            agg = csv_data[agg_key]
            if isinstance(agg, pd.DataFrame) and not agg.empty:
                pid_col = next((c for c in agg.columns if isinstance(c, str) and c.lower() == "pid"), None)
                if pid_col:
                    total_users = int(pd.Series(agg[pid_col]).nunique())
    except Exception:
        total_users = None

    users_in_activities = int(df["pid"].nunique()) if "pid" in df.columns else 0
    if total_users is None or total_users <= 0:
        total_users = users_in_activities

    # Points/rewards fields
    if "points" not in df.columns and "rewardedParticipations" in df.columns:
        df["points"] = df["rewardedParticipations"].apply(extract_points)

    if "num_rewards" not in df.columns:
        if "rewardedParticipations" in df.columns:
            df["num_rewards"] = df["rewardedParticipations"].apply(lambda x: len(extract_detailed_rewards(x)))
        else:
            df["num_rewards"] = 0

    per_user = df.groupby("pid", dropna=True).agg(
        total_points=("points", "sum"),
        total_rewards=("num_rewards", "sum"),
        active_days=("date", lambda s: pd.Series(s).dropna().nunique()),
    ).reset_index()

    active_users = per_user[(per_user["total_points"] > 0) | (per_user["total_rewards"] > 0)]
    active_n = int(active_users["pid"].nunique())
    passive_n = max(0, int(total_users) - active_n)

    # Active players per day
    apd_mean, apd_sd = (None, None)
    if "pid" in df.columns and "date" in df.columns:
        active_players_per_day = df.dropna(subset=["pid", "date"]).groupby("date")["pid"].nunique()
        if not active_players_per_day.empty:
            apd_mean, apd_sd = _mean_sd(active_players_per_day)

    # Average active days per participant
    aad_mean, aad_sd = _mean_sd(per_user["active_days"])

    # Retention proxy
    retention_mean, retention_sd = (None, None)
    try:
        user_dropout = df.groupby("pid").agg(first=("createdAt", "min"), last=("createdAt", "max")).dropna()
        user_dropout["retention_days"] = (user_dropout["last"] - user_dropout["first"]).dt.days
        retention_mean, retention_sd = _mean_sd(user_dropout["retention_days"])
    except Exception:
        pass

    # Retention rate per week (wave)
    retention_by_wave_lines: List[str] = []
    if "wave" in df.columns and "pid" in df.columns:
        try:
            wave_active = df.dropna(subset=["pid", "wave"]).groupby("wave")["pid"].nunique().sort_index()
            for w, n in wave_active.items():
                retention_by_wave_lines.append(
                    f"  - Week {int(w)}: {int(n)} active users ({_fmt_pct(int(n), int(total_users))})"
                )
        except Exception:
            pass

    # Usage by day of week
    dow_lines: List[str] = []
    if "date" in df.columns:
        try:
            dow = pd.to_datetime(df["date"], errors="coerce").dt.day_name()
            dow_counts = dow.value_counts()
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            dow_counts = dow_counts.reindex(day_order).fillna(0).astype(int)
            total_acts = int(dow_counts.sum())
            for day in day_order:
                n = int(dow_counts.loc[day])
                dow_lines.append(f"  - {day}: {n} ({_fmt_pct(n, total_acts)})")
        except Exception:
            pass

    # Usage by hour buckets (FIXED LABELS)
    hour_bucket_lines: List[str] = []
    peak_hour_line = "N/A"
    if "hour" in df.columns:
        try:
            hours = pd.to_numeric(df["hour"], errors="coerce").dropna().astype(int)
            if not hours.empty:
                peak_hour = int(hours.value_counts().idxmax())
                peak_hour_line = f"{peak_hour:02d}:00"

                buckets = hours.apply(_bucket_hour).value_counts()
                bucket_order = [
                    "morning (6AM-11AM)",
                    "afternoon (12PM-5PM)",
                    "evening (6PM-11PM)",
                    "night (12AM-5AM)",
                ]
                buckets = buckets.reindex(bucket_order).fillna(0).astype(int)
                total = int(buckets.sum())
                for b in bucket_order:
                    n = int(buckets.loc[b])
                    hour_bucket_lines.append(f"  - {b}: {n} ({_fmt_pct(n, total)})")
        except Exception:
            pass

    # Activities distribution
    type_lines: List[str] = []
    if "type" in df.columns:
        type_counts = df["type"].astype(str).value_counts()
        total = int(type_counts.sum())
        for t, n in type_counts.items():
            type_lines.append(f"  - {t}: {int(n)} ({_fmt_pct(int(n), total)})")

    # Activities per player mean/sd
    acts_per_player = df.groupby("pid").size() if "pid" in df.columns else pd.Series(dtype=float)
    acts_pp_mean, acts_pp_sd = _mean_sd(acts_per_player)

    # Points per player mean/sd
    points_per_player = (
        df.groupby("pid")["points"].sum() if ("pid" in df.columns and "points" in df.columns) else pd.Series(dtype=float)
    )
    ppp_mean, ppp_sd = _mean_sd(points_per_player)

    # Points by activity type
    points_by_type_lines: List[str] = []
    if "type" in df.columns and "points" in df.columns:
        pbt = df.groupby("type")["points"].sum().sort_values(ascending=False)
        total = float(pbt.sum()) if not pbt.empty else 0.0
        for t, pts in pbt.items():
            pct = (float(pts) / total * 100.0) if total > 0 else 0.0
            points_by_type_lines.append(f"  - {t}: {float(pts):.0f} points ({pct:.1f}%)")

    # Tasks by provider (if desc_tasks exists)
    provider_lines: List[str] = []
    try:
        if "desc_tasks" in csv_data and isinstance(csv_data["desc_tasks"], pd.DataFrame) and not csv_data["desc_tasks"].empty:
            tasks_desc_df = csv_data["desc_tasks"].copy()
            col_map = {c.lower(): c for c in tasks_desc_df.columns}
            name_col = col_map.get("name") or col_map.get("label") or col_map.get("rule") or col_map.get("task") or col_map.get("task_name")
            providers_col = col_map.get("dataproviders")

            if name_col and providers_col and "rewardedParticipations" in df.columns:
                tmp = df[["pid", "rewardedParticipations"]].copy()
                tmp["detailed_rewards"] = tmp["rewardedParticipations"].apply(extract_detailed_rewards)
                tmp = tmp.explode("detailed_rewards").dropna(subset=["detailed_rewards"])
                tmp = tmp[tmp["detailed_rewards"].apply(lambda x: isinstance(x, dict))]
                tmp["task_name"] = tmp["detailed_rewards"].apply(lambda r: str(r.get("rule")).strip() if r.get("rule") is not None else None)
                tmp = tmp.dropna(subset=["task_name"])

                def _norm(s: str) -> str:
                    return re.sub(r"\s+", " ", str(s)).strip().lower()

                def _parse_providers(val) -> List[str]:
                    if pd.isna(val):
                        return []
                    if isinstance(val, list):
                        return [str(x).strip() for x in val if str(x).strip()]
                    s = str(val).strip()
                    try:
                        js = re.sub("'", '"', s)
                        parsed = json.loads(js) if (js.startswith("[") and js.endswith("]")) else None
                        if isinstance(parsed, list):
                            return [str(x).strip() for x in parsed if str(x).strip()]
                    except Exception:
                        pass
                    parts = re.split(r"[;,]", s)
                    return [p.strip() for p in parts if p.strip()]

                tasks_desc_df["task_name_norm"] = tasks_desc_df[name_col].apply(_norm)
                tasks_desc_df["providers_list"] = tasks_desc_df[providers_col].apply(_parse_providers)
                providers_map = tasks_desc_df.set_index("task_name_norm")["providers_list"].to_dict()

                tmp["task_name_norm"] = tmp["task_name"].apply(_norm)
                tmp["providers_list"] = tmp["task_name_norm"].map(providers_map)
                mapped = tmp.dropna(subset=["providers_list"]).explode("providers_list").rename(columns={"providers_list": "provider"})
                mapped = mapped.dropna(subset=["provider"])

                if not mapped.empty:
                    counts = mapped.groupby("provider").size().sort_values(ascending=False)
                    total = int(counts.sum())
                    for prov, n in counts.items():
                        provider_lines.append(f"  - {prov}: {int(n)} ({_fmt_pct(int(n), total)})")
    except Exception:
        pass

    # Compose text
    lines: List[str] = []
    lines.append("USAGE METRICS")
    lines.append("=============")
    lines.append("Participating users (completing activities):")
    lines.append(f"  - Active users: {active_n}/{int(total_users)} ({_fmt_pct(active_n, int(total_users))})")
    lines.append(f"  - Passive users: {passive_n}/{int(total_users)} ({_fmt_pct(passive_n, int(total_users))})")
    lines.append("")
    lines.append(f"Average active days / participant: {_fmt_mean_sd(aad_mean, aad_sd)} days")
    lines.append(f"Active players per day: {_fmt_mean_sd(apd_mean, apd_sd)} players/day")
    lines.append(f"Average time to first inactivity (proxy = last-first activity): {_fmt_mean_sd(retention_mean, retention_sd)} days")
    lines.append("")
    lines.append("Retention rate / week (% active users among all participants):")
    lines.extend(retention_by_wave_lines if retention_by_wave_lines else ["  - N/A"])
    lines.append("")
    lines.append("Usage time / day of week (% of activities):")
    lines.extend(dow_lines if dow_lines else ["  - N/A"])
    lines.append("")
    lines.append("Usage time / hour of day buckets (% of activities):")
    lines.extend(hour_bucket_lines if hour_bucket_lines else ["  - N/A"])
    lines.append("")
    lines.append(f"Peak activity hour: {peak_hour_line}")
    lines.append("")
    lines.append("INTERACTION METRICS")
    lines.append("===================")
    lines.append("Activities distribution (% per type):")
    lines.extend(type_lines if type_lines else ["  - N/A"])
    lines.append("")
    lines.append("Tasks by data provider (% of task completions):")
    lines.extend(provider_lines if provider_lines else ["  - N/A (requires desc_tasks + rewards->rule mapping)"])
    lines.append("")
    lines.append(f"Average completed activities / participant: {_fmt_mean_sd(acts_pp_mean, acts_pp_sd)} activities")
    lines.append(f"Average rewarded points / participant: {_fmt_mean_sd(ppp_mean, ppp_sd)} points")
    lines.append("")
    lines.append("Rewarded points by activity type (% of total points):")
    lines.extend(points_by_type_lines if points_by_type_lines else ["  - N/A"])
    lines.append("")

    text = "\n".join(lines)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    logger.info(f"Saved descriptive summary text -> {out_path}")
    return out_path


# -----------------------------------------------------------------------------
# Helpers: independent colormap for many categorical series
# -----------------------------------------------------------------------------
def get_independent_colormap(n_colors: int):
    import numpy as np
    from matplotlib.colors import ListedColormap, hsv_to_rgb

    palettes = [
        "tab20", "tab20b", "tab20c", "tab10",
        "Set3", "Accent", "Dark2", "Set2", "Set1", "Pastel1", "Pastel2",
    ]

    colors_list = []
    for name in palettes:
        cmap = plt.get_cmap(name)
        if hasattr(cmap, "colors"):
            colors_list.extend(list(cmap.colors))
        else:
            colors_list.extend([cmap(i) for i in np.linspace(0, 1, 20)])

    # Deduplicate while preserving order
    unique = []
    seen = set()
    for rgba in colors_list:
        key = tuple(round(float(x), 5) for x in rgba)
        if key not in seen:
            seen.add(key)
            unique.append(rgba)

    if n_colors <= len(unique):
        selected = unique[:n_colors]
    else:
        selected = list(unique)
        extra = n_colors - len(unique)
        for i in range(extra):
            h = (i / max(1, extra))
            rgb = hsv_to_rgb([h, 0.75, 0.9])
            selected.append((rgb[0], rgb[1], rgb[2], 1.0))

    return ListedColormap(selected, name="independent")


INDEPENDENT_COLORMAP = get_independent_colormap

# -----------------------------------------------------------------------------
# Activities analysis (core)
# -----------------------------------------------------------------------------
def analyze_activities(
    csv_data: Dict[str, pd.DataFrame],
    json_data: Optional[Dict[str, Union[pd.DataFrame, Dict]]] = None,
) -> Optional[Tuple[pd.DataFrame, int, Optional[Dict], Optional[Dict], Optional[Dict]]]:
    if "activities" not in csv_data:
        logger.error("Activities data not found in the provided data (missing key 'activities')")
        return None

    activities = csv_data["activities"].copy()
    if activities.empty:
        logger.error("Activities DataFrame is empty")
        return None

    required_columns = ["createdAt", "pid", "type", "rewardedParticipations"]
    missing_columns = [c for c in required_columns if c not in activities.columns]
    if missing_columns:
        logger.error(f"Activities DataFrame missing required columns: {missing_columns}")
        return None

    # Preprocess timestamps + derived fields
    try:
        activities["createdAt"] = pd.to_datetime(activities["createdAt"], errors="coerce", utc=True)
        activities["date"] = activities["createdAt"].dt.floor("D")
        activities["hour"] = activities["createdAt"].dt.hour

        start = activities["date"].min()
        if pd.isna(start):
            activities["wave"] = pd.NA
        else:
            activities["wave"] = ((activities["date"] - start).dt.days // 7) + 1

        # Rewards parsing
        activities["points"] = activities["rewardedParticipations"].apply(extract_points)
        activities["detailed_rewards"] = activities["rewardedParticipations"].apply(extract_detailed_rewards)
        activities["num_rewards"] = activities["detailed_rewards"].apply(lambda x: len(x) if isinstance(x, list) else 0)

    except Exception as e:
        logger.error(f"Error processing activities data: {e}")
        return None

    # Active/passive classification by rewards
    active_user_ids: set = set()
    passive_user_ids: List = []
    active_users_count = 0
    passive_users_count = 0

    try:
        per_user = activities.groupby("pid").agg(
            total_rewards=("num_rewards", "sum"),
            total_points=("points", "sum"),
        ).reset_index()

        active_user_ids = set(per_user[(per_user["total_rewards"] > 0) | (per_user["total_points"] > 0)]["pid"].tolist())

        # Enrolled users: prefer aggregation sheet pid; otherwise union of any pid-like columns; fallback to activities
        enrolled_user_ids: set = set()
        for key, df in csv_data.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            try:
                cols_map = {c.lower(): c for c in df.columns}
                pid_col = cols_map.get("pid") or cols_map.get("playerid")
                if pid_col and pid_col in df.columns:
                    enrolled_user_ids.update(pd.Series(df[pid_col]).dropna().unique().tolist())
            except Exception:
                pass
        if not enrolled_user_ids:
            enrolled_user_ids = set(activities["pid"].dropna().unique().tolist())

        passive_user_ids = sorted(list(enrolled_user_ids - active_user_ids))
        active_users_count = len(active_user_ids)
        passive_users_count = len(passive_user_ids)

        engagement_map = {pid: ("Active" if pid in active_user_ids else "Passive") for pid in activities["pid"].dropna().unique()}
        activities["engagement_by_rewards"] = activities["pid"].map(engagement_map)

        # Pie chart
        total_users_considered = active_users_count + passive_users_count
        if total_users_considered > 0:
            def plot_active_passive_pie():
                labels = [f"Active ({active_users_count})", f"Passive ({passive_users_count})"]
                sizes = [active_users_count, passive_users_count]
                try:
                    cmap = plt.get_cmap(PIE_COLORMAP)
                    colors = [cmap(0), cmap(3)]
                except Exception:
                    colors = None
                wedges, *_ = plt.pie(
                    sizes,
                    labels=labels,
                    autopct="%1.1f%%" if sum(sizes) > 0 else None,
                    colors=colors,
                    startangle=90,
                    wedgeprops={"edgecolor": "white"},
                )
                plt.title("Active vs Passive Players (by rewards)")
                plt.axis("equal")
                legend_labels = [
                    "Active: players with any rewarded activity",
                    "Passive: enrolled players with no rewards recorded",
                ]
                plt.legend(wedges, legend_labels, title="Legend", loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=True)

            create_and_save_figure(
                plot_active_passive_pie,
                os.path.join(OUTPUT_VISUALIZATIONS_DIR, "player_active_vs_passive_pie.png"),
                figsize=(8, 6),
            )

    except Exception as e:
        logger.error(f"Error classifying active/passive users: {e}")

    # Descriptive stats (drop 'properties' if present)
    try:
        activities_for_stats = activities.copy()
        if "properties" in activities_for_stats.columns:
            activities_for_stats = activities_for_stats.drop(columns=["properties"])
        generate_descriptive_stats(activities_for_stats, "Activities Data", None)
        del activities_for_stats
    except Exception as e:
        logger.error(f"Error generating descriptive statistics: {e}")

    # Basic plots: distribution, daily counts, points by type
    try:
        activity_counts = activities["type"].value_counts()
        if not activity_counts.empty:
            def plot_activity_types_distribution():
                ax = activity_counts.plot(kind="bar", colormap=BAR_COLORMAP)
                plt.title("Distribution of Activity Types")
                plt.xlabel("Activity Type")
                plt.ylabel("Count")
                plt.xticks(rotation=45, ha="right")

            create_and_save_figure(
                plot_activity_types_distribution,
                os.path.join(OUTPUT_VISUALIZATIONS_DIR, "activity_types_distribution.png"),
                figsize=(12, 6),
            )
    except Exception as e:
        logger.error(f"Error creating activity types distribution visualization: {e}")

    # Wave comparisons
    try:
        def plot_wave_comparisons():
            fig, axes = plt.subplots(2, 1, figsize=(12, 12))
            wave_activity_counts = activities.groupby("wave").size()
            wave_activity_counts.plot(kind="bar", ax=axes[0], colormap=BAR_COLORMAP)
            axes[0].set_title("Number of Activities per Wave")
            axes[0].set_xlabel("Wave")
            axes[0].set_ylabel("Number of Activities")

            wave_points = activities.groupby("wave")["points"].mean()
            wave_points.plot(kind="bar", ax=axes[1], colormap=BAR_COLORMAP)
            axes[1].set_title("Average Points per Wave")
            axes[1].set_xlabel("Wave")
            axes[1].set_ylabel("Average Points")
            plt.tight_layout()

        create_and_save_figure(
            plot_wave_comparisons,
            os.path.join(OUTPUT_VISUALIZATIONS_DIR, "wave_comparisons.png"),
            figsize=(12, 12),
        )

        # By activity type
        def plot_wave_comparisons_by_activity_type():
            type_wave_counts = pd.crosstab(activities["wave"], activities["type"])
            cmap = INDEPENDENT_COLORMAP(type_wave_counts.shape[1])
            type_wave_counts.plot(kind="bar", figsize=(12, 6), colormap=cmap)
            plt.title("Activities per Wave by Activity Types")
            plt.xlabel("Wave")
            plt.ylabel("Number of Activities")
            plt.legend(title="Activity Type")
            plt.tight_layout()

        create_and_save_figure(
            plot_wave_comparisons_by_activity_type,
            os.path.join(OUTPUT_VISUALIZATIONS_DIR, "wave_comparisons_by_activity_type.png"),
            figsize=(12, 8),
        )

        # Points by wave & type
        def plot_wave_points_by_activity_type():
            wave_type_points = activities.groupby(["wave", "type"])["points"].mean().unstack()
            cmap = INDEPENDENT_COLORMAP(wave_type_points.shape[1])
            wave_type_points.plot(kind="bar", figsize=(12, 6), colormap=cmap)
            plt.title("Average Points per Wave by Activity Types")
            plt.xlabel("Wave")
            plt.ylabel("Average Points")
            plt.legend(title="Activity Type")
            plt.tight_layout()

        create_and_save_figure(
            plot_wave_points_by_activity_type,
            os.path.join(OUTPUT_VISUALIZATIONS_DIR, "wave_points_by_activity_type.png"),
            figsize=(12, 8),
        )

		# By player: heatmaps for counts and avg points
        def plot_wave_comparisons_by_player():
            # Preconditions
            if "wave" not in activities.columns or "pid" not in activities.columns:
                raise ValueError("Missing required columns for wave-by-player plot: wave and pid")

            dfp = activities.dropna(subset=["wave", "pid"]).copy()
            if dfp.empty:
                raise ValueError("No non-null wave/pid records available for wave-by-player plot")

            # Normalize ids for plotting
            dfp["pid_str"] = dfp["pid"].astype(str)

            # Choose Top-N players by overall activity volume (readable plot)
            player_volume = dfp.groupby("pid_str").size().sort_values(ascending=False)
            if player_volume.empty:
                raise ValueError("No players available for wave-by-player plot")

            players = player_volume.index.tolist()

            # Build pivots
            wave_player_counts = (
                dfp.groupby(["wave", "pid_str"]).size().unstack(fill_value=0)
                .reindex(columns=players, fill_value=0)
                .sort_index()
            )

            # Avg points per wave per player
            if "points" not in dfp.columns:
                dfp["points"] = 0
            wave_player_points = (
                dfp.groupby(["wave", "pid_str"])["points"].mean().unstack()
                .reindex(columns=players)
                .sort_index()
                .fillna(0.0)
            )

            if wave_player_counts.values.sum() <= 0:
                raise ValueError("Wave-by-player counts are all zero; skipping plot")

            # Plot
            fig, axes = plt.subplots(2, 1, figsize=(18, 12), sharex=True)

            sns.heatmap(
                wave_player_counts,
                ax=axes[0],
                cmap=SEQUENTIAL_HEATMAP_COLORMAP,
                annot=True,
                fmt="d", # <- counts
                linewidths=0.2,
            )
            axes[0].set_title(f"Activities per Wave by Player ({len(players)} Players)")
            axes[0].set_xlabel("")
            axes[0].set_ylabel("Wave")

            sns.heatmap(
                wave_player_points,
                ax=axes[1],
                cmap=SEQUENTIAL_HEATMAP_COLORMAP,
                annot=True,
                fmt=".1f",
                linewidths=0.2,
            )
            axes[1].set_title("Average Points per Wave by Player")
            axes[1].set_xlabel("Player ID")
            axes[1].set_ylabel("Wave")

            plt.tight_layout()

        create_and_save_figure(
            plot_wave_comparisons_by_player,
            os.path.join(OUTPUT_VISUALIZATIONS_DIR, "wave_comparisons_by_player.png"),
            figsize=(18, 12),
        )


    except Exception as e:
        logger.error(f"Error creating wave comparisons: {e}")

    # Daily activities time series
    try:
        daily_activities = activities.groupby("date").size()
        if not daily_activities.empty:
            def plot_activities_over_time():
                series = daily_activities.sort_index()
                date_labels = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in series.index]
                x_pos = range(len(series))
                plt.bar(x_pos, series.values, width=0.6)
                plt.title("Number of Activities per Day")
                plt.xlabel("Date")
                plt.ylabel("Number of Activities")
                ax = plt.gca()
                ax.set_xticks(list(x_pos))
                ax.set_xticklabels(date_labels)
                ax.tick_params(axis="x", rotation=90)
                plt.margins(x=0.01)

            create_and_save_figure(
                plot_activities_over_time,
                os.path.join(OUTPUT_VISUALIZATIONS_DIR, "activities_over_time.png"),
                figsize=(14, 7),
            )
    except Exception as e:
        logger.error(f"Error creating activities over time visualization: {e}")

    # Points by activity type (total)
    try:
        points_by_type = activities.groupby("type")["points"].sum().sort_values(ascending=False)
        if not points_by_type.empty:
            def plot_points_by_activity_type():
                points_by_type.plot(kind="bar", colormap=BAR_COLORMAP)
                plt.title("Total Points by Activity Type")
                plt.xlabel("Activity Type")
                plt.ylabel("Total Points")
                plt.xticks(rotation=45, ha="right")

            create_and_save_figure(
                plot_points_by_activity_type,
                os.path.join(OUTPUT_VISUALIZATIONS_DIR, "points_by_activity_type.png"),
                figsize=(12, 6),
            )
    except Exception as e:
        logger.error(f"Error creating points by activity type visualization: {e}")

    # Points by player
    try:
        points_by_user = activities.groupby("pid")["points"].sum().sort_values(ascending=False)
        if not points_by_user.empty:
            def plot_points_by_user():
                plot_data = points_by_user
                plt.title("Total Rewarded Points by Player")
                plot_data.plot(kind="bar", colormap=BAR_COLORMAP)
                plt.xlabel("Player ID")
                plt.ylabel("Total Rewarded Points")
                plt.xticks(rotation=45, ha="right")

            create_and_save_figure(
                plot_points_by_user,
                os.path.join(OUTPUT_VISUALIZATIONS_DIR, "points_by_player.png"),
                figsize=(12, 6),
            )
    except Exception as e:
        logger.error(f"Error creating points by user visualization: {e}")

    # Rewards by activity type (mean)
    try:
        rewards_by_type = activities.groupby("type")["points"].mean().sort_values(ascending=False)
        if not rewards_by_type.empty:
            def plot_rewards_by_activity_type():
                rewards_by_type.plot(kind="bar", colormap=BAR_COLORMAP)
                plt.title("Average Rewarded Points by Activity Type")
                plt.xlabel("Activity Type")
                plt.ylabel("Average Rewarded Points")
                plt.xticks(rotation=45, ha="right")

            create_and_save_figure(
                plot_rewards_by_activity_type,
                os.path.join(OUTPUT_VISUALIZATIONS_DIR, "rewards_by_activity_type.png"),
                figsize=(12, 6),
            )
    except Exception as e:
        logger.error(f"Error creating rewards by activity type visualization: {e}")

    # Points over time
    try:
        daily_points = activities.groupby("date")["points"].sum()
        if not daily_points.empty:
            def plot_points_over_time():
                series = daily_points.sort_index()
                date_labels = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in series.index]
                x_pos = range(len(series))
                plt.bar(x_pos, series.values, width=0.6)
                plt.title("Total Rewarded Points per Day")
                plt.xlabel("Date")
                plt.ylabel("Total Rewarded Points")
                ax = plt.gca()
                ax.set_xticks(list(x_pos))
                ax.set_xticklabels(date_labels)
                ax.tick_params(axis="x", rotation=90)
                plt.margins(x=0.01)

            create_and_save_figure(
                plot_points_over_time,
                os.path.join(OUTPUT_VISUALIZATIONS_DIR, "points_over_time.png"),
                figsize=(14, 7),
            )
    except Exception as e:
        logger.error(f"Error creating points over time visualization: {e}")

    # Reward-level analysis by challenge and rule (explode detailed_rewards)
    try:
        rewards_df = activities[["pid", "date", "type", "detailed_rewards"]].copy()
        rewards_df = rewards_df.explode("detailed_rewards")
        rewards_df = rewards_df[rewards_df["detailed_rewards"].apply(lambda x: isinstance(x, dict))]

        if not rewards_df.empty:
            def _extract_reward_points(r):
                try:
                    return int(r.get("points", 0)) if isinstance(r, dict) else 0
                except Exception:
                    return 0

            def _extract_challenge(r):
                name, cid = None, None
                if isinstance(r, dict):
                    ch = r.get("challenge")
                    if isinstance(ch, dict):
                        name = ch.get("name") or ch.get("label")
                        cid = ch.get("xid") or ch.get("id")
                    if name is None:
                        for key in ("challengeName", "name", "label"):
                            if key in r and pd.notna(r.get(key)):
                                name = r.get(key)
                                break
                    if cid is None:
                        for key in ("challengeId", "challenge_id", "cid", "id", "xid"):
                            if key in r and pd.notna(r.get(key)):
                                cid = r.get(key)
                                break
                if isinstance(cid, list) and cid:
                    cid = cid[0]
                return name, cid

            def _extract_rule(r):
                if isinstance(r, dict):
                    val = r.get("rule")
                    if isinstance(val, (str, int, float)):
                        return str(val)
                return None

            rewards_df["reward_points"] = rewards_df["detailed_rewards"].apply(_extract_reward_points)
            rewards_df[["challenge_name", "challenge_id"]] = rewards_df["detailed_rewards"].apply(lambda r: pd.Series(_extract_challenge(r)))
            rewards_df["rule_name"] = rewards_df["detailed_rewards"].apply(_extract_rule)

            rewards_df["challenge_name"] = rewards_df["challenge_name"].fillna("Unknown")
            rewards_df["rule_name"] = rewards_df["rule_name"].fillna("Unknown")

            # Counts by challenge (Top 30)
            challenge_counts = rewards_df["challenge_name"].value_counts().head(30)
            if not challenge_counts.empty:
                def plot_rewards_by_challenge_count():
                    challenge_counts.plot(kind="bar", colormap=BAR_COLORMAP)
                    plt.title("Count of Rewarded Activities by Challenge (Top 30)")
                    plt.xlabel("Challenge Name")
                    plt.ylabel("Count of Rewards")
                    plt.xticks(rotation=45, ha="right")

                create_and_save_figure(
                    plot_rewards_by_challenge_count,
                    os.path.join(OUTPUT_VISUALIZATIONS_DIR, "rewards_count_by_challenge.png"),
                    figsize=(14, 7),
                )

            # Total points by challenge (ALL)
            points_by_challenge = rewards_df.groupby("challenge_name")["reward_points"].sum().sort_values(ascending=False)
            if not points_by_challenge.empty:
                def plot_points_by_challenge():
                    points_by_challenge.plot(kind="bar", colormap=BAR_COLORMAP)
                    plt.title("Total Rewarded Points by Challenge")
                    plt.xlabel("Challenge Name")
                    plt.ylabel("Total Points")
                    plt.xticks(rotation=45, ha="right")

                create_and_save_figure(
                    plot_points_by_challenge,
                    os.path.join(OUTPUT_VISUALIZATIONS_DIR, "points_by_challenge.png"),
                    figsize=(14, 7),
                )

            # Counts by rule (Top 30)
            rule_counts = rewards_df["rule_name"].value_counts().head(30)
            if not rule_counts.empty:
                def plot_rewards_by_rule_count():
                    rule_counts.plot(kind="bar", colormap=BAR_COLORMAP)
                    plt.title("Count of Rewarded Activities by Rule (Top 30)")
                    plt.xlabel("Rule Name")
                    plt.ylabel("Count of Rewards")
                    plt.xticks(rotation=45, ha="right")

                create_and_save_figure(
                    plot_rewards_by_rule_count,
                    os.path.join(OUTPUT_VISUALIZATIONS_DIR, "rewards_count_by_rule.png"),
                    figsize=(14, 7),
                )

            # Total points by rule (ALL)
            points_by_rule = rewards_df.groupby("rule_name")["reward_points"].sum().sort_values(ascending=False)
            if not points_by_rule.empty:
                def plot_points_by_rule():
                    points_by_rule.plot(kind="bar", colormap=BAR_COLORMAP)
                    plt.title("Total Rewarded Points by Rule")
                    plt.xlabel("Rule Name")
                    plt.ylabel("Total Points")
                    plt.xticks(rotation=45, ha="right")

                create_and_save_figure(
                    plot_points_by_rule,
                    os.path.join(OUTPUT_VISUALIZATIONS_DIR, "points_by_rule.png"),
                    figsize=(14, 7),
                )

            # Per-type: points by challenge (barh)
            try:
                out_dir = ensure_dir(os.path.join(OUTPUT_VISUALIZATIONS_DIR, "by_type"))
                for t in rewards_df["type"].dropna().unique().tolist():
                    sub = rewards_df[rewards_df["type"] == t]
                    series = sub.groupby("challenge_name")["reward_points"].sum().sort_values(ascending=False)
                    if series.empty:
                        continue
                    fname = os.path.join(out_dir, f"points_by_challenge_type_{safe_filename(t)}.png")
                    fig_h = compute_barh_fig_height(len(series))

                    def plot_points_by_challenge_for_type(series=series, t=t):
                        series.plot(kind="barh", colormap=BAR_COLORMAP)
                        plt.title(f"Total Rewarded Points by Challenge — Activity Type: {t}")
                        plt.xlabel("Total Points")
                        plt.ylabel("Challenge Name")
                        try:
                            plt.gca().invert_yaxis()
                        except Exception:
                            pass

                    create_and_save_figure(plot_points_by_challenge_for_type, fname, figsize=(14, fig_h))
            except Exception as e:
                logger.error(f"Error creating per-type challenge analyses: {e}")

            # Per-challenge: points by rule (barh)
            try:
                out_dir = ensure_dir(os.path.join(OUTPUT_VISUALIZATIONS_DIR, "by_challenge"))
                for ch in rewards_df["challenge_name"].dropna().unique().tolist():
                    sub = rewards_df[rewards_df["challenge_name"] == ch]
                    series = sub.groupby("rule_name")["reward_points"].sum().sort_values(ascending=False)
                    if series.empty:
                        continue
                    fname = os.path.join(out_dir, f"points_by_rule_challenge_{safe_filename(ch)}.png")
                    fig_h = compute_barh_fig_height(len(series))

                    def plot_points_by_rule_for_challenge(series=series, ch=ch):
                        series.plot(kind="barh", colormap=BAR_COLORMAP)
                        plt.title(f"Total Rewarded Points by Rule — Challenge: {ch}")
                        plt.xlabel("Total Points")
                        plt.ylabel("Rule Name")
                        try:
                            plt.gca().invert_yaxis()
                        except Exception:
                            pass

                    create_and_save_figure(plot_points_by_rule_for_challenge, fname, figsize=(14, fig_h))
            except Exception as e:
                logger.error(f"Error creating per-challenge rule analyses: {e}")

    except Exception as e:
        logger.error(f"Error creating challenge/rule analyses: {e}")

    # User activity distribution
    user_activity = None
    try:
        user_activity = activities.groupby("pid").size().sort_values(ascending=False)
        if not user_activity.empty:
            def plot_user_activity_distribution():
                plot_data = user_activity
                plt.title("Number of Activities by Player")
                plot_data.plot(kind="bar", colormap=BAR_COLORMAP)
                plt.xlabel("Player ID")
                plt.ylabel("Number of Activities")
                plt.xticks(rotation=45, ha="right")

            create_and_save_figure(
                plot_user_activity_distribution,
                os.path.join(OUTPUT_VISUALIZATIONS_DIR, "player_activity_distribution.png"),
                figsize=(12, 6),
            )
    except Exception as e:
        logger.error(f"Error creating user activity distribution visualization: {e}")

    # Heatmap: activity type distribution by player (guarded)
    try:
        if activities["pid"].nunique() >= 2 and activities["type"].nunique() >= 2:
            user_type_counts = pd.crosstab(activities["pid"], activities["type"])
            if not user_type_counts.empty:
                # Normalize by row
                row_sums = user_type_counts.sum(axis=1)
                user_type_counts = user_type_counts[row_sums > 0]
                user_type_counts_norm = user_type_counts.div(user_type_counts.sum(axis=1), axis=0)

                # Limit users + types
                users = user_type_counts_norm.index.tolist()

                heat = user_type_counts_norm.loc[users]
                if heat.shape[1] > MAX_TYPES_HEATMAP:
                    top_types = user_type_counts.sum().nlargest(MAX_TYPES_HEATMAP).index
                    heat = heat[top_types]

                if not heat.empty:
                    def plot_activity_type_by_user():
                        annot_fontsize = 7
                        sns.heatmap(heat, cmap=SEQUENTIAL_HEATMAP_COLORMAP, annot=True, fmt=".2f", linewidths=0.5, annot_kws={"fontsize": annot_fontsize})
                        plt.title(f"Activity Type Distribution by Player ({len(users)} Players)")
                        plt.xlabel("Activity Type")
                        plt.ylabel("Player ID")

                    height = min(20, 8 + max(0, (len(users) - 10) / 5 * 0.5))
                    create_and_save_figure(
                        plot_activity_type_by_user,
                        os.path.join(OUTPUT_VISUALIZATIONS_DIR, "activity_type_by_player.png"),
                        figsize=(14, height),
                    )
    except Exception as e:
        logger.error(f"Error creating activity type by user heatmap: {e}")

    # Campaign metrics
    campaign_metrics: Optional[Dict] = None
    unique_users_count = 0
    campaign_start = None
    campaign_end = None
    campaign_length_days = 0

    try:
        unique_users_count = int(activities["pid"].nunique())
        campaign_start = activities["createdAt"].min()
        campaign_end = activities["createdAt"].max()
        campaign_length_days = int((campaign_end - campaign_start).days) if pd.notna(campaign_start) and pd.notna(campaign_end) else 0

        campaign_name = "GameBus Campaign"
        campaign_abbr = ""

        for key, df in csv_data.items():
            if isinstance(key, str) and key.startswith("desc_") and isinstance(df, pd.DataFrame) and not df.empty and "name" in df.columns:
                try:
                    names = df["name"].dropna()
                    if not names.empty:
                        campaign_name = names.iloc[0]
                        if "abbreviation" in df.columns:
                            row_idx = names.index[0]
                            if row_idx in df.index and not pd.isna(df.loc[row_idx, "abbreviation"]):
                                campaign_abbr = df.loc[row_idx, "abbreviation"]
                        break
                except Exception:
                    pass

        campaign_metrics = {
            "unique_users": unique_users_count,
            "length_days": campaign_length_days,
            "start_date": campaign_start.date() if pd.notna(campaign_start) else None,
            "end_date": campaign_end.date() if pd.notna(campaign_end) else None,
            "name": campaign_name,
            "abbreviation": campaign_abbr,
            "active_users_count": active_users_count,
            "passive_users_count": passive_users_count,
            "passive_user_ids": passive_user_ids,
        }

        try:
            generate_descriptive_summary_text(activities, csv_data, campaign_metrics)
        except Exception as e:
            logger.error(f"Error generating descriptive summary text: {e}")

    except Exception as e:
        logger.error(f"Error calculating campaign metrics: {e}")

    # Dropout metrics (last - first activity, days)
    dropout_metrics: Optional[Dict] = None
    user_dropout = pd.DataFrame()

    try:
        user_dropout = activities.groupby("pid").agg(first_activity=("createdAt", "min"), last_activity=("createdAt", "max")).dropna()
        user_dropout["dropout_days"] = (user_dropout["last_activity"] - user_dropout["first_activity"]).dt.days
        user_dropout.loc[user_dropout["dropout_days"] < 0, "dropout_days"] = 0

        dropout_stats = user_dropout["dropout_days"].describe(percentiles=[0.25, 0.5, 0.75, 0.9]).round(2)
        dropout_metrics = {
            "avg_dropout_days": float(dropout_stats.get("mean", 0.0)),
            "median_dropout_days": float(dropout_stats.get("50%", 0.0)),
            "min_dropout_days": float(dropout_stats.get("min", 0.0)),
            "max_dropout_days": float(dropout_stats.get("max", 0.0)),
        }

        def plot_dropout_rates_histogram():
            sns.histplot(user_dropout["dropout_days"], kde=True, bins=20, color="tab:blue")
            plt.title("Distribution of User Dropout (First to Last Activity)")
            plt.xlabel("Days")
            plt.ylabel("Number of Users")

        create_and_save_figure(
            plot_dropout_rates_histogram,
            os.path.join(OUTPUT_VISUALIZATIONS_DIR, "dropout_rates_distribution.png"),
            figsize=(10, 6),
        )

    except Exception as e:
        logger.error(f"Error calculating dropout metrics: {e}")

    # Joining metrics (first activity - campaign start)
    joining_metrics: Optional[Dict] = None
    try:
        if not user_dropout.empty and campaign_start is not None and pd.notna(campaign_start):
            user_dropout["joining_days"] = (user_dropout["first_activity"] - campaign_start).dt.days
            joining_stats = user_dropout["joining_days"].describe(percentiles=[0.25, 0.5, 0.75, 0.9]).round(2)
            joining_metrics = {
                "avg_joining_days": float(joining_stats.get("mean", 0.0)),
                "median_joining_days": float(joining_stats.get("50%", 0.0)),
                "min_joining_days": float(joining_stats.get("min", 0.0)),
                "max_joining_days": float(joining_stats.get("max", 0.0)),
            }

            def plot_joining_rates_histogram():
                sns.histplot(user_dropout["joining_days"], kde=True, bins=20, color="tab:red")
                plt.title("Distribution of User Joining Rates")
                plt.xlabel("Days Between Campaign Start and First Activity")
                plt.ylabel("Number of Users")

            create_and_save_figure(
                plot_joining_rates_histogram,
                os.path.join(OUTPUT_VISUALIZATIONS_DIR, "joining_rates_distribution.png"),
                figsize=(10, 6),
            )

            def plot_combined_rates():
                plt.figure(figsize=(12, 7))
                sns.kdeplot(user_dropout["dropout_days"], label="Dropout", fill=True, alpha=0.3, color="blue")
                sns.kdeplot(user_dropout["joining_days"], label="Joining", fill=True, alpha=0.3, color="red")
                plt.title("Distribution of Dropout vs Joining Rates")
                plt.xlabel("Days")
                plt.ylabel("Density")
                plt.legend()

            create_and_save_figure(
                plot_combined_rates,
                os.path.join(OUTPUT_VISUALIZATIONS_DIR, "combined_dropout_joining_rates.png"),
                figsize=(12, 7),
            )

            def plot_combined_boxplots():
                df_long = pd.concat(
                    [
                        pd.DataFrame({"days": user_dropout["dropout_days"], "metric": "Dropout"}),
                        pd.DataFrame({"days": user_dropout["joining_days"], "metric": "Joining"}),
                    ],
                    ignore_index=True,
                )
                ax = sns.boxplot(x="metric", y="days", data=df_long, hue="metric", palette=["blue", "red"], dodge=False)
                if ax.legend_:
                    ax.legend_.remove()
                plt.title("Dropout vs Joining (Boxplots)")
                plt.xlabel("")
                plt.ylabel("Days")

            create_and_save_figure(
                plot_combined_boxplots,
                os.path.join(OUTPUT_VISUALIZATIONS_DIR, "combined_dropout_joining_boxplots.png"),
                figsize=(12, 7),
            )

    except Exception as e:
        logger.error(f"Error calculating joining metrics: {e}")

    # Churn rate (30-day inactivity)
    try:
        daily = activities.copy()
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
        daily = daily[daily["pid"].notna()].copy()
        daily["pid_str"] = daily["pid"].astype(str)

        series_start = daily["date"].min()
        series_end = daily["date"].max()
        if pd.isna(series_start) or pd.isna(series_end):
            raise ValueError("Invalid date range for churn computation")

        date_index = pd.date_range(series_start, series_end, freq="D")

        active_pivot = (
            daily.groupby(["date", "pid_str"]).size().unstack(fill_value=0).reindex(index=date_index, fill_value=0)
        )
        if active_pivot.shape[1] == 0:
            raise ValueError("No users for churn computation")

        active_bool = active_pivot > 0

        first_activity_series = daily.groupby("pid_str")["date"].min().reindex(active_bool.columns)
        first_activity_series = pd.to_datetime(first_activity_series)

        idx_vals = active_bool.index.values.astype("datetime64[ns]")
        fa_vals = first_activity_series.values.astype("datetime64[ns]")
        joined_mask = pd.DataFrame(
            idx_vals[:, None] >= fa_vals[None, :],
            index=active_bool.index,
            columns=active_bool.columns,
        )

        active_last_30 = active_bool.rolling(window=30, min_periods=1).sum() > 0
        churned_mask = (~active_last_30) & joined_mask

        churned_count = churned_mask.sum(axis=1)
        joined_count = joined_mask.sum(axis=1)
        denom = joined_count.replace(0, pd.NA).astype("Float64")
        churn_rate = (churned_count.astype("Float64") / denom).fillna(0.0)

        churn_df = pd.DataFrame(
            {
                "date": churn_rate.index,
                "churned_count": churned_count.values,
                "joined_count": joined_count.values,
                "churn_rate": churn_rate.values,
            }
        ).set_index("date")

        def plot_churn_rate_over_time():
            plt.plot(churn_df.index, churn_df["churn_rate"] * 100.0, color="tab:purple", linewidth=2)
            plt.title("Churn Rate Over Time (No activity in last 30 days)")
            plt.xlabel("Date")
            plt.ylabel("Churn Rate (%)")
            plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)

        create_and_save_figure(
            plot_churn_rate_over_time,
            os.path.join(OUTPUT_VISUALIZATIONS_DIR, "churn_rate_over_time.png"),
            figsize=(12, 6),
        )

        def plot_churn_counts_over_time():
            # Two-series line plot: joined vs churned (counts)
            plt.plot(churn_df.index, churn_df["joined_count"], linewidth=2, linestyle="--")
            plt.plot(churn_df.index, churn_df["churned_count"], linewidth=2)

            plt.title("Churn Counts Over Time (Joined vs Churned; 30-day inactivity)")
            plt.xlabel("Date")
            plt.ylabel("Number of Users")
            plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)
            plt.legend(["Joined (cumulative by first activity)", "Churned (inactive last 30 days)"])

        create_and_save_figure(
            plot_churn_counts_over_time,
            os.path.join(OUTPUT_VISUALIZATIONS_DIR, "churn_counts_over_time.png"),
            figsize=(12, 6),
        )


    except Exception as e:
        logger.error(f"Error creating churn rate visualization: {e}")

    # Usage by day of week
    try:
        plot_data = activities.copy()
        plot_data["day_of_week"] = pd.to_datetime(plot_data["date"], errors="coerce").dt.day_name()
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        activities_by_day = plot_data.groupby("day_of_week").size().reindex(day_order).fillna(0).astype(int)

        def plot_usage_by_day_of_week():
            x = range(len(day_order))
            counts = [int(activities_by_day.get(day, 0)) for day in day_order]
            plt.bar(x, counts, color=sns.color_palette("colorblind", 10)[0])
            plt.title("Usage by Day of Week")
            plt.xlabel("Day of Week")
            plt.ylabel("Number of Activities")
            plt.xticks(list(x), day_order, rotation=45, ha="right")
            plt.margins(x=0.01)

        create_and_save_figure(
            plot_usage_by_day_of_week,
            os.path.join(OUTPUT_VISUALIZATIONS_DIR, "usage_by_day_of_week.png"),
            figsize=(10, 6),
        )

    except Exception as e:
        logger.error(f"Error creating day-of-week usage plot: {e}")

    # Heatmap: day-of-week x hour
    try:
        heatmap_data = activities.copy()
        heatmap_data["day_of_week"] = pd.to_datetime(heatmap_data["date"], errors="coerce").dt.day_name()
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        activity_heatmap_data = pd.crosstab(index=heatmap_data["day_of_week"], columns=heatmap_data["hour"]).fillna(0)
        activity_heatmap_data = activity_heatmap_data.reindex(day_order).fillna(0)
        activity_heatmap_data = activity_heatmap_data.reindex(columns=range(24), fill_value=0)

        if activity_heatmap_data.values.sum() > 0:
            def plot_activity_heatmap_by_time():
                sns.heatmap(activity_heatmap_data, cmap=SEQUENTIAL_HEATMAP_COLORMAP, annot=True, fmt=".0f", linewidths=0.5)
                plt.title("Activity Heatmap by Day of Week and Hour of Day")
                plt.xlabel("Hour of Day")
                plt.ylabel("Day of Week")

            create_and_save_figure(
                plot_activity_heatmap_by_time,
                os.path.join(OUTPUT_VISUALIZATIONS_DIR, "activity_heatmap_by_time.png"),
                figsize=(15, 8),
            )
    except Exception as e:
        logger.error(f"Error creating activity heatmap by time: {e}")

    # Stacked bar: top activity types by date (daily)
    try:
        activity_types_by_date = pd.crosstab(index=activities["date"], columns=activities["type"])
        if not activity_types_by_date.empty:
            type_counts = activities["type"].value_counts()
            top_count = min(5, len(type_counts))
            top_types = type_counts.nlargest(top_count).index.tolist()

            filtered = activity_types_by_date[top_types].copy()
            filtered.index = pd.to_datetime(filtered.index, errors="coerce")
            filtered = filtered.sort_index()

            if len(filtered.index) > 0:
                full_range = pd.date_range(filtered.index.min(), filtered.index.max(), freq="D")
                filtered = filtered.reindex(full_range).fillna(0)

            date_labels = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in filtered.index]

            def plot_stacked():
                plt.figure(figsize=PLOT_FIGSIZE_ACTIVITY_TYPES_STACKED)
                cmap = INDEPENDENT_COLORMAP(filtered.shape[1])
                filtered.plot(kind="bar", stacked=True, colormap=cmap, width=1, ax=plt.gca())
                plt.title("Activity Types Distribution by Date (Top 5 Types)")
                plt.xlabel("Date")
                plt.ylabel("Number of Activities")
                plt.legend(title="Activity Type")
                ax = plt.gca()
                ax.set_xticks(range(len(filtered)))
                ax.set_xticklabels(date_labels)
                ax.tick_params(axis="x", rotation=90)
                plt.tight_layout()

            # Save directly (this plot is intentionally very wide)
            out_path = os.path.join(OUTPUT_VISUALIZATIONS_DIR, "activity_types_stacked_by_date.png")
            plot_stacked()
            fig = plt.gcf()
            _apply_label_truncation(fig, LABEL_MAX_CHARS)
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved figure -> {out_path}")

    except Exception as e:
        logger.error(f"Error creating stacked bar chart by date: {e}")

    # User engagement heatmap by day (guarded)
    try:
        user_day = pd.crosstab(index=activities["pid"], columns=activities["date"]).fillna(0)
        if not user_day.empty:
            users = user_day.sum(axis=1).sort_values(ascending=False)

            # limit days
            cols = list(user_day.columns)
            cols_sorted = sorted(cols)
            if len(cols_sorted) > MAX_DAYS_HEATMAP:
                cols_sorted = cols_sorted[-MAX_DAYS_HEATMAP:]

            heat = user_day.loc[users, cols_sorted].copy()
            # Format column labels
            try:
                heat.columns = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in heat.columns]
            except Exception:
                heat.columns = [str(d) for d in heat.columns]

            if heat.values.sum() > 0:
                def plot_user_engagement_heatmap():
                    sns.heatmap(heat, cmap=SEQUENTIAL_HEATMAP_COLORMAP, annot=True, fmt="d", annot_kws={"fontsize": 5}, linewidths=0.2)
                    plt.title(f"Player Engagement Heatmap by Day ({len(users)} Players)")
                    plt.xlabel("Date")
                    plt.ylabel("Player ID")

                create_and_save_figure(
                    plot_user_engagement_heatmap,
                    os.path.join(OUTPUT_VISUALIZATIONS_DIR, "player_engagement_heatmap.png"),
                    figsize=(16, 10),
                )
    except Exception as e:
        logger.error(f"Error creating user engagement heatmap: {e}")

    return activities, unique_users_count, dropout_metrics, joining_metrics, campaign_metrics


# -----------------------------------------------------------------------------
# Visualizations / challenges / tasks analysis (desc_* + rewards mapping)
# -----------------------------------------------------------------------------
def analyze_visualizations_challenges_tasks(csv_data: Dict[str, pd.DataFrame]) -> None:
    required_sheets = ["desc_visualizations", "desc_challenges", "desc_tasks", "activities"]
    missing_sheets = [s for s in required_sheets if s not in csv_data]
    if missing_sheets:
        logger.warning(f"Skipping visualization/challenge/task analysis; missing sheets: {missing_sheets}")
        return

    tasks_df = csv_data["desc_tasks"].copy()
    activities_df = csv_data["activities"].copy()

    # Ensure createdAt fields
    if "createdAt" in activities_df.columns and not pd.api.types.is_datetime64_any_dtype(activities_df["createdAt"]):
        activities_df["createdAt"] = pd.to_datetime(activities_df["createdAt"], errors="coerce", utc=True)
    if "date" not in activities_df.columns and "createdAt" in activities_df.columns:
        activities_df["date"] = activities_df["createdAt"].dt.date

    # Activity completion by type
    try:
        if "type" in activities_df.columns:
            activity_counts = activities_df["type"].value_counts()

            def plot_activity_completion():
                plt.bar(range(len(activity_counts)), activity_counts.values, color=sns.color_palette("colorblind", 10)[0])
                plt.xlabel("Activity Type")
                plt.ylabel("Count")
                plt.title("Activity Completion by Type")
                plt.xticks(range(len(activity_counts)), activity_counts.index, rotation=45, ha="right")
                plt.tight_layout()

            create_and_save_figure(
                plot_activity_completion,
                os.path.join(OUTPUT_VISUALIZATIONS_DIR, "activity_completion.png"),
                figsize=(14, 8),
            )
    except Exception as e:
        logger.error(f"Error creating activity completion plot: {e}")

    # Heatmaps by hour/day
    try:
        if "createdAt" in activities_df.columns and "type" in activities_df.columns:
            activities_df["hour"] = activities_df["createdAt"].dt.hour
            activities_df["day_of_week"] = activities_df["createdAt"].dt.day_name()
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

            activity_hour_counts = pd.crosstab(activities_df["hour"], activities_df["type"]).reindex(range(24), fill_value=0)
            def plot_activity_type_by_hour():
                sns.heatmap(activity_hour_counts, cmap=SEQUENTIAL_HEATMAP_COLORMAP, annot=True, fmt="d", linewidths=0.2)
                plt.title("Activity Types by Hour of Day")
                plt.xlabel("Activity Type")
                plt.ylabel("Hour of Day")

            create_and_save_figure(
                plot_activity_type_by_hour,
                os.path.join(OUTPUT_VISUALIZATIONS_DIR, "activity_type_by_hour.png"),
                figsize=(14, 8),
            )

            activity_day_counts = pd.crosstab(activities_df["day_of_week"], activities_df["type"]).reindex(day_order).fillna(0)
            def plot_activity_type_by_day():
                sns.heatmap(activity_day_counts, cmap=SEQUENTIAL_HEATMAP_COLORMAP, annot=True, fmt="d", linewidths=0.2)
                plt.title("Activity Types by Day of Week")
                plt.xlabel("Activity Type")
                plt.ylabel("Day of Week")

            create_and_save_figure(
                plot_activity_type_by_day,
                os.path.join(OUTPUT_VISUALIZATIONS_DIR, "activity_type_by_day.png"),
                figsize=(14, 8),
            )

    except Exception as e:
        logger.error(f"Error creating hour/day heatmaps: {e}")

    # Active users per day
    try:
        if "pid" in activities_df.columns and "date" in activities_df.columns:
            active_users_per_day = activities_df.groupby("date")["pid"].nunique().sort_index()

            def plot_active_users_per_day():
                series = active_users_per_day
                date_labels = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in series.index]
                x_pos = range(len(series))
                plt.bar(x_pos, series.values, color=sns.color_palette("colorblind", 10)[0], width=0.6)
                plt.title("Number of Active Players per Day")
                plt.xlabel("Date")
                plt.ylabel("Number of Active Players")
                ax = plt.gca()
                ax.set_xticks(list(x_pos))
                ax.set_xticklabels(date_labels)
                ax.tick_params(axis="x", rotation=90)
                plt.margins(x=0.01)

            create_and_save_figure(
                plot_active_users_per_day,
                os.path.join(OUTPUT_VISUALIZATIONS_DIR, "active_players_per_day.png"),
                figsize=(12, 6),
            )
    except Exception as e:
        logger.error(f"Error plotting active users per day: {e}")

    # Completed tasks analysis (rules from rewardedParticipations -> provider mapping)
    try:
        if "rewardedParticipations" not in activities_df.columns or "pid" not in activities_df.columns:
            logger.warning("Skipping completed task plots; missing rewardedParticipations or pid")
            return

        tmp = activities_df[["pid", "type", "date", "rewardedParticipations"]].copy()
        tmp["detailed_rewards"] = tmp["rewardedParticipations"].apply(extract_detailed_rewards)
        tmp = tmp.explode("detailed_rewards").dropna(subset=["detailed_rewards"])
        tmp = tmp[tmp["detailed_rewards"].apply(lambda x: isinstance(x, dict))]

        tmp["task_name"] = tmp["detailed_rewards"].apply(lambda r: str(r.get("rule")).strip() if r.get("rule") is not None else None)
        tmp = tmp.dropna(subset=["task_name"])

        task_completions_df = tmp[["pid", "task_name", "type", "date"]].rename(columns={"type": "activity_type"})
        if task_completions_df.empty:
            logger.warning("No task completion records (rules) found in rewards")
            return

        unique_task_completions = task_completions_df.drop_duplicates(subset=["pid", "task_name"])

        # tasks_by_provider mapping via desc_tasks
        try:
            col_map = {c.lower(): c for c in tasks_df.columns}
            name_col = col_map.get("name") or col_map.get("label") or col_map.get("rule") or col_map.get("task") or col_map.get("task_name")
            providers_col = col_map.get("dataproviders")

            if name_col and providers_col:
                def _norm_name(s):
                    return re.sub(r"\s+", " ", str(s)).strip().lower()

                def _parse_providers(val):
                    if pd.isna(val):
                        return []
                    if isinstance(val, list):
                        return [str(x).strip() for x in val if str(x).strip()]
                    s = str(val).strip()
                    try:
                        js = re.sub("'", '"', s)
                        parsed = json.loads(js) if (js.startswith("[") and js.endswith("]")) else None
                        if isinstance(parsed, list):
                            return [str(x).strip() for x in parsed if str(x).strip()]
                    except Exception:
                        pass
                    parts = re.split(r"[;,]", s)
                    return [p.strip() for p in parts if p.strip()]

                tasks_df["task_name_norm"] = tasks_df[name_col].apply(_norm_name)
                tasks_df["providers_list"] = tasks_df[providers_col].apply(_parse_providers)

                providers_map = (
                    tasks_df.groupby("task_name_norm")["providers_list"]
                    .apply(lambda s: sorted({p for lst in s for p in (lst or [])}))
                    .to_dict()
                )

                task_completions_df["task_name_norm"] = task_completions_df["task_name"].apply(_norm_name)
                task_completions_df["providers_list"] = task_completions_df["task_name_norm"].map(providers_map)

                mapped = task_completions_df.dropna(subset=["providers_list"]).explode("providers_list").rename(columns={"providers_list": "provider"})
                provider_counts = mapped.dropna(subset=["provider"]).groupby("provider").size().sort_values(ascending=False)

                if not provider_counts.empty:
                    def plot_tasks_by_provider():
                        provider_counts.plot(kind="bar", colormap=BAR_COLORMAP)
                        plt.title("Tasks Performed by Users per Data Provider")
                        plt.xlabel("Data Provider")
                        plt.ylabel("Task Completions (reward entries)")
                        plt.xticks(rotation=45, ha="right")

                    create_and_save_figure(
                        plot_tasks_by_provider,
                        os.path.join(OUTPUT_VISUALIZATIONS_DIR, "tasks_by_provider.png"),
                        figsize=(12, 6),
                    )
        except Exception as e:
            logger.error(f"Error creating tasks_by_provider: {e}")

        # tasks completed per day (unique user-task per day)
        try:
            if "date" in task_completions_df.columns:
                per_day = task_completions_df.dropna(subset=["date"]).drop_duplicates(subset=["pid", "task_name", "date"]).groupby("date").size().sort_index()

                def plot_tasks_completed_per_day():
                    series = per_day
                    date_labels = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in series.index]
                    x_pos = range(len(series))
                    plt.bar(x_pos, series.values, color=sns.color_palette("colorblind", 10)[0], width=0.6)
                    plt.title("Tasks Completed per Day")
                    plt.xlabel("Date")
                    plt.ylabel("Tasks Completed (unique user-task)")
                    ax = plt.gca()
                    ax.set_xticks(list(x_pos))
                    ax.set_xticklabels(date_labels)
                    ax.tick_params(axis="x", rotation=90)
                    plt.margins(x=0.01)

                create_and_save_figure(
                    plot_tasks_completed_per_day,
                    os.path.join(OUTPUT_VISUALIZATIONS_DIR, "tasks_completed_per_day.png"),
                    figsize=(12, 6),
                )
        except Exception as e:
            logger.error(f"Error creating tasks_completed_per_day plot: {e}")

        # tasks completed per user (unique tasks)
        try:
            per_user = unique_task_completions.groupby("pid").size().sort_values(ascending=False)
            if not per_user.empty:
                def plot_tasks_completed_per_user():
                    plot_data = per_user.head(50) if len(per_user) > 50 else per_user
                    plt.title("Tasks Completed per Player (Top 50)" if len(per_user) > 50 else "Tasks Completed per Player")
                    plot_data.plot(kind="bar", colormap=BAR_COLORMAP)
                    plt.xlabel("Player ID")
                    plt.ylabel("Tasks Completed (unique tasks)")
                    plt.xticks(rotation=45, ha="right")

                create_and_save_figure(
                    plot_tasks_completed_per_user,
                    os.path.join(OUTPUT_VISUALIZATIONS_DIR, "tasks_completed_per_player.png"),
                    figsize=(12, 6),
                )
        except Exception as e:
            logger.error(f"Error creating tasks_completed_per_player plot: {e}")

    except Exception as e:
        logger.error(f"Error generating completed tasks analysis: {e}")


# -----------------------------------------------------------------------------
# Geofence analysis (FIXED: indentation + robust numeric handling)
# -----------------------------------------------------------------------------
def analyze_geofence_data(json_data: Dict[str, Union[pd.DataFrame, Dict]]) -> None:
    geofence_dfs = []

    for key, data in json_data.items():
        if "geofence" not in key.lower():
            continue
        if not isinstance(data, pd.DataFrame):
            continue

        # Extract player id from key (supports player_123_geofence or user_123_geofence)
        try:
            parts = key.split("_")
            if len(parts) >= 3 and parts[0] in ("player", "user"):
                player_id = parts[1]
            else:
                logger.warning(f"Key {key} does not match expected geofence naming; skipping")
                continue

            df = data.copy()
            df["player_id"] = player_id
            df["user_id"] = player_id  # backward compatibility

            required_columns = ["LATITUDE", "LONGITUDE", "ALTITUDE", "SPEED", "ERROR", "TIMESTAMP"]
            for col in required_columns:
                if col not in df.columns:
                    df[col] = pd.NaT if col == "TIMESTAMP" else 0

            geofence_dfs.append(df)
        except Exception as e:
            logger.warning(f"Error extracting player id from {key}: {e}")

    if not geofence_dfs:
        logger.warning("No geofence data found")
        return

    all_geofence_data = pd.concat(geofence_dfs, ignore_index=True)

    if "TIMESTAMP" in all_geofence_data.columns:
        all_geofence_data["TIMESTAMP"] = pd.to_datetime(all_geofence_data["TIMESTAMP"], errors="coerce")

    for col in ["LATITUDE", "LONGITUDE", "ALTITUDE", "SPEED", "ERROR"]:
        if col in all_geofence_data.columns:
            all_geofence_data[col] = pd.to_numeric(all_geofence_data[col], errors="coerce")

    speed_series = all_geofence_data["SPEED"].dropna()
    if speed_series.empty:
        logger.warning("No valid SPEED values in geofence data; skipping geofence analysis")
        return

    # Outlier trimming (3*IQR)
    Q1 = speed_series.quantile(0.25)
    Q3 = speed_series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR

    filtered = all_geofence_data[(all_geofence_data["SPEED"] >= lower_bound) & (all_geofence_data["SPEED"] <= upper_bound)].copy()
    logger.info(f"Geofence outliers removed: {len(all_geofence_data) - len(filtered)}")

    # Temporal analysis
    if "TIMESTAMP" in filtered.columns and not filtered["TIMESTAMP"].isna().all():
        filtered["hour_of_day"] = filtered["TIMESTAMP"].dt.hour
        filtered["hour_of_day"] = pd.Categorical(filtered["hour_of_day"], categories=list(range(24)), ordered=True)

        def plot_hourly_activity():
            hourly_counts = filtered["hour_of_day"].value_counts().sort_index().reindex(range(24), fill_value=0)
            sns.barplot(x=list(hourly_counts.index), y=list(hourly_counts.values))
            plt.title("Geofence Activity by Hour of Day")
            plt.xlabel("Hour of Day")
            plt.ylabel("Number of Records")
            plt.xticks(range(0, 24))
            plt.xlim(-0.5, 23.5)
            plt.grid(True, alpha=0.3)

        create_and_save_figure(
            plot_hourly_activity,
            os.path.join(OUTPUT_VISUALIZATIONS_DIR, "geofence_hourly_activity.png"),
            figsize=(12, 6),
        )

        def plot_speed_by_hour():
            sns.boxplot(data=filtered, x="hour_of_day", y="SPEED", order=list(range(24)))
            plt.title("Speed by Hour of Day")
            plt.xlabel("Hour of Day")
            plt.ylabel("Speed")
            plt.xticks(range(0, 24))
            plt.xlim(-0.5, 23.5)
            plt.grid(True, alpha=0.3)

        create_and_save_figure(
            plot_speed_by_hour,
            os.path.join(OUTPUT_VISUALIZATIONS_DIR, "geofence_speed_by_hour.png"),
            figsize=(12, 6),
        )

        # Trajectory plot (color by speed)
        try:
            trajectory_data = filtered.sort_values("TIMESTAMP").dropna(subset=["LONGITUDE", "LATITUDE"])

            def plot_movement_trajectory():
                plt.plot(trajectory_data["LONGITUDE"], trajectory_data["LATITUDE"], "b-", alpha=0.7)
                sc = plt.scatter(
                    trajectory_data["LONGITUDE"],
                    trajectory_data["LATITUDE"],
                    c=trajectory_data["SPEED"],
                    cmap="viridis",
                    alpha=0.8,
                )
                plt.colorbar(sc, label="Speed")
                plt.title("Movement Trajectory (Color indicates Speed)")
                plt.xlabel("Longitude")
                plt.ylabel("Latitude")
                plt.grid(True, alpha=0.3)

            create_and_save_figure(
                plot_movement_trajectory,
                os.path.join(OUTPUT_VISUALIZATIONS_DIR, "geofence_movement_trajectory.png"),
                figsize=(12, 8),
            )
        except Exception as e:
            logger.warning(f"Could not create movement trajectory plot: {e}")

    # 3D visualization (FIXED)
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        def plot_3d_visualization():
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection="3d")
            scatter = ax.scatter(
                filtered["LONGITUDE"],
                filtered["LATITUDE"],
                filtered["ALTITUDE"],
                c=filtered["SPEED"],
                cmap="viridis",
                alpha=0.6,
            )
            plt.colorbar(scatter, label="Speed")
            ax.set_title("3D Geofence Visualization")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_zlabel("Altitude")

        create_and_save_figure(
            plot_3d_visualization,
            os.path.join(OUTPUT_VISUALIZATIONS_DIR, "geofence_3d_visualization.png"),
            figsize=(12, 10),
        )

    except Exception as e:
        logger.warning(f"Could not create 3D visualization: {e}")


# -----------------------------------------------------------------------------
# Daily steps analysis (DAY_AGGREGATE + variants)
# -----------------------------------------------------------------------------
def analyze_day_aggregate_steps(
    json_data: Dict[str, Union[pd.DataFrame, Dict]],
    campaign_metrics: Optional[Dict] = None,
) -> Optional[pd.DataFrame]:
    try:
        records: List[pd.DataFrame] = []

        def _extract_from_df(df: pd.DataFrame, user_id: str) -> Optional[pd.DataFrame]:
            if df is None or df.empty:
                return None
            local = df.copy()

            date_candidates = ["X_DATE", "DATE", "START_DATE", "END_DATE"]
            date_col = next((c for c in date_candidates if c in local.columns), None)
            if date_col is None:
                return None

            local["__date"] = pd.to_datetime(local[date_col], errors="coerce").dt.date
            if local["__date"].isna().all() and "END_DATE" in local.columns and date_col != "END_DATE":
                local["__date"] = pd.to_datetime(local["END_DATE"], errors="coerce").dt.date

            steps_candidates = ["STEPS_SUM", "STEP_SUM", "STEPS"]
            steps_col = next((c for c in steps_candidates if c in local.columns), None)
            if steps_col is None:
                return None

            out = local[["__date", steps_col]].dropna(subset=["__date"]).rename(columns={steps_col: "steps"})
            out["steps"] = pd.to_numeric(out["steps"], errors="coerce").fillna(0)
            if out.empty:
                return None
            out["user_id"] = str(user_id)
            return out

        for key, data in json_data.items():
            k_lower = key.lower()
            m = re.match(r"(?:player|user)_(\d+)_", key)
            user_id = m.group(1) if m else key

            if isinstance(data, pd.DataFrame):
                if "day_aggregate" in k_lower:
                    df_out = _extract_from_df(data, user_id)
                    if df_out is not None:
                        records.append(df_out)

            elif isinstance(data, dict) and (k_lower.startswith("player_") or k_lower.startswith("user_")) and k_lower.endswith("_all_data"):
                for section in ["day_aggregate", "day_aggregate_walk", "day_aggregate_run"]:
                    section_data = data.get(section)
                    if isinstance(section_data, list) and section_data:
                        try:
                            df_section = pd.DataFrame(section_data)
                            df_out = _extract_from_df(df_section, user_id)
                            if df_out is not None:
                                records.append(df_out)
                        except Exception as e:
                            logger.warning(f"Failed parsing section '{section}' for {key}: {e}")

        if not records:
            logger.warning("No DAY_AGGREGATE step data found; skipping steps plots")
            return None

        combined = pd.concat(records, ignore_index=True)
        combined = combined.groupby(["__date", "user_id"], as_index=False)["steps"].sum().rename(columns={"__date": "date"})
        combined["steps"] = pd.to_numeric(combined["steps"], errors="coerce").fillna(0)
        combined = combined.sort_values(["date", "user_id"])

        inferred = combined.groupby("date", as_index=False)["steps"].sum()

        def _parse_ts(val):
            return pd.to_datetime(val, errors="coerce")

        min_ts = _parse_ts(inferred["date"]).min()
        max_ts = _parse_ts(inferred["date"]).max()

        start_ts = _parse_ts((campaign_metrics or {}).get("start_date")) if campaign_metrics else pd.NaT
        end_ts = _parse_ts((campaign_metrics or {}).get("end_date")) if campaign_metrics else pd.NaT
        if pd.isna(start_ts):
            start_ts = min_ts
        if pd.isna(end_ts):
            end_ts = max_ts
        if pd.isna(start_ts) or pd.isna(end_ts) or start_ts > end_ts:
            start_ts, end_ts = min_ts, max_ts

        full_range = pd.date_range(start=start_ts.normalize(), end=end_ts.normalize(), freq="D")

        total = inferred.copy()
        total["date"] = pd.to_datetime(total["date"], errors="coerce")
        total = total.set_index("date").reindex(full_range).fillna(0).rename_axis("date").reset_index()

        def plot_total():
            ax = plt.gca()
            ax.plot(total["date"], total["steps"], marker="o")
            ax.set_title("Steps Trend (All Users)")
            ax.set_xlabel("Date")
            ax.set_ylabel("Steps")
            ticks = pd.to_datetime(list(full_range))
            labels = [d.strftime("%Y-%m-%d") for d in ticks]
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=90, ha="center")
            ax.set_xlim(ticks[0], ticks[-1])

        create_and_save_figure(
            plot_total,
            os.path.join(OUTPUT_VISUALIZATIONS_DIR, "steps_trend.png"),
            figsize=PLOT_FIGSIZE_ACTIVITY_TYPES_STACKED,
        )

        return combined

    except Exception as e:
        logger.error(f"Error analyzing DAY_AGGREGATE steps: {e}")
        return None


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------
def generate_analysis_report(
    csv_data: Dict[str, pd.DataFrame],
    json_data: Dict[str, Union[pd.DataFrame, Dict]],
    activities: Optional[pd.DataFrame],
    campaign_metrics: Optional[Dict],
    dropout_metrics: Optional[Dict],
    joining_metrics: Optional[Dict] = None,
) -> str:
    data_analysis_dir = ensure_dir(os.path.join(PROJECT_ROOT, "data_analysis"))
    report_path = os.path.join(data_analysis_dir, "analysis_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("GameBus Data Analysis Report\n")
        f.write("===========================\n\n")

        f.write("1. Data Loading\n")
        f.write("---------------\n")
        if csv_data:
            f.write(f"✓ Excel data loaded successfully: {len(csv_data)} sheets\n")
        else:
            f.write("✗ No Excel data loaded\n")

        if json_data:
            f.write(f"✓ JSON data loaded successfully: {len(json_data)} files\n")
        else:
            f.write("✗ No JSON data loaded\n")

        f.write("\n2. Campaign Summary\n")
        f.write("------------------\n")
        cm = campaign_metrics or {}
        name = cm.get("name", "GameBus Campaign")
        abbr = cm.get("abbreviation", "")
        start_date = cm.get("start_date", "Unknown")
        end_date = cm.get("end_date", "Unknown")
        length_days = cm.get("length_days", 0)
        unique_users = cm.get("unique_users", 0)

        f.write(f"{name} ({abbr})\n\n" if abbr else f"{name}\n\n")
        f.write(f"Start Date: {start_date}\n")
        f.write(f"End Date: {end_date}\n")
        f.write(f"Length: {length_days} days\n")

        # total players from aggregation sheet if possible
        agg_total_users = None
        try:
            agg_key = next((k for k in csv_data.keys() if isinstance(k, str) and k.lower() == "aggregation"), None)
            if agg_key:
                df_agg = csv_data[agg_key]
                if isinstance(df_agg, pd.DataFrame) and not df_agg.empty:
                    pid_col = next((c for c in df_agg.columns if isinstance(c, str) and c.lower() == "pid"), None)
                    if pid_col:
                        agg_total_users = int(pd.Series(df_agg[pid_col]).nunique())
        except Exception:
            pass

        total_users_for_report = agg_total_users if isinstance(agg_total_users, int) and agg_total_users > 0 else unique_users
        f.write(f"Total number of Players in the Campaign: {total_users_for_report}\n")

        active_users_count = cm.get("active_users_count", None)
        passive_users_count = cm.get("passive_users_count", None)
        if active_users_count is not None and passive_users_count is not None:
            f.write(f"Active Players (rewarded): {active_users_count}\n")
            f.write(f"Passive Players (enrolled, never rewarded): {passive_users_count}\n")
            passive_ids = cm.get("passive_user_ids", [])
            if isinstance(passive_ids, list) and passive_ids:
                preview = ", ".join(map(str, passive_ids[:10]))
                f.write(f"Passive Player IDs (first 10): {preview}\n")

        f.write("\n3. Dropout Metrics\n")
        f.write("-----------------\n")
        dm = dropout_metrics or {}
        if dm:
            f.write(f"Average (days): {dm.get('avg_dropout_days', 'N/A')}\n")
            f.write(f"Median (days): {dm.get('median_dropout_days', 'N/A')}\n")
            f.write(f"Min (days): {dm.get('min_dropout_days', 'N/A')}\n")
            f.write(f"Max (days): {dm.get('max_dropout_days', 'N/A')}\n")
        else:
            f.write("N/A\n")

        f.write("\n4. Joining Metrics\n")
        f.write("-----------------\n")
        jm = joining_metrics or {}
        if jm:
            f.write(f"Average (days): {jm.get('avg_joining_days', 'N/A')}\n")
            f.write(f"Median (days): {jm.get('median_joining_days', 'N/A')}\n")
            f.write(f"Min (days): {jm.get('min_joining_days', 'N/A')}\n")
            f.write(f"Max (days): {jm.get('max_joining_days', 'N/A')}\n")
        else:
            f.write("N/A\n")

    logger.info(f"Saved analysis report -> {report_path}")
    return report_path


def create_complete_report(
    csv_data: Dict[str, pd.DataFrame],
    json_data: Dict[str, Union[pd.DataFrame, Dict]],
    activities_result: Optional[Union[pd.DataFrame, Tuple]],
) -> str:
    acts = None
    campaign_metrics = None
    dropout_metrics = None
    joining_metrics = None

    if isinstance(activities_result, tuple) and len(activities_result) >= 5:
        acts, _unique_users_count, dropout_metrics, joining_metrics, campaign_metrics = activities_result
    else:
        acts = activities_result if isinstance(activities_result, pd.DataFrame) else None

    return generate_analysis_report(csv_data, json_data, acts, campaign_metrics, dropout_metrics, joining_metrics)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    try:
        ensure_output_dirs()

        # Load data
        try:
            csv_data = load_excel_files()
            logger.info(f"Loaded {len(csv_data)} Excel sheets" if csv_data else "No Excel data loaded")
        except Exception as e:
            logger.error(f"Error loading Excel files: {e}")
            csv_data = {}

        try:
            json_data = load_json_files()
            logger.info(f"Loaded {len(json_data)} JSON files" if json_data else "No JSON data loaded")
        except Exception as e:
            logger.error(f"Error loading JSON files: {e}")
            json_data = {}

        # Analyze activities
        try:
            result = analyze_activities(csv_data, json_data)
            if result:
                activities, unique_users_count, dropout_metrics, joining_metrics, campaign_metrics = result
                logger.info(f"Activities analyzed: {unique_users_count} unique active users")
            else:
                result = None
                activities = None
                campaign_metrics = None
        except Exception as e:
            logger.error(f"Error analyzing activities data: {e}")
            result = None
            activities = None
            campaign_metrics = None

        # Analyze geofence
        try:
            analyze_geofence_data(json_data)
        except Exception as e:
            logger.error(f"Error analyzing geofence data: {e}")

        # Analyze daily steps
        try:
            analyze_day_aggregate_steps(json_data, campaign_metrics)
        except Exception as e:
            logger.error(f"Error analyzing daily steps: {e}")

        # Analyze desc_visualizations / challenges / tasks
        try:
            analyze_visualizations_challenges_tasks(csv_data)
        except Exception as e:
            logger.error(f"Error analyzing visualizations/challenges/tasks: {e}")

        # Report
        try:
            report_path = create_complete_report(csv_data, json_data, result if result else activities)
            logger.info(f"Report written to: {report_path}")
        except Exception as e:
            logger.error(f"Error generating report: {e}")

        print("\nAnalysis complete.")
        print(f"- Visualizations and outputs saved to: '{OUTPUT_VISUALIZATIONS_DIR}'")

    except Exception as e:
        logger.error(f"Unexpected error in main(): {e}")
        print("An error occurred during analysis. See log file for details.")


if __name__ == "__main__":
    main()
