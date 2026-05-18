# -----------------------------------------------------------------------------
# Path setup (project root + config import)
# -----------------------------------------------------------------------------
# Add the project root to the Python path to find the config module
import os
import sys
import re
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Callable, Tuple, Optional


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


WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

HOUR_BUCKET_LABELS = [
    "00:00-05:59",
    "06:00-11:59",
    "12:00-17:59",
    "18:00-23:59",
]

HOUR_BUCKET_BINS = [-1, 5, 11, 17, 23]


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("gamebus.analysis")

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



def build_scoped_plot_filename(base_name: str, scope_label: Optional[str], ext: str = ".png") -> str:
    """
    Build a scope-specific filename so per-scope plots do not overwrite each other.

    Examples:
    - build_scoped_plot_filename("usage_by_day_of_week", "GameBus")
      -> "usage_by_day_of_week_gamebus.png"
    - build_scoped_plot_filename("active_players_per_day", "Combined")
      -> "active_players_per_day_combined.png"
    """
    base_root, base_ext = os.path.splitext(base_name)
    if base_ext:
        ext = base_ext
    if not ext.startswith("."):
        ext = f".{ext}"

    scope_slug = safe_filename(str(scope_label).strip().lower() if scope_label else "combined")
    base_slug = safe_filename(base_root or base_name)

    return f"{base_slug}_{scope_slug}{ext}"

# -------------------
# Plotting helpers
# ----------------

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
    bottom_adjust: Optional[float] = None,
    apply_label_truncation: bool = True,
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

    if apply_label_truncation:
        _apply_label_truncation(fig, LABEL_MAX_CHARS)

    try:
        if bottom_adjust is not None:
            fig.subplots_adjust(bottom=bottom_adjust)
    except Exception:
        pass

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
# numeric/text helpers
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
    try:
        hour = int(h)
    except Exception:
        return HOUR_BUCKET_LABELS[0]

    if 6 <= hour <= 11:
        return HOUR_BUCKET_LABELS[1]
    if 12 <= hour <= 17:
        return HOUR_BUCKET_LABELS[2]
    if 18 <= hour <= 23:
        return HOUR_BUCKET_LABELS[3]
    return HOUR_BUCKET_LABELS[0]
