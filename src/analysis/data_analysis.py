"""
GameBus Health Behavior Mining Data Analysis Script

This script analyzes data from the GameBus Health Behavior Mining project, including:
1. Excel files in the config directory containing campaign, and activity data
2. JSON files in the data_raw directory containing user-specific data like activity types, and mood logs

The script generates:
1. Descriptive statistics tables for each variable, providing insights into:
   - Central tendency (mean, median)
   - Dispersion (standard deviation, min, max)
   - Distribution characteristics (quartiles, skewness)

2. Various visualizations to provide insights into:
   - Activity patterns and distributions
   - User engagement and participation
   - Movement types and physical activity metrics

3. Bivariate analyses to explore relationships between variables:
   - Correlation analyses
   - Cross-tabulations for categorical variables
   - Relationship visualizations

All visualizations are saved to the 'data_analysis/visualizations' directory.
Descriptive statistics tables are printed to the console and saved to the 'data_analysis/statistics' directory.

Usage:
    python -m src.analysis.data_analysis

Requirements:
    - pandas
    - numpy
    - matplotlib
    - seaborn
    - json
    - scipy (for statistical tests)
"""

import os
import sys
import pandas as pd
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import logging
from scipy import stats
from typing import Callable, Tuple, Dict, Optional, Union, List

# Add the project root to the Python path to find the config module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config.paths import PROJECT_ROOT

# Set up logging
# Create logs directory if it doesn't exist
logs_dir = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(logs_dir, exist_ok=True)

# Configure logger for data analysis only
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Clear any existing handlers to avoid duplicate logs
if logger.hasHandlers():
    logger.handlers.clear()

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add console handler (only warnings and above to console)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Add file handler for analysis logs
file_handler = logging.FileHandler(os.path.join(logs_dir, 'data_analysis.log'))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Prevent propagation to root logger to avoid duplicate logs
logger.propagate = False

# Set style for plots
plt.style.use('ggplot')
sns.set(style="whitegrid")
sns.set_palette("colorblind")

# Colormaps for different types of visualizations
BAR_COLORMAP = 'tab20'  # Qualitative palette for categorical bars
PIE_COLORMAP = 'tab10'  # Distinct colors for categorical data
CORRELATION_HEATMAP_COLORMAP = 'coolwarm'  # Diverging colormap for correlation matrices
SEQUENTIAL_HEATMAP_COLORMAP = 'viridis'  # Perceptually uniform sequential colormap
LINE_COLORMAP = 'tab10'  # Good for distinguishing multiple lines
HISTOGRAM_COLORMAP = 'Blues'  # Sequential colormap for frequency distributions
BOX_COLORMAP = 'Set2'  # Categorical colormap for categorical comparisons
REGRESSION_COLORMAP = 'Blues'  # Sequential colormap for showing relationships
SINGLE_SERIES_COLOR = sns.color_palette("colorblind", 10)[0]

# Max characters for any textual label on charts (tick labels, legend entries, axes titles)
LABEL_MAX_CHARS = 30
ELLIPSIS = '...'


def get_independent_colormap(n_colors: int):
    """
	Return a qualitative (non-sequential) colormap suitable for many independent variables.
	The function aggregates several qualitative palettes (Tab20 family, Set*, etc.) and,
	if needed, extends with evenly spaced hues in HSV space to reach the requested size.

	Args:
		n_colors: Number of distinct colors needed.

	Returns:
		matplotlib.colors.ListedColormap instance with n_colors entries.
	"""
    import numpy as np  # Local import to avoid adding a hard dependency globally
    from matplotlib import pyplot as _plt
    from matplotlib.colors import ListedColormap, hsv_to_rgb

    palettes = [
        'tab20', 'tab20b', 'tab20c', 'tab10',
        'Set3', 'Accent', 'Dark2', 'Set2', 'Set1', 'Pastel1', 'Pastel2'
    ]

    colors_list = []
    for name in palettes:
        cmap = _plt.get_cmap(name)
        # If the cmap has a predefined .colors list (ListedColormap), use it; otherwise sample
        if hasattr(cmap, 'colors'):
            colors_list.extend(list(cmap.colors))
        else:
            # Fallback: sample 20 evenly spaced colors
            colors_list.extend([cmap(i) for i in np.linspace(0, 1, 20)])

    # Deduplicate while preserving order
    unique = []
    seen = set()
    for rgba in colors_list:
        key = tuple(round(x, 5) for x in rgba)
        if key not in seen:
            seen.add(key)
            unique.append(rgba)

    if n_colors <= len(unique):
        selected = unique[:n_colors]
    else:
        selected = list(unique)
        extra = n_colors - len(unique)
        # Generate additional distinct hues in HSV space
        for i in range(extra):
            h = (i / max(1, extra))
            rgb = hsv_to_rgb([h, 0.75, 0.9])  # vivid but readable
            selected.append((rgb[0], rgb[1], rgb[2], 1.0))

    return ListedColormap(selected, name='independent')


# Set as a function to generate distinct colors dynamically where needed
INDEPENDENT_COLORMAP = get_independent_colormap  # Non-sequential for independent variables

# Plot size defaults (width, height) in inches
PLOT_FIGSIZE_ACTIVITY_TYPES_STACKED = (30, 8)  # Adjust here to increase width of 'activity_types_stacked_by_date'

# Constants
OUTPUT_VISUALIZATIONS_DIR = os.path.join(PROJECT_ROOT, 'data_analysis')
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data_raw')


# Small utilities for filesystem setup

def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist and return the same path."""
    os.makedirs(path, exist_ok=True)
    return path


def ensure_output_dirs() -> None:
    """Ensure the base output directories exist"""
    ensure_dir(OUTPUT_VISUALIZATIONS_DIR)


def safe_filename(text: Optional[str], max_len: int = 120) -> str:
    """Return a filesystem-safe filename fragment derived from text.

    - Replaces spaces with underscores
    - Removes characters not in [A-Za-z0-9._-]
    - Truncates to max_len
    """
    if not text:
        return "unknown"
    try:
        s = str(text)
    except Exception:
        s = "unknown"
    s = s.replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9._-]", "", s)
    if not s:
        s = "unknown"
    return s[:max_len]


def compute_barh_fig_height(n_bars: int, row_height: float = 0.35, min_height: float = 4.0,
                            max_height: float = 30.0) -> float:
    """Compute a reasonable figure height for horizontal bar charts based on number of bars."""
    try:
        n = int(n_bars)
    except Exception:
        n = 10
    h = n * row_height + 2.0  # add some padding for titles/labels
    h = max(min_height, min(max_height, h))
    return h


def _truncate_text(text: Optional[str], max_chars: int = LABEL_MAX_CHARS) -> str:
    """Return text truncated to max_chars, appending ELLIPSIS if needed.

    Args:
        text: Input text (None-safe)
        max_chars: Maximum number of characters allowed (including ellipsis if applied)

    Returns:
        Truncated string (or empty string if input is falsy)
    """
    if not text:
        return ""
    try:
        s = str(text)
    except Exception:
        return ""
    if len(s) <= max_chars:
        return s
    # Ensure the ellipsis is included within the limit
    cut = max(0, max_chars - len(ELLIPSIS))
    return (s[:cut] + ELLIPSIS) if cut > 0 else ELLIPSIS


def _apply_label_truncation(fig, max_chars: int = LABEL_MAX_CHARS) -> None:
    """Truncate long texts on all axes in the figure.

    Applies to:
    - X/Y tick labels
    - Legend texts
    """
    try:
        # Ensure tick labels are realized from formatters before reading them
        try:
            fig.canvas.draw()
        except Exception:
            pass

        for ax in fig.get_axes():
            # Helper to safely truncate and set tick labels while preserving rotation/alignment
            def _truncate_axis_labels(axis: str):
                try:
                    if axis == 'x':
                        ticks = ax.get_xticks()
                        texts = ax.get_xticklabels()
                        formatter = ax.get_xaxis().get_major_formatter()
                        set_labels = ax.set_xticklabels
                    else:
                        ticks = ax.get_yticks()
                        texts = ax.get_yticklabels()
                        formatter = ax.get_yaxis().get_major_formatter()
                        set_labels = ax.set_yticklabels

                    # Preserve current rotation and alignment from existing labels if available
                    rot = texts[0].get_rotation() if texts else 0
                    ha = texts[0].get_ha() if texts else 'center'

                    # Prefer existing text; if empty (formatter-based), compute via formatter
                    label_strings = [t.get_text() for t in texts]
                    if not any(label_strings) and len(ticks) == len(texts):
                        try:
                            # Most formatters are callable
                            label_strings = [str(formatter(t)) for t in ticks]
                        except Exception:
                            label_strings = [t.get_text() for t in texts]

                    # Truncate and re-apply as fixed labels matching current ticks
                    truncated = [_truncate_text(s, max_chars) if isinstance(s, str) else '' for s in label_strings]
                    set_labels(truncated)

                    # Re-apply rotation/alignment
                    new_texts = ax.get_xticklabels() if axis == 'x' else ax.get_yticklabels()
                    for nt in new_texts:
                        nt.set_rotation(rot)
                        nt.set_ha(ha)
                except Exception:
                    pass

            # Apply to both axes
            _truncate_axis_labels('x')
            _truncate_axis_labels('y')

            # Legend entries
            leg = ax.get_legend()
            if leg is not None:
                for txt in leg.get_texts():
                    try:
                        t = txt.get_text()
                        if t:
                            txt.set_text(_truncate_text(t, max_chars))
                    except Exception:
                        pass
    except Exception:
        # Never fail plotting because of label truncation
        pass


def create_and_save_figure(
        plot_function: Callable[[], None],
        filename: str,
        figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Helper function to create, save, and close a figure.

    Args:
        plot_function: Function that creates the plot
        filename: Path to save the figure
        figsize: Size of the figure (width, height) in inches

    Returns:
        None
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=figsize)
    plot_function()
    # After the plot is created, truncate any overly long labels
    _apply_label_truncation(plt.gcf(), LABEL_MAX_CHARS)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()  # Close the figure to free memory


def generate_descriptive_stats(df, title, filename=None):
    """
    Generate descriptive statistics for a dataframe

    Args:
        df: DataFrame to analyze
        title: Title for the statistics table
        filename: Not used, kept for backward compatibility

    Returns:
        None
    """
    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # Generate statistics for numerical columns
    if len(numerical_cols) > 0:
        num_stats = df[numerical_cols].describe(percentiles=[.25, .5, .75, .9]).T
        # Add additional statistics
        num_stats['skew'] = df[numerical_cols].skew()
        num_stats['kurtosis'] = df[numerical_cols].kurtosis()
        num_stats['median'] = df[numerical_cols].median()
        num_stats['variance'] = df[numerical_cols].var()

        # Round all numerical values to 2 decimal places
        num_stats = num_stats.round(2)


def _load_excel_sheets(path: str, prefix: str = "") -> Dict[str, pd.DataFrame]:
    """
    Load all sheets from a given Excel file into a dictionary.

    Args:
        path: Full path to the Excel file.
        prefix: Optional prefix to add to sheet names (used to avoid key collisions).

    Returns:
        Dict[str, pd.DataFrame]: Dictionary of DataFrames keyed by (prefixed) sheet name.
    """
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
                    logger.warning(f"Sheet '{sheet_name}' in {path} is empty, skipping")
                    continue
                key = f"{prefix}{sheet_name}"
                sheets[key] = df
                logger.info(
                    f"Loaded sheet '{sheet_name}' from {path} with {len(df)} rows and {len(df.columns)} columns")
            except Exception as e:
                logger.error(f"Error loading sheet '{sheet_name}' from {path}: {e}")
    except Exception as e:
        logger.error(f"Error opening {path}: {e}")
    return sheets


def load_excel_files() -> Dict[str, pd.DataFrame]:
    """
    Load campaign data from campaign_data.zip (CSV files) in the config directory.
    Also loads campaign_desc.xlsx sheets (prefixed with desc_) when present for backward compatibility.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary of DataFrames keyed by logical names
    """
    import zipfile

    # Ensure config directory exists
    ensure_dir(CONFIG_DIR)

    data_dict: Dict[str, pd.DataFrame] = {}

    # 1) Load CSVs from campaign_data.zip
    zip_path = os.path.join(CONFIG_DIR, 'campaign_data.zip')
    if os.path.exists(zip_path):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Map filename (lowercased) to target keys expected by the analysis code
                name_map = {
                    '1-aggregated-data.csv': 'aggregation',
                    '2-activities.csv': 'activities',
                    '3-navigation-events.csv': 'navigation',
                    '4-notification-events.csv': 'notification_events',
                    '5-sensor-events.csv': 'sensor_events',
                }
                # Build lookup for members (handle nested paths inside the zip, case-insensitive)
                members = {os.path.basename(n).lower(): n for n in zf.namelist() if not n.endswith('/')}  # files only

                for fname, key in name_map.items():
                    lf = fname.lower()
                    if lf in members:
                        try:
                            with zf.open(members[lf]) as f:
                                df = pd.read_csv(f)
                                if df.empty:
                                    logger.warning(f"CSV '{fname}' in {zip_path} is empty, skipping")
                                    continue
                                data_dict[key] = df
                                logger.info(f"Loaded '{fname}' from {zip_path} into key '{key}' with {len(df)} rows and {len(df.columns)} columns")
                        except Exception as e:
                            logger.error(f"Error reading '{fname}' from {zip_path}: {e}")
                    else:
                        logger.warning(f"Expected file '{fname}' not found in {zip_path}")
        except zipfile.BadZipFile as e:
            logger.error(f"Invalid ZIP file {zip_path}: {e}")
        except Exception as e:
            logger.error(f"Error opening ZIP {zip_path}: {e}")
    else:
        logger.warning(f"campaign_data.zip not found in {CONFIG_DIR}")

    # 2) Optionally load description sheets from campaign_desc.xlsx (prefixed)
    try:
        data_dict.update(_load_excel_sheets(os.path.join(CONFIG_DIR, 'campaign_desc.xlsx'), prefix="desc_"))
    except Exception as e:
        logger.error(f"Error loading campaign_desc.xlsx: {e}")

    if not data_dict:
        logger.warning("No data loaded from campaign_data.zip or campaign_desc.xlsx")

    return data_dict


def load_json_files() -> Dict[str, Union[pd.DataFrame, Dict]]:
    """
    Load all JSON files from the raw directory.

    Returns:
        Dict[str, Union[pd.DataFrame, Dict]]: Dictionary of DataFrames or dictionaries, keyed by filename
    """
    # Ensure raw data directory exists
    ensure_dir(RAW_DATA_DIR)

    # Get list of JSON files
    json_files = glob.glob(f'{RAW_DATA_DIR}/*.json')
    data_dict = {}

    if not json_files:
        logger.warning(f"No JSON files found in {RAW_DATA_DIR}")
        return data_dict

    for file in json_files:
        # Extract user ID and data type from filename
        filename = os.path.basename(file)
        key = filename.split('.')[0]  # Remove .json extension

        try:
            with open(file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in {file}: {e}")
                    continue

            # Convert to DataFrame if it's a list
            if isinstance(data, list):
                if not data:  # Skip empty lists
                    logger.warning(f"Empty data list in {filename}, skipping")
                    continue

                # Optimize DataFrame construction for very large JSON lists
                if isinstance(data[0], dict):
                    CHUNK_SIZE = 5000
                    n = len(data)
                    if n > CHUNK_SIZE:
                        frames = []
                        for start in range(0, n, CHUNK_SIZE):
                            end = min(start + CHUNK_SIZE, n)
                            chunk = data[start:end]
                            frames.append(pd.DataFrame.from_records(chunk))
                        df = pd.concat(frames, ignore_index=True, sort=True)
                    else:
                        df = pd.DataFrame.from_records(data)
                else:
                    # If it's a list of scalars or non-dicts, wrap them
                    df = pd.DataFrame({'value': data})

                if df.empty:
                    logger.warning(f"Empty DataFrame created from {filename}, skipping")
                    continue

                data_dict[key] = df
            else:
                if not data:  # Skip empty dictionaries
                    logger.warning(f"Empty data dictionary in {filename}, skipping")
                    continue

                data_dict[key] = data
        except FileNotFoundError:
            logger.error(f"File not found: {file}")
        except PermissionError:
            logger.error(f"Permission denied when accessing {file}")
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")

    if not data_dict:
        logger.warning("No data loaded from JSON files")

    return data_dict


def _parse_rewards_jsonlike(rewards_str: str) -> List[Dict]:
    """
    Parse reward entries from a JSON-like string (single-quoted JSON) safely.

    Args:
        rewards_str: JSON-like string containing reward information

    Returns:
        List[Dict]: Parsed list of rewards (dictionaries) or empty list if parsing fails
    """
    if not isinstance(rewards_str, str) or not rewards_str:
        return []
    try:
        return json.loads(rewards_str.replace("'", '"'))
    except (json.JSONDecodeError, AttributeError, TypeError):
        return []


def extract_points(rewards_str: str) -> int:
    """
    Extract points from a JSON-like string in the rewardedParticipations column.

    Args:
        rewards_str: JSON-like string containing reward information

    Returns:
        int: Sum of points from all rewards, or 0 if parsing fails
    """
    rewards = _parse_rewards_jsonlike(rewards_str)
    if not rewards:
        return 0
    return sum(reward.get('points', 0) for reward in rewards)


def extract_detailed_rewards(rewards_str: str) -> List[Dict]:
    """
    Extract detailed reward information from a JSON-like string in the rewardedParticipations column.

    Args:
        rewards_str: JSON-like string containing reward information

    Returns:
        List[Dict]: List of reward dictionaries, or empty list if parsing fails
    """
    return _parse_rewards_jsonlike(rewards_str)


def analyze_activities(csv_data: Dict[str, pd.DataFrame], json_data: Dict[str, Union[pd.DataFrame, Dict]] = None) -> \
Optional[Tuple]:
    """
    Analyze activities data from the campaign data.

    This function analyzes activity data, including:
    - Activity types and frequencies
    - Points distribution
  		- User engagement patterns
    - Temporal patterns (daily, hourly)
    - User drop-out rates
    - Active vs. passive usage

    Args:
        csv_data: Dictionary of DataFrames containing campaign data
        json_data: Dictionary of DataFrames or dictionaries from JSON files

   	Returns:
		Optional[Tuple]: Tuple containing:
			- activities DataFrame
			- unique_users_count
			- dropout_metrics dictionary
			- joining_metrics dictionary
			- campaign_metrics dictionary
		Returns None if activities data is not found or invalid
    """
    # Check if activities data exists
    if 'activities' not in csv_data:
        logger.error("Activities data not found in the provided data")
        return None

    # Get activities data
    activities = csv_data['activities'].copy()  # Create a copy to avoid modifying the original

    # Check if the DataFrame is empty
    if activities.empty:
        logger.error("Activities DataFrame is empty")
        return None

    # Check for required columns
    required_columns = ['createdAt', 'pid', 'type', 'rewardedParticipations']
    missing_columns = [col for col in required_columns if col not in activities.columns]
    if missing_columns:
        logger.error(f"Activities DataFrame is missing required columns: {missing_columns}")
        return None

    # Process the data
    try:
        # Convert createdAt to datetime
        activities['createdAt'] = pd.to_datetime(activities['createdAt'])
        activities['date'] = activities['createdAt'].dt.date
        activities['hour'] = activities['createdAt'].dt.hour

        # Extract points from rewardedParticipations column
        activities['points'] = activities['rewardedParticipations'].apply(extract_points)

        # Extract detailed reward information
        activities['detailed_rewards'] = activities['rewardedParticipations'].apply(extract_detailed_rewards)

        # Create a column with the number of rewards per activity
        activities['num_rewards'] = activities['detailed_rewards'].apply(len)

        # Classify users into Active (any rewarded activity) and Passive (enrolled but never rewarded)
        try:
            # Aggregate rewards/points per user
            per_user = activities.groupby('pid').agg(
                total_rewards=('num_rewards', 'sum'),
                total_points=('points', 'sum')
            ).reset_index()

            active_user_ids = set(
                per_user[(per_user['total_rewards'] > 0) | (per_user['total_points'] > 0)]['pid'].tolist())

            # Infer enrolled users from any loaded sheets that contain a pid/playerId column; fallback to activities' pids
            enrolled_user_ids = set()
            for key, df in csv_data.items():
                if isinstance(df, pd.DataFrame):
                    try:
                        cols_map = {c.lower(): c for c in df.columns}
                        pid_col = cols_map.get('pid') or cols_map.get('playerid')
                        if pid_col is not None and pid_col in df.columns:
                            enrolled_user_ids.update(pd.Series(df[pid_col]).dropna().unique().tolist())
                    except Exception:
                        pass
            if not enrolled_user_ids:
                enrolled_user_ids = set(activities['pid'].dropna().unique().tolist())

            passive_user_ids = sorted(list(enrolled_user_ids - active_user_ids))
            active_users_count = len(active_user_ids)
            passive_users_count = len(passive_user_ids)

            # Map engagement label back to activities (by rewards definition)
            engagement_map = {pid: ('Active' if pid in active_user_ids else 'Passive') for pid in
                              activities['pid'].dropna().unique()}
            activities['engagement_by_rewards'] = activities['pid'].map(engagement_map)

            # Create visualization: Active vs Passive users (by rewards)
            try:
                total_users_considered = active_users_count + passive_users_count
                if total_users_considered > 0:
                    def plot_active_passive_pie():
                        labels = ['Active', 'Passive']
                        sizes = [active_users_count, passive_users_count]
                        # Choose colors from the configured colormap
                        try:
                            cmap = plt.get_cmap(PIE_COLORMAP)
                            colors = [cmap(0), cmap(3)]
                        except Exception:
                            colors = None
                        wedges, _texts, _autotexts = plt.pie(
                            sizes,
                            labels=[f"Active ({active_users_count})", f"Passive ({passive_users_count})"],
                            autopct='%1.1f%%' if sum(sizes) > 0 else None,
                            colors=colors,
                            startangle=90,
                            wedgeprops={'edgecolor': 'white'}
                        )
                        plt.title('Active vs Passive Players')
                        plt.axis('equal')
                        legend_labels = [
                            "Active: players who performed rewarded activities/tasks",
                            "Passive: players who were enrolled in a campaign but never got any rewards for tasks"
                        ]
                        try:
                            plt.legend(wedges, legend_labels, title='Legend', loc='upper center',
                                       bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=True)
                        except Exception:
                            plt.legend(legend_labels, title='Legend', loc='upper center', bbox_to_anchor=(0.5, -0.12),
                                       ncol=1, frameon=True)

                    create_and_save_figure(
                        plot_active_passive_pie,
                        f'{OUTPUT_VISUALIZATIONS_DIR}/player_active_vs_passive_pie.png',
                        figsize=(8, 6)
                    )
                else:
                    logger.warning('No users to plot for Active vs Passive pie chart')
            except Exception as e:
                logger.error(f"Error creating Active vs Passive users pie chart: {e}")

        except Exception as e2:
            logger.error(f"Error classifying active/passive users: {e2}")
            active_user_ids = set()
            passive_user_ids = []
            active_users_count = 0
            passive_users_count = 0
    except Exception as e:
        logger.error(f"Error processing activities data: {e}")
        return None

    # Generate descriptive statistics for activities data
    try:
        # Check if 'properties' column exists and drop it before generating statistics
        # This is to avoid generating a huge file filled with '-' characters
        activities_for_stats = activities.copy()
        if 'properties' in activities_for_stats.columns:
            activities_for_stats = activities_for_stats.drop(columns=['properties'])

        generate_descriptive_stats(activities_for_stats, "Activities Data", None)

        # Free memory
        del activities_for_stats
    except Exception as e:
        logger.error(f"Error generating descriptive statistics: {e}")
    # Continue with the analysis even if statistics generation fails

    # Create visualizations
    try:
        # 1. Activity types distribution
        # This visualization shows the frequency of different activity types
        # Helps identify which activities are most common among users
        activity_counts = activities['type'].value_counts()

        if activity_counts.empty:
            logger.warning("No activity types found for distribution visualization")
        else:
            def plot_activity_types_distribution():
                ax = activity_counts.plot(kind='bar', colormap=BAR_COLORMAP)
                plt.title('Distribution of Activity Types')
                plt.xlabel('Activity Type')
                plt.ylabel('Count')
                plt.xticks(rotation=45)

            create_and_save_figure(
                plot_activity_types_distribution,
                f'{OUTPUT_VISUALIZATIONS_DIR}/activity_types_distribution.png',
                figsize=(12, 6)
            )
            logger.info("Created activity types distribution visualization")
    except Exception as e:
        logger.error(f"Error creating activity types distribution visualization: {e}")
    # Continue with other visualizations even if this one fails

    # Wave Comparisons Barplots
    try:
        # This visualization compares metrics across different waves (time periods)
        # Helps understand how user behavior changes over time

        # Check if 'wave' column exists, if not, create it based on date ranges
        if 'wave' not in activities.columns:
            # Determine the start date and calculate waves based on weekly periods
            start_date = activities['date'].min()
            activities['wave'] = ((activities['date'] - start_date).dt.days // 7) + 1

        # Function to create wave comparison barplots
        def plot_wave_comparisons():
            # Create a figure with multiple subplots
            fig, axes = plt.subplots(2, 1, figsize=(12, 12))

            # 1. Activities per wave
            wave_activity_counts = activities.groupby('wave').size()
            wave_activity_counts.plot(kind='bar', ax=axes[0], colormap=BAR_COLORMAP)
            axes[0].set_title('Number of Activities per Wave')
            axes[0].set_xlabel('Wave')
            axes[0].set_ylabel('Number of Activities')

            # 2. Average points per wave
            wave_points = activities.groupby('wave')['points'].mean()
            wave_points.plot(kind='bar', ax=axes[1], colormap=BAR_COLORMAP)
            axes[1].set_title('Average Points per Wave')
            axes[1].set_xlabel('Wave')
            axes[1].set_ylabel('Average Points')

            plt.tight_layout()

        create_and_save_figure(
            plot_wave_comparisons,
            f'{OUTPUT_VISUALIZATIONS_DIR}/wave_comparisons.png',
            figsize=(12, 12)
        )

        # If 'pid' (player ID) column exists, create wave comparisons by player
        if 'pid' in activities.columns:
            def plot_wave_comparisons_by_player():
                # Create pivot table: waves vs players with activity counts
                # Note: Using all players may reduce readability if there are many players
                # Consider adding filtering if the visualization becomes too cluttered
                player_wave_counts = pd.crosstab(
                    activities['wave'],
                    activities['pid']
                )

                # Plot the data
                cmap = INDEPENDENT_COLORMAP(player_wave_counts.shape[1]) if callable(
                    INDEPENDENT_COLORMAP) else INDEPENDENT_COLORMAP
                ax = player_wave_counts.plot(kind='bar', figsize=(12, 6), colormap=cmap)
                plt.title('Activities per Wave by Players')
                plt.xlabel('Wave')
                plt.ylabel('Number of Activities')
                # Place legend below the plot
                handles, labels = ax.get_legend_handles_labels()
                ncols = min(6, max(1, len(labels)))
                legend = plt.legend(handles, labels, title='Player ID', loc='upper center', bbox_to_anchor=(0.5, -0.15),
                                    ncol=ncols, frameon=False)
                # Add a bit of extra bottom margin so the legend fits nicely
                plt.subplots_adjust(bottom=0.2)
                plt.tight_layout()

            create_and_save_figure(
                plot_wave_comparisons_by_player,
                f'{OUTPUT_VISUALIZATIONS_DIR}/wave_comparisons_by_player.png',
                figsize=(12, 8)
            )

        # Create wave comparisons by activity type
        def plot_wave_comparisons_by_activity_type():
            # Create pivot table: waves vs activity types with activity counts
            # Note: Using all activity types may reduce readability if there are many types
            # Consider adding filtering if the visualization becomes too cluttered
            type_wave_counts = pd.crosstab(
                activities['wave'],
                activities['type']
            )

            # Plot the data
            cmap = INDEPENDENT_COLORMAP(type_wave_counts.shape[1]) if callable(
                INDEPENDENT_COLORMAP) else INDEPENDENT_COLORMAP
            type_wave_counts.plot(kind='bar', figsize=(12, 6), colormap=cmap)
            plt.title('Activities per Wave by Activity Types')
            plt.xlabel('Wave')
            plt.ylabel('Number of Activities')
            plt.legend(title='Activity Type')
            plt.tight_layout()

        create_and_save_figure(
            plot_wave_comparisons_by_activity_type,
            f'{OUTPUT_VISUALIZATIONS_DIR}/wave_comparisons_by_activity_type.png',
            figsize=(12, 8)
        )

        # Create wave comparisons of average points by activity type
        def plot_wave_points_by_activity_type():
            # Group by wave and type, calculate mean points
            # Note: Using all activity types may reduce readability if there are many types
            # Consider adding filtering if the visualization becomes too cluttered
            wave_type_points = activities.groupby(['wave', 'type'])['points'].mean().unstack()

            # Plot the data
            cmap = INDEPENDENT_COLORMAP(wave_type_points.shape[1]) if callable(
                INDEPENDENT_COLORMAP) else INDEPENDENT_COLORMAP
            wave_type_points.plot(kind='bar', figsize=(12, 6), colormap=cmap)
            plt.title('Average Points per Wave by Activity Types')
            plt.xlabel('Wave')
            plt.ylabel('Average Points')
            plt.legend(title='Activity Type')
            plt.tight_layout()

        create_and_save_figure(
            plot_wave_points_by_activity_type,
            f'{OUTPUT_VISUALIZATIONS_DIR}/wave_points_by_activity_type.png',
            figsize=(12, 8)
        )
    except Exception as e:
        logger.error(f"Error creating wave comparisons barplots: {e}")
    # Continue with other visualizations even if this one fails

    # 2. Activities over time
    try:
        # This visualization shows the number of activities recorded each day
        # Helps identify trends, patterns, and potential periods of high/low engagement
        daily_activities = activities.groupby('date').size()

        if daily_activities.empty:
            logger.warning("No daily activities data for time series visualization")
        else:
            def plot_activities_over_time():
                # Plot by actual dates on the x-axis (each campaign day)
                series = daily_activities.sort_index()
                date_labels = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in series.index]
                x_pos = range(len(series))
                plt.bar(x_pos, series.values, width=0.6)
                plt.title('Number of Activities per Day')
                plt.xlabel('Date')
                plt.ylabel('Number of Activities')
                ax = plt.gca()
                ax.set_xticks(list(x_pos))
                ax.set_xticklabels(date_labels)
                ax.tick_params(axis='x', rotation=90)
                for tick in ax.get_xticklabels():
                    tick.set_horizontalalignment('center')
                plt.margins(x=0.01)

            create_and_save_figure(
                plot_activities_over_time,
                f'{OUTPUT_VISUALIZATIONS_DIR}/activities_over_time.png',
                figsize=(14, 7)
            )
    except Exception as e:
        logger.error(f"Error creating activities over time visualization: {e}")
    # Continue with other visualizations even if this one fails

    # 3. Points distribution by activity type
    try:
        # This visualization shows the total points earned for each activity type
        # Helps identify which activities are most rewarded and potentially most valuable
        points_by_type = activities.groupby('type')['points'].sum().sort_values(ascending=False)

        if points_by_type.empty:
            logger.warning("No points data for activity type distribution visualization")
        else:
            def plot_points_by_activity_type():
                points_by_type.plot(kind='bar', colormap=BAR_COLORMAP)
                plt.title('Total Points by Activity Type')
                plt.xlabel('Activity Type')
                plt.ylabel('Total Points')
                plt.xticks(rotation=45)

            create_and_save_figure(
                plot_points_by_activity_type,
                f'{OUTPUT_VISUALIZATIONS_DIR}/points_by_activity_type.png',
                figsize=(12, 6)
            )
    except Exception as e:
        logger.error(f"Error creating points by activity type visualization: {e}")
    # Continue with other visualizations even if this one fails

    # 3.1 Rewarded Points Analysis - Points per User
    try:
        # This visualization shows the total rewarded points earned by each user
        # Helps identify which users are earning the most points
        points_by_user = activities.groupby('pid')['points'].sum().sort_values(ascending=False)

        if points_by_user.empty:
            logger.warning("No points data for user distribution visualization")
        else:
            def plot_points_by_user():
                # Limit to top 50 users if there are too many
                plot_data = points_by_user
                if len(points_by_user) > 50:
                    plot_data = points_by_user.head(50)
                    plt.title('Total Rewarded Points by Player (Top 50 Players)')
                else:
                    plt.title('Total Rewarded Points by Player')

                plot_data.plot(kind='bar', colormap=BAR_COLORMAP)
                plt.xlabel('Player ID')
                plt.ylabel('Total Rewarded Points')
                plt.xticks(rotation=45)

            create_and_save_figure(
                plot_points_by_user,
                f'{OUTPUT_VISUALIZATIONS_DIR}/points_by_player.png',
                figsize=(12, 6)
            )
            logger.info("Created points by user visualization")
    except Exception as e:
        logger.error(f"Error creating points by user visualization: {e}")
    # Continue with other visualizations even if this one fails

    # 3.2 Rewarded Points Analysis - Average Points per Activity Type
    try:
        # This visualization shows the average rewarded points per activity type
        # Helps identify which activity types grant the most points
        rewards_by_type = activities.groupby('type')['points'].mean().sort_values(ascending=False)

        if rewards_by_type.empty:
            logger.warning("No points data for activity type distribution visualization")
        else:
            def plot_rewards_by_activity_type():
                rewards_by_type.plot(kind='bar', colormap=BAR_COLORMAP)
                plt.title('Average Rewarded Points by Activity Type')
                plt.xlabel('Activity Type')
                plt.ylabel('Average Rewarded Points')
                plt.xticks(rotation=45)

            create_and_save_figure(
                plot_rewards_by_activity_type,
                f'{OUTPUT_VISUALIZATIONS_DIR}/rewards_by_activity_type.png',
                figsize=(12, 6)
            )
    except Exception as e:
        logger.error(f"Error creating rewards by activity type visualization: {e}")
    # Continue with other visualizations even if this one fails

    # 3.3 Rewarded Points Analysis - Points Over Time
    try:
        # This visualization shows the total rewarded points earned over time
        # Helps identify trends in point earning behavior
        daily_points = activities.groupby('date')['points'].sum()

        if daily_points.empty:
            logger.warning("No daily points data for time series visualization")
        else:
            def plot_points_over_time():
                # Plot by actual dates on the x-axis (each campaign day)
                series = daily_points.sort_index()
                date_labels = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in series.index]
                x_pos = range(len(series))
                plt.bar(x_pos, series.values, width=0.6)
                plt.title('Total Rewarded Points per Day')
                plt.xlabel('Date')
                plt.ylabel('Total Rewarded Points')
                ax = plt.gca()
                ax.set_xticks(list(x_pos))
                ax.set_xticklabels(date_labels)
                ax.tick_params(axis='x', rotation=90)
                for tick in ax.get_xticklabels():
                    tick.set_horizontalalignment('center')
                plt.margins(x=0.01)

            create_and_save_figure(
                plot_points_over_time,
                f'{OUTPUT_VISUALIZATIONS_DIR}/points_over_time.png',
                figsize=(14, 7)
            )
    except Exception as e:
        logger.error(f"Error creating points over time visualization: {e}")
    # Continue with other visualizations even if this one fails

    # 3.4 Reward-level analysis by Challenge and by Rule (from rewardedParticipations)
    try:
        # Build a reward-level table by exploding detailed_rewards
        if 'detailed_rewards' not in activities.columns:
            activities['detailed_rewards'] = activities['rewardedParticipations'].apply(extract_detailed_rewards)

        rewards_df = activities[['pid', 'date', 'type', 'detailed_rewards']].copy()
        rewards_df = rewards_df.explode('detailed_rewards')
        if rewards_df['detailed_rewards'].isna().all():
            logger.warning("No detailed rewards to analyze by challenge/rule")
        else:
            rewards_df = rewards_df[rewards_df['detailed_rewards'].notna()]

            # Safe extractors
            def _extract_reward_points(r):
                try:
                    return int(r.get('points', 0)) if isinstance(r, dict) else 0
                except Exception:
                    return 0

            def _extract_challenge(r):
                """
                Returns (challenge_name, challenge_id)
                Note: some datasets may use 'xid' as list or scalar; normalize to string.
                """
                name, cid = None, None
                if isinstance(r, dict):
                    ch = r.get('challenge')
                    if isinstance(ch, dict):
                        name = ch.get('name') or ch.get('label')
                        cid = ch.get('xid') or ch.get('id')
                    # Fallbacks if challenge is not nested
                    if name is None:
                        for key in ('challengeName', 'name', 'label'):
                            if key in r and pd.notna(r.get(key)):
                                name = r.get(key)
                                break
                    if cid is None:
                        for key in ('challengeId', 'challenge_id', 'cid', 'id', 'xid'):
                            if key in r and pd.notna(r.get(key)):
                                cid = r.get(key)
                                break
                # Normalize potential list ids
                if isinstance(cid, list) and cid:
                    cid = cid[0]
                return name, cid

            def _extract_rule(r):
                if isinstance(r, dict):
                    val = r.get('rule')
                    if isinstance(val, (str, int, float)):
                        return str(val)
                return None

            rewards_df['reward_points'] = rewards_df['detailed_rewards'].apply(_extract_reward_points)
            rewards_df[['challenge_name', 'challenge_id']] = rewards_df['detailed_rewards'].apply(
                lambda r: pd.Series(_extract_challenge(r))
            )
            rewards_df['rule_name'] = rewards_df['detailed_rewards'].apply(_extract_rule)

            # Replace empties with friendly labels for grouping
            rewards_df['challenge_name'] = rewards_df['challenge_name'].fillna('Unknown')
            rewards_df['rule_name'] = rewards_df['rule_name'].fillna('Unknown')

            # Challenge: count of rewards per challenge (Top 30)
            challenge_counts = rewards_df['challenge_name'].value_counts().head(30)
            if challenge_counts.empty:
                logger.warning("No data for challenge-based counts")
            else:
                def plot_rewards_by_challenge_count():
                    ax = challenge_counts.plot(kind='bar', colormap=BAR_COLORMAP)
                    plt.title('Count of Rewarded Activities by Challenge (Top 30)')
                    plt.xlabel('Challenge Name')
                    plt.ylabel('Count of Rewards')
                    plt.xticks(rotation=45, ha='right')

                create_and_save_figure(
                    plot_rewards_by_challenge_count,
                    f'{OUTPUT_VISUALIZATIONS_DIR}/rewards_count_by_challenge.png',
                    figsize=(14, 7)
                )

            # Challenge: total points by challenge (show ALL)
            points_by_challenge = rewards_df.groupby('challenge_name')['reward_points'] \
                .sum().sort_values(ascending=False)
            if points_by_challenge.empty:
                logger.warning("No data for challenge-based points")
            else:
                def plot_points_by_challenge():
                    ax = points_by_challenge.plot(kind='bar', colormap=BAR_COLORMAP)
                    plt.title('Total Rewarded Points by Challenge')
                    plt.xlabel('Challenge Name')
                    plt.ylabel('Total Points')
                    plt.xticks(rotation=45, ha='right')

                create_and_save_figure(
                    plot_points_by_challenge,
                    f'{OUTPUT_VISUALIZATIONS_DIR}/points_by_challenge.png',
                    figsize=(14, 7)
                )

            # Rule: count of rewards per rule (Top 30)
            rule_counts = rewards_df['rule_name'].value_counts().head(30)
            if rule_counts.empty:
                logger.warning("No data for rule-based counts")
            else:
                def plot_rewards_by_rule_count():
                    ax = rule_counts.plot(kind='bar', colormap=BAR_COLORMAP)
                    plt.title('Count of Rewarded Activities by Rule (Top 30)')
                    plt.xlabel('Rule Name')
                    plt.ylabel('Count of Rewards')
                    plt.xticks(rotation=45, ha='right')

                create_and_save_figure(
                    plot_rewards_by_rule_count,
                    f'{OUTPUT_VISUALIZATIONS_DIR}/rewards_count_by_rule.png',
                    figsize=(14, 7)
                )

            # Rule: total points by rule (show ALL)
            points_by_rule = rewards_df.groupby('rule_name')['reward_points'] \
                .sum().sort_values(ascending=False)
            if points_by_rule.empty:
                logger.warning("No data for rule-based points")
            else:
                def plot_points_by_rule():
                    ax = points_by_rule.plot(kind='bar', colormap=BAR_COLORMAP)
                    plt.title('Total Rewarded Points by Rule')
                    plt.xlabel('Rule Name')
                    plt.ylabel('Total Points')
                    plt.xticks(rotation=45, ha='right')

                create_and_save_figure(
                    plot_points_by_rule,
                    f'{OUTPUT_VISUALIZATIONS_DIR}/points_by_rule.png',
                    figsize=(14, 7)
                )

            # NEW: Per-Activity-Type analysis — points by challenge (ALL challenges per type)
            try:
                types_list = [t for t in rewards_df['type'].dropna().unique().tolist()]
                if not types_list:
                    logger.warning("No activity types found for per-type challenge analysis")
                else:
                    out_dir = os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'by_type')
                    ensure_dir(out_dir)
                    for t in types_list:
                        sub = rewards_df[rewards_df['type'] == t]
                        series = sub.groupby('challenge_name')['reward_points'].sum().sort_values(ascending=False)
                        if series.empty:
                            logger.info(f"No challenge points for activity type '{t}', skipping")
                            continue

                        fname = os.path.join(out_dir, f"points_by_challenge_type_{safe_filename(t)}.png")
                        fig_h = compute_barh_fig_height(len(series))

                        def plot_points_by_challenge_for_type(series=series, t=t):
                            ax = series.plot(kind='barh', colormap=BAR_COLORMAP)
                            plt.title(f"Total Rewarded Points by Challenge — Activity Type: {t}")
                            plt.xlabel('Total Points')
                            plt.ylabel('Challenge Name')
                            try:
                                plt.gca().invert_yaxis()  # highest at top
                            except Exception:
                                pass

                        create_and_save_figure(
                            plot_points_by_challenge_for_type,
                            fname,
                            figsize=(14, fig_h)
                        )
                        logger.info(f"Saved per-type challenge points figure for type '{t}' with {len(series)} challenges to {fname}")
            except Exception as e:
                logger.error(f"Error creating per-activity-type challenge analyses: {e}")

            # NEW: Per-Challenge analysis — points by rule (ALL rules per challenge)
            try:
                challenges_list = [c for c in rewards_df['challenge_name'].dropna().unique().tolist()]
                if not challenges_list:
                    logger.warning("No challenges found for per-challenge rule analysis")
                else:
                    out_dir = os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'by_challenge')
                    ensure_dir(out_dir)
                    for ch in challenges_list:
                        sub = rewards_df[rewards_df['challenge_name'] == ch]
                        series = sub.groupby('rule_name')['reward_points'].sum().sort_values(ascending=False)
                        if series.empty:
                            logger.info(f"No rule points for challenge '{ch}', skipping")
                            continue

                        fname = os.path.join(out_dir, f"points_by_rule_challenge_{safe_filename(ch)}.png")
                        fig_h = compute_barh_fig_height(len(series))

                        def plot_points_by_rule_for_challenge(series=series, ch=ch):
                            ax = series.plot(kind='barh', colormap=BAR_COLORMAP)
                            plt.title(f"Total Rewarded Points by Rule — Challenge: {ch}")
                            plt.xlabel('Total Points')
                            plt.ylabel('Rule Name')
                            try:
                                plt.gca().invert_yaxis()
                            except Exception:
                                pass

                        create_and_save_figure(
                            plot_points_by_rule_for_challenge,
                            fname,
                            figsize=(14, fig_h)
                        )
                        logger.info(f"Saved per-challenge rule points figure for challenge '{ch}' with {len(series)} rules to {fname}")
            except Exception as e:
                logger.error(f"Error creating per-challenge rule analyses: {e}")

    except Exception as e:
        logger.error(f"Error creating challenge/rule analyses: {e}")
    # Continue regardless of errors in this block

    # 4. User activity distribution
    try:
        # This visualization shows the number of activities recorded by each user
        # Helps identify the most active users and potential engagement disparities
        user_activity = activities.groupby('pid').size().sort_values(ascending=False)

        if user_activity.empty:
            logger.warning("No user activity data for distribution visualization")
        else:
            def plot_user_activity_distribution():
                # Limit to top 50 users if there are too many
                plot_data = user_activity
                if len(user_activity) > 50:
                    plot_data = user_activity.head(50)
                    plt.title('Number of Activities by Player (Top 50 Players)')
                else:
                    plt.title('Number of Activities by Player')

                plot_data.plot(kind='bar', colormap=BAR_COLORMAP)
                plt.xlabel('Player ID')
                plt.ylabel('Number of Activities')

            create_and_save_figure(
                plot_user_activity_distribution,
                f'{OUTPUT_VISUALIZATIONS_DIR}/player_activity_distribution.png',
                figsize=(12, 6)
            )
    except Exception as e:
        logger.error(f"Error creating user activity distribution visualization: {e}")
    # Continue with other visualizations even if this one fails

    # 5. Activity Type Distribution by User (Bivariate Analysis)
    try:
        # This visualization shows how different users engage with different activity types
        # Helps identify user preferences and potential targeting opportunities
        if len(activities['pid'].unique()) < 2 or len(activities['type'].unique()) < 2:
            logger.warning(
                "Insufficient data for activity type by user visualization (need at least 2 users and 2 activity types)")
        else:
            user_type_counts = pd.crosstab(activities['pid'], activities['type'])

            if user_type_counts.empty:
                logger.warning("Empty crosstab for activity type by user visualization")
            else:
                try:
                    # Normalize the counts to get proportions
                    row_sums = user_type_counts.sum(axis=1)
                    # Check for zero sums to avoid division by zero
                    if (row_sums == 0).any():
                        # Filter out rows with zero sum
                        user_type_counts = user_type_counts[row_sums > 0]
                        row_sums = user_type_counts.sum(axis=1)

                    user_type_counts_norm = user_type_counts.div(row_sums, axis=0)

                    top_users = user_activity.index
                    logger.info(f"Using all {len(top_users)} users for activity type heatmap")

                    # Check if all top users are in the normalized dataframe
                    valid_users = [p for p in top_users if p in user_type_counts_norm.index]
                    if not valid_users:
                        logger.warning("No valid users for activity type heatmap")
                    else:
                        user_type_heatmap = user_type_counts_norm.loc[valid_users]

                        # Limit the number of activity types if there are too many
                        if user_type_heatmap.shape[1] > 15:
                            # Keep only the most common activity types
                            top_types = user_type_counts.sum().nlargest(15).index
                            user_type_heatmap = user_type_heatmap[top_types]
                            logger.info(
                                f"Limited activity type heatmap to top 15 activity types out of {user_type_counts.shape[1]}")

                        def plot_activity_type_by_user():
                            # Adjust font size for annotations based on number of users
                            # Start with font size 10 for <= 10 users, decrease as user count increases
                            annot_fontsize = max(6, 10 - (len(valid_users) - 10) / 10)

                            sns.heatmap(user_type_heatmap, cmap=SEQUENTIAL_HEATMAP_COLORMAP,
                                        annot=True, fmt='.2f', linewidths=0.5,
                                        annot_kws={'fontsize': annot_fontsize})
                            plt.title(f'Activity Type Distribution by Player (All {len(valid_users)} Players)')
                            plt.xlabel('Activity Type')
                            plt.ylabel('Player ID')

                        # Adjust figure size based on number of users
                        # Base size is 14x8, but add 0.5 height for each additional 5 users beyond 10
                        height = 8 + max(0, (len(valid_users) - 10) / 5 * 0.5)
                        # Cap the height at a reasonable maximum
                        height = min(height, 20)

                        create_and_save_figure(
                            plot_activity_type_by_user,
                            f'{OUTPUT_VISUALIZATIONS_DIR}/activity_type_by_player.png',
                            figsize=(14, height)
                        )
                except Exception as e:
                    logger.error(f"Error in normalization or heatmap creation: {e}")
    except Exception as e:
        logger.error(f"Error creating activity type by user visualization: {e}")
    # Continue with other visualizations even if this one fails

    # Calculate campaign metrics
    try:
        # Count unique users in the campaign
        unique_users_count = activities['pid'].nunique()

        # Calculate campaign length (days between first and last activity)
        campaign_start = activities['createdAt'].min()
        campaign_end = activities['createdAt'].max()
        campaign_length_days = (campaign_end - campaign_start).days

        # Extract campaign name and abbreviation from campaign_desc.xlsx if available
        campaign_name = "GameBus Campaign"  # Default name
        campaign_abbr = ""  # Default empty abbreviation
        # Look for campaign name and abbreviation in any sheet from campaign_desc.xlsx
        for key, df in csv_data.items():
            if key.startswith('desc_') and 'name' in df.columns:
                # Found a sheet with a 'name' column
                try:
                    # Try to get the first non-empty name
                    names = df['name'].dropna()
                    if not names.empty:
                        campaign_name = names.iloc[0]

                        # Try to get the abbreviation from the same row if available
                        if 'abbreviation' in df.columns:
                            # Get the abbreviation from the same row as the name
                            row_idx = names.index[0]
                            if row_idx in df.index and not pd.isna(df.loc[row_idx, 'abbreviation']):
                                campaign_abbr = df.loc[row_idx, 'abbreviation']
                        break
                except Exception as e:
                    logger.warning(f"Error extracting campaign name/abbreviation from {key}: {e}")

        # Create campaign metrics dictionary
        campaign_metrics = {
            'unique_users': unique_users_count,
            'length_days': campaign_length_days,
            'start_date': campaign_start.date(),
            'end_date': campaign_end.date(),
            'name': campaign_name,
            'abbreviation': campaign_abbr,
            'active_users_count': active_users_count if 'active_users_count' in locals() else 0,
            'passive_users_count': passive_users_count if 'passive_users_count' in locals() else 0,
            'passive_user_ids': passive_user_ids if 'passive_user_ids' in locals() else []
        }
    except Exception as e:
        logger.error(f"Error calculating campaign metrics: {e}")
        campaign_metrics = None
        unique_users_count = 0
        campaign_length_days = 0
        campaign_start = None
        campaign_end = None

    # Calculate drop-out rates
    try:
        # Group activities by user to find first and last activity overall
        user_dropout = activities.groupby('pid').agg({
            'createdAt': ['min', 'max']
        })
        user_dropout.columns = ['first_activity', 'last_activity']

        # Calculate time difference in days (as integers)
        user_dropout['dropout_days'] = (user_dropout['last_activity'] -
                                        user_dropout['first_activity']).dt.days

        # Handle negative dropout days (should not happen with proper data)
        if (user_dropout['dropout_days'] < 0).any():
            user_dropout.loc[user_dropout['dropout_days'] < 0, 'dropout_days'] = 0

        # Generate descriptive statistics for drop-out rates
        dropout_stats = user_dropout['dropout_days'].describe(percentiles=[.25, .5, .75, .9])
        dropout_stats = dropout_stats.round(2)

        # Create dropout metrics dictionary
        dropout_metrics = {
            'avg_dropout_days': dropout_stats['mean'],
            'median_dropout_days': dropout_stats['50%'],
            'min_dropout_days': dropout_stats['min'],
            'max_dropout_days': dropout_stats['max']
        }

        logger.info(f"Calculated dropout metrics: avg={dropout_metrics['avg_dropout_days']:.1f} days, " +
                    f"median={dropout_metrics['median_dropout_days']:.1f} days")
    except Exception as e:
        logger.error(f"Error calculating dropout rates: {e}")
        user_dropout = pd.DataFrame()
        dropout_stats = None
        dropout_metrics = None

    # Create visualizations for drop-out rates
    try:
        if user_dropout.empty or 'dropout_days' not in user_dropout.columns:
            logger.warning("No dropout data available for visualization")
        else:
            # 1. Histogram of drop-out rates
            def plot_dropout_rates_histogram():
                sns.histplot(user_dropout['dropout_days'], kde=True, bins=20, color='tab:blue')
                plt.title('Distribution of User Drop-out Rates')
                plt.xlabel('Time Between First and Last Activity (days)')
                plt.ylabel('Number of Users')

            create_and_save_figure(
                plot_dropout_rates_histogram,
                f'{OUTPUT_VISUALIZATIONS_DIR}/dropout_rates_distribution.png',
                figsize=(10, 6)
            )
    except Exception as e:
        logger.error(f"Error creating dropout rates histogram: {e}")
    # Continue with other visualizations even if this one fails

    # Calculate joining rates
    try:
        # Use the same user_dropout DataFrame which already has first_activity
        if user_dropout.empty or 'first_activity' not in user_dropout.columns:
            logger.warning("No joining data available for analysis")
            joining_metrics = None
        elif campaign_start is None:
            logger.warning("Campaign start date not available for joining rate analysis")
            joining_metrics = None
        else:
            # Calculate days between campaign start and first activity
            user_dropout['joining_days'] = (user_dropout['first_activity'] - campaign_start).dt.days

            # Handle negative joining days (users who joined before campaign start)
            if (user_dropout['joining_days'] < 0).any():
                logger.info("Some users joined before the official campaign start")

            # Generate descriptive statistics for joining rates
            joining_stats = user_dropout['joining_days'].describe(percentiles=[.25, .5, .75, .9])
            joining_stats = joining_stats.round(2)

            # Create joining metrics dictionary
            joining_metrics = {
                'avg_joining_days': joining_stats['mean'],
                'median_joining_days': joining_stats['50%'],
                'min_joining_days': joining_stats['min'],
                'max_joining_days': joining_stats['max']
            }

            logger.info(f"Calculated joining metrics: avg={joining_metrics['avg_joining_days']:.1f} days, " +
                        f"median={joining_metrics['median_joining_days']:.1f} days")

            # Create visualizations for joining rates
            # 1. Histogram of joining rates
            def plot_joining_rates_histogram():
                sns.histplot(user_dropout['joining_days'], kde=True, bins=20, color='tab:red')
                plt.title('Distribution of User Joining Rates')
                plt.xlabel('Days Between Campaign Start and First Activity')
                plt.ylabel('Number of Users')

            create_and_save_figure(
                plot_joining_rates_histogram,
                f'{OUTPUT_VISUALIZATIONS_DIR}/joining_rates_distribution.png',
                figsize=(10, 6)
            )

            # 3. Combined plot of dropout and joining rates (KDE)
            def plot_combined_rates():
                plt.figure(figsize=(12, 7))
                # Plot KDE for dropout rates
                sns.kdeplot(user_dropout['dropout_days'], label='Dropout Rates', color='blue', fill=True, alpha=0.3)
                # Plot KDE for joining rates
                sns.kdeplot(user_dropout['joining_days'], label='Joining Rates', color='red', fill=True, alpha=0.3)
                plt.title('Distribution of User Dropout and Joining Rates')
                plt.xlabel('Days')
                plt.ylabel('Density')
                plt.legend()
                # Add explanatory text
                plt.figtext(0.01, 0.01,
                            'Dropout Rates: Days between first and last activity\nJoining Rates: Days between campaign start and first activity',
                            fontsize=9, ha='left')

            create_and_save_figure(
                plot_combined_rates,
                f'{OUTPUT_VISUALIZATIONS_DIR}/combined_dropout_joining_rates.png',
                figsize=(12, 7)
            )

            # 4. Combined boxplot of dropout and joining rates
            def plot_combined_boxplots():
                # Prepare data for side-by-side boxplots
                import pandas as pd
                # Create a long-format DataFrame for seaborn
                dropout_df = pd.DataFrame({
                    'days': user_dropout['dropout_days'],
                    'metric': 'Dropout Rates'
                })
                joining_df = pd.DataFrame({
                    'days': user_dropout['joining_days'],
                    'metric': 'Joining Rates'
                })
                combined_df = pd.concat([dropout_df, joining_df])

                # Create the boxplot
                sns.boxplot(x='metric', y='days', data=combined_df, hue='metric', palette=['blue', 'red'], legend=False,
                            dodge=False)
                plt.title('Distribution of User Dropout and Joining Rates')
                plt.xlabel('')
                plt.ylabel('Days')
                # Add explanatory text
                plt.figtext(0.01, 0.01,
                            'Dropout Rates: Days between first and last activity\nJoining Rates: Days between campaign start and first activity',
                            fontsize=9, ha='left')

            create_and_save_figure(
                plot_combined_boxplots,
                f'{OUTPUT_VISUALIZATIONS_DIR}/combined_dropout_joining_boxplots.png',
                figsize=(12, 7)
            )
    except Exception as e:
        logger.error(f"Error calculating or visualizing joining rates: {e}")
        joining_metrics = None

    # Churn rate (no activity for 30 days) over time
    try:
        if activities.empty:
            logger.warning("No activities data available for churn rate visualization")
        else:
            # Prepare daily activity matrix by user
            daily = activities.copy()
            # Ensure 'date' is datetime (normalized to date)
            daily['date'] = pd.to_datetime(daily['date'])
            # Use a stable string representation for pid to avoid dtype mismatches during reindexing
            try:
                daily['pid_str'] = daily['pid'].astype(str)
            except Exception:
                daily['pid_str'] = daily['pid']

            # Build complete campaign date range
            series_start = daily['date'].min()
            series_end = daily['date'].max()
            date_index = pd.date_range(series_start, series_end, freq='D')

            # Build pivot table: rows = date, cols = pid (as string), value = any activity that day (bool)
            try:
                active_pivot = (
                    daily.groupby(['date', 'pid_str']).size()
                        .unstack(fill_value=0)
                        .reindex(index=date_index, fill_value=0)
                )
            except Exception as e:
                logger.error(f"Error preparing daily activity pivot for churn: {e}")
                active_pivot = pd.DataFrame(index=date_index)

            if active_pivot.shape[1] == 0:
                logger.warning("No user activity columns available for churn rate computation")
            else:
                # Boolean matrix: True if user was active on that day
                active_bool = active_pivot > 0

                # First activity date per user (joined date)
                # Compute robustly from raw daily data (by pid_str), avoiding boolean masking pitfalls
                try:
                    first_activity_series = (
                        daily.groupby('pid_str')['date'].min()
                            .reindex(active_bool.columns)
                    )
                    # Ensure proper datetime dtype
                    first_activity_series = pd.to_datetime(first_activity_series)
                except Exception as e:
                    logger.error(f"Error computing first activity dates for churn (groupby): {e}")
                    # Fallback: attempt via boolean mask stack (may be slower / brittle)
                    try:
                        stacked = active_bool[active_bool].stack().reset_index()
                        stacked.columns = ['date', 'pid_str', 'active']
                        first_activity_series = (
                            stacked.groupby('pid_str')['date'].min()
                                   .reindex(active_bool.columns)
                        )
                        first_activity_series = pd.to_datetime(first_activity_series)
                    except Exception as e2:
                        logger.error(f"Fallback method for first activity dates failed: {e2}")
                        first_activity_series = pd.Series(index=active_bool.columns, dtype='datetime64[ns]')

                # Users considered in denominator: those who have joined by given date
                try:
                    idx_vals = active_bool.index.values.astype('datetime64[ns]')
                    fa_vals = first_activity_series.values.astype('datetime64[ns]')
                    # joined_mask[date, pid] = date >= first_activity(pid)
                    joined_mask = pd.DataFrame(
                        idx_vals[:, None] >= fa_vals[None, :],
                        index=active_bool.index,
                        columns=active_bool.columns
                    )
                except Exception as e:
                    logger.error(f"Error building joined mask for churn: {e}")
                    joined_mask = pd.DataFrame(False, index=active_bool.index, columns=active_bool.columns)

                # Whether user has been active in the last 30 days (including current date)
                try:
                    active_last_30 = active_bool.rolling(window=30, min_periods=1).sum() > 0
                except Exception as e:
                    logger.error(f"Error computing 30-day rolling activity for churn: {e}")
                    active_last_30 = active_bool.copy()

                # Churned users at date: joined but no activity in last 30 days
                churned_mask = (~active_last_30) & joined_mask

                # Aggregate counts per day
                churned_count = churned_mask.sum(axis=1)
                joined_count = joined_mask.sum(axis=1)
                # Avoid division by zero
                with pd.option_context('mode.use_inf_as_na', True):
                    denom = joined_count.replace(0, float('nan')).astype(float)
                    churn_rate = (churned_count.astype(float) / denom).fillna(0.0)

                churn_df = pd.DataFrame({
                    'date': churn_rate.index,
                    'churned_count': churned_count.values,
                    'joined_count': joined_count.values,
                    'churn_rate': churn_rate.values
                }).set_index('date')

                # Sanity check: if we have activity but joined_count stays zero, log a warning for diagnostics
                try:
                    if active_bool.values.sum() > 0 and churn_df['joined_count'].max() == 0:
                        logger.warning(
                            "Joined count is zero for all days despite existing activities. "
                            "This typically indicates pid alignment issues. Using pid as string in this block."
                        )
                except Exception:
                    pass

                # Plot churn rate over time
                def plot_churn_rate_over_time():
                    plt.plot(churn_df.index, churn_df['churn_rate'] * 100.0, color='tab:purple', linewidth=2)
                    plt.title('Churn Rate Over Time (No activity in last 30 days)')
                    plt.xlabel('Date')
                    plt.ylabel('Churn Rate (%)')
                    # Add a subtle grid for readability
                    plt.grid(alpha=0.3, linestyle='--', linewidth=0.5)

                create_and_save_figure(
                    plot_churn_rate_over_time,
                    f'{OUTPUT_VISUALIZATIONS_DIR}/churn_rate_over_time.png',
                    figsize=(12, 6)
                )

                # Optional: plot churned and joined counts for context when data is large enough
                try:
                    if len(churn_df) <= 180:  # avoid overly dense bar chart for very long campaigns
                        def plot_churn_counts():
                            plt.plot(churn_df.index, churn_df['joined_count'], label='Joined (cumulative)', color='tab:blue')
                            plt.plot(churn_df.index, churn_df['churned_count'], label='Churned', color='tab:orange')
                            plt.title('Joined vs Churned Users Over Time (30-day inactivity)')
                            plt.xlabel('Date')
                            plt.ylabel('Number of Users')
                            plt.legend()
                            plt.grid(alpha=0.3, linestyle='--', linewidth=0.5)

                        create_and_save_figure(
                            plot_churn_counts,
                            f'{OUTPUT_VISUALIZATIONS_DIR}/churn_counts_over_time.png',
                            figsize=(12, 6)
                        )
                except Exception as e:
                    logger.warning(f"Unable to create churn counts plot: {e}")
    except Exception as e:
        logger.error(f"Error creating churn rate visualization: {e}")
    # Continue with other visualizations even if this one fails

    # 2. Bar chart of usage by day of week (using only date information)
    try:
        if activities.empty:
            logger.warning("No activities data available for day of week visualization")
        else:
            # Create a new implementation that uses only date information
            try:
                # Make a copy to avoid modifying the original
                plot_data = activities.copy()

                # Convert date to day of week if not already done
                if 'day_of_week' not in plot_data.columns:
                    plot_data['day_of_week'] = pd.to_datetime(plot_data['date']).dt.day_name()

                # Order days of week correctly
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

                # Count activities by day of week
                activities_by_day = plot_data.groupby('day_of_week').size()

                # Check if we have data for all days
                missing_days = [day for day in day_order if day not in activities_by_day.index]
                if missing_days:
                    logger.info(f"Missing data for days: {missing_days}")

                # Reindex to ensure correct order
                activities_by_day = activities_by_day.reindex(day_order)

                if activities_by_day.isnull().values.any():
                    logger.warning("Some days have no data, filling with zeros")
                    activities_by_day = activities_by_day.fillna(0)

                if activities_by_day.empty or activities_by_day.sum() <= 0:
                    logger.warning("No activities data by day of week for visualization")

                else:
                    def plot_usage_by_day_of_week():
                        # bar positions
                        x = range(len(day_order))
                        # ensure consistent order and lengths
                        counts = [int(activities_by_day.get(day, 0)) for day in day_order]
                        plt.bar(x, counts, color=SINGLE_SERIES_COLOR)
                        plt.title('Usage by Day of Week')
                        plt.xlabel('Day of Week')
                        plt.ylabel('Number of Activities')
                        plt.xticks(x, day_order, rotation=45, ha='right')
                        # small bottom margin for labels
                        plt.margins(x=0.01)

                    create_and_save_figure(
                        plot_usage_by_day_of_week,
                        f'{OUTPUT_VISUALIZATIONS_DIR}/usage_by_day_of_week.png',
                        figsize=(10, 6)
                    )

            except Exception as e:
                logger.error(f"Error processing data for day of week visualization: {e}")
    except Exception as e:
        logger.error(f"Error creating usage by day of week visualization: {e}")
    # Continue with other visualizations even if this one fails

    # 4. Activity heatmap by day of week and hour of day
    try:
        if activities.empty:
            logger.warning("No activities data available for heatmap visualization")
        else:
            try:
                # This visualization shows when users are most active during the week
                # Helps identify peak activity times and patterns

                # Make a copy to avoid modifying the original
                heatmap_data = activities.copy()

                # Convert date to day of week if not already done
                if 'day_of_week' not in heatmap_data.columns:
                    heatmap_data['day_of_week'] = pd.to_datetime(heatmap_data['date']).dt.day_name()

                # Define day order
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

                # Create a pivot table of activity counts by day of week and hour
                try:
                    activity_heatmap_data = pd.crosstab(
                        index=heatmap_data['day_of_week'],
                        columns=heatmap_data['hour']
                    ).fillna(0)

                    # Check if we have data for all days
                    missing_days = [day for day in day_order if day not in activity_heatmap_data.index]
                    if missing_days:
                        logger.info(f"Missing days in heatmap: {missing_days}")

                    # Reindex to ensure days are in correct order
                    activity_heatmap_data = activity_heatmap_data.reindex(day_order)

                    # Fill NaN values with 0
                    activity_heatmap_data = activity_heatmap_data.fillna(0)

                    # Ensure all 24 hours are present and in order
                    activity_heatmap_data = activity_heatmap_data.reindex(columns=range(24), fill_value=0)

                    if activity_heatmap_data.empty or activity_heatmap_data.sum().sum() == 0:
                        logger.warning("Empty heatmap data, skipping visualization")
                    else:
                        def plot_activity_heatmap_by_time():
                            # Use a fixed format string for the heatmap
                            # We'll use '.0f' for all values since most will be integers
                            sns.heatmap(activity_heatmap_data, cmap=SEQUENTIAL_HEATMAP_COLORMAP, annot=True,
                                        fmt='.0f', linewidths=0.5)
                            plt.title('Activity Heatmap by Day of Week and Hour of Day')
                            plt.xlabel('Hour of Day')
                            plt.ylabel('Day of Week')

                        create_and_save_figure(
                            plot_activity_heatmap_by_time,
                            f'{OUTPUT_VISUALIZATIONS_DIR}/activity_heatmap_by_time.png',
                            figsize=(15, 8)
                        )
                except Exception as e:
                    logger.error(f"Error creating crosstab for heatmap: {e}")
            except Exception as e:
                logger.error(f"Error processing data for heatmap visualization: {e}")
    except Exception as e:
        logger.error(f"Error creating activity heatmap by time: {e}")
    # Continue with other visualizations even if this one fails

    # 5. Stacked bar chart of activity types over time
    try:
        if activities.empty:
            logger.warning("No activities data available for stacked bar chart visualization")
        else:
            try:
                # This visualization shows how different types of activities contribute to overall engagement
                # Helps identify trends in activity type preferences over time

                # Group activities by date and type, and count occurrences
                activity_types_by_date = pd.crosstab(
                    index=activities['date'],
                    columns=activities['type']
                )

                if activity_types_by_date.empty:
                    logger.warning("Empty crosstab for activity types by date visualization")
                else:
                    # Get the top 5 most common activity types for readability
                    type_counts = activities['type'].value_counts()

                    if type_counts.empty:
                        logger.warning("No activity types found for stacked bar chart")
                    else:
                        # Determine how many activity types to show
                        if len(type_counts) > 5:
                            top_count = 5
                            logger.info(
                                f"Limiting stacked bar chart to top {top_count} activity types out of {len(type_counts)}")
                        else:
                            top_count = len(type_counts)
                            logger.info(f"Using all {top_count} activity types for stacked bar chart")

                        top_activity_types = type_counts.nlargest(top_count).index.tolist()

                        # Check if any of the top types are not in the crosstab
                        missing_types = [t for t in top_activity_types if t not in activity_types_by_date.columns]
                        if missing_types:
                            logger.warning(f"Some top activity types are missing from the crosstab: {missing_types}")
                            # Remove missing types from the list
                            top_activity_types = [t for t in top_activity_types if t not in missing_types]

                        if not top_activity_types:
                            logger.warning("No valid activity types for stacked bar chart")
                        else:
                            # Filter for only the top activity types
                            activity_types_by_date_filtered = activity_types_by_date[top_activity_types]

                            # Display per campaign day (starting from 1) instead of aggregating by week
                            # Ensure every calendar day is present (including zero-activity days)
                            # 1) Convert index to datetime and sort
                            try:
                                activity_types_by_date_filtered.index = pd.to_datetime(
                                    activity_types_by_date_filtered.index, errors='coerce')
                                activity_types_by_date_filtered = activity_types_by_date_filtered.sort_index()
                            except Exception:
                                pass

                            # 2) Reindex to full daily date range and fill missing with zeros
                            try:
                                if len(activity_types_by_date_filtered.index) > 0:
                                    full_range = pd.date_range(start=activity_types_by_date_filtered.index.min(),
                                                               end=activity_types_by_date_filtered.index.max(),
                                                               freq='D')
                                    activity_types_by_date_filtered = activity_types_by_date_filtered.reindex(
                                        full_range).fillna(0)
                            except Exception:
                                pass

                            # Keep actual dates on the x-axis (no conversion to campaign day numbers)
                            # Prepare labels as date strings for clarity
                            try:
                                date_labels = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in
                                               activity_types_by_date_filtered.index]
                            except Exception:
                                date_labels = [str(d) for d in activity_types_by_date_filtered.index]

                            # Set labels and title for dates
                            title = 'Activity Types Distribution by Date'
                            xlabel = 'Date'

                            # Plot stacked bar chart
                            plt.figure(figsize=PLOT_FIGSIZE_ACTIVITY_TYPES_STACKED)
                            # Determine independent colormap (function or instance)
                            cmap = INDEPENDENT_COLORMAP(activity_types_by_date_filtered.shape[1]) if callable(
                                INDEPENDENT_COLORMAP) else INDEPENDENT_COLORMAP
                            activity_types_by_date_filtered.plot(kind='bar', stacked=True, colormap=cmap, width=1,
                                                                 ax=plt.gca())
                            plt.title(title)
                            plt.xlabel(xlabel)
                            plt.ylabel('Number of Activities')
                            plt.legend(title='Activity Type')
                            # Show every date label (including zero-activity days) and rotate 180 degrees
                            td = len(activity_types_by_date_filtered)
                            if td > 0:
                                ax = plt.gca()
                                ax.set_xticks(range(td))
                                ax.set_xticklabels(date_labels)
                                ax.tick_params(axis='x', rotation=90)
                                for tick in ax.get_xticklabels():
                                    tick.set_horizontalalignment('center')
                            # Truncate long labels on axes/legend before saving
                            _apply_label_truncation(plt.gcf(), LABEL_MAX_CHARS)
                            plt.tight_layout()
                            plt.savefig(f"{OUTPUT_VISUALIZATIONS_DIR}/activity_types_stacked_by_date.png",
                                        bbox_inches='tight')
                            plt.close()
            except Exception as e:
                logger.error(f"Error processing data for stacked bar chart: {e}")
    except Exception as e:
        logger.error(f"Error creating activity types stacked by date visualization: {e}")
    # Continue with other visualizations even if this one fails

    # 6. User engagement heatmap
    try:
        if activities.empty:
            logger.warning("No activities data available for user engagement heatmap")
        else:
            try:
                # This visualization shows user activity patterns over time
                # Helps identify which users are most active and when

                # Create a pivot table of activity counts by user and date
                user_activity_heatmap = pd.crosstab(
                    index=activities['pid'],
                    columns=activities['date']
                ).fillna(0)

                if user_activity_heatmap.empty:
                    logger.warning("Empty crosstab for user engagement heatmap")
                else:
                    # Determine how many users to show
                    if user_activity is None or user_activity.empty:
                        logger.warning("No user activity data for user engagement heatmap")
                    else:
                        top_users = user_activity.index
                        logger.info(f"Using all {len(top_users)} users for engagement heatmap")

                        # Check if all top users are in the heatmap
                        missing_users = [u for u in top_users if u not in user_activity_heatmap.index]
                        if missing_users:
                            logger.warning(f"Some top users are missing from the heatmap: {missing_users}")
                            # Remove missing users from the list
                            top_users = [u for u in top_users if u not in missing_users]

                        if len(top_users) == 0:
                            logger.warning("No valid users for engagement heatmap")
                        else:
                            # Filter to only include top users
                            user_activity_heatmap_filtered = user_activity_heatmap.loc[top_users]

                            # Always keep daily granularity for the user engagement heatmap (no weekly aggregation)
                            # Ensure columns are datetime for consistent formatting
                            if not isinstance(user_activity_heatmap_filtered.columns, pd.DatetimeIndex):
                                try:
                                    user_activity_heatmap_filtered.columns = pd.to_datetime(
                                        user_activity_heatmap_filtered.columns)
                                except Exception:
                                    pass

                            # Format column labels as 'YYYY-MM-DD' for readability
                            try:
                                user_activity_heatmap_filtered.columns = [pd.to_datetime(d).strftime('%Y-%m-%d') for d
                                                                          in user_activity_heatmap_filtered.columns]
                            except Exception:
                                user_activity_heatmap_filtered.columns = [str(d) for d in
                                                                          user_activity_heatmap_filtered.columns]

                            title = f'Player Engagement Heatmap by Day (All {len(top_users)} Players)'
                            xlabel = 'Date'

                            if user_activity_heatmap_filtered.empty or user_activity_heatmap_filtered.values.sum() == 0:
                                logger.warning("Empty filtered heatmap data, skipping visualization")
                            else:
                                def plot_user_engagement_heatmap():
                                    # Adjust font size for annotations based on number of users and dates
                                    # Start with font size 10 for small matrices, decrease as size increases
                                    num_users = user_activity_heatmap_filtered.shape[0]
                                    num_dates = user_activity_heatmap_filtered.shape[1]
                                    annot_fontsize = max(6, 10 - (num_users * num_dates) / 200)

                                    # Use integer format for activity counts
                                    sns.heatmap(user_activity_heatmap_filtered, cmap=SEQUENTIAL_HEATMAP_COLORMAP,
                                                annot=True, fmt='.0f', linewidths=0.5,
                                                annot_kws={'fontsize': annot_fontsize})
                                    plt.title(title)
                                    plt.xlabel(xlabel)
                                    plt.ylabel('Player ID')
                                    # Add explanation of how engagement is calculated
                                    plt.figtext(0.5, 0.01,
                                                "Note: Engagement is calculated as the total number of activities per player. " +
                                                "Colors represent activity count.",
                                                ha='center', fontsize=9, wrap=True)

                                create_and_save_figure(
                                    plot_user_engagement_heatmap,
                                    f'{OUTPUT_VISUALIZATIONS_DIR}/player_engagement_heatmap.png',
                                    figsize=(16, 10)
                                )
            except Exception as e:
                logger.error(f"Error processing data for user engagement heatmap: {e}")
    except Exception as e:
        logger.error(f"Error creating user engagement heatmap: {e}")
    # Continue with other visualizations even if this one fails

    # Prepare return values
    logger.info("Preparing return values from analyze_activities")

    # Ensure campaign_metrics is properly defined if it wasn't created earlier
    if 'campaign_metrics' not in locals() or campaign_metrics is None:
        logger.warning("Campaign metrics not calculated, creating empty dictionary")
        campaign_metrics = {
            'unique_users': unique_users_count if 'unique_users_count' in locals() else 0,
            'length_days': campaign_length_days if 'campaign_length_days' in locals() else 0,
            'start_date': campaign_start.date() if 'campaign_start' in locals() and campaign_start is not None else None,
            'end_date': campaign_end.date() if 'campaign_end' in locals() and campaign_end is not None else None,
            'name': "GameBus Campaign",  # Default campaign name
            'abbreviation': "",  # Default empty abbreviation
            'active_users_count': active_users_count if 'active_users_count' in locals() else 0,
            'passive_users_count': passive_users_count if 'passive_users_count' in locals() else 0,
            'passive_user_ids': passive_user_ids if 'passive_user_ids' in locals() else []        }

    try:
        # Ensure statistics directory exists
        ensure_dir(OUTPUT_VISUALIZATIONS_DIR)

    except Exception as e:
        logger.error(f"Error creating statistics directory or saving files: {e}")



    # Return the results
    try:
        return activities, unique_users_count, dropout_metrics, joining_metrics, campaign_metrics
    except Exception as e:
        logger.error(f"Error returning analysis results: {e}")
        # Return default values if there's an error
        return None, 0, None, None, None


def classify_user_engagement(df):
    """
    Classifies users into active and passive engagement categories based on their activities.

    Args:
        df (DataFrame): DataFrame containing user activity data

    Returns:
        DataFrame: Original DataFrame with additional 'engagement_type' column
    """
    if 'user_id' not in df.columns:
        return df

    # Group by user to get their total activities
    user_summary = df.groupby('user_id').agg({
        'DURATION': 'sum',
        'STEPS': 'sum',
        'KCALORIES': 'sum'
    }).reset_index()

    # Classify users based on their activity levels
    def determine_engagement(row):
        if row['DURATION'] > 0 or row['STEPS'] > 0 or row['KCALORIES'] > 0:
            return 'Active'
        return 'Passive'

    user_summary['engagement_type'] = user_summary.apply(determine_engagement, axis=1)

    # Merge the engagement type back to the original DataFrame
    df = df.merge(
        user_summary[['user_id', 'engagement_type']],
        on='user_id',
        how='left'
    )

    return df


def analyze_engagement_patterns(df):
    """
    Analyzes engagement patterns by comparing active and passive users.

    Args:
        df (DataFrame): DataFrame containing user activity data with engagement_type
    """
    try:
        if 'engagement_type' not in df.columns:
            logger.warning("Engagement type not found in data. Run classify_user_engagement first.")
            return

        # Check if we have any data to analyze
        if df.empty:
            logger.warning("No data to analyze in analyze_engagement_patterns")
            return

        # Log the columns in the dataframe for debugging
        logger.info(f"Columns in dataframe: {df.columns.tolist()}")

        # Check if we have user_id column
        if 'user_id' not in df.columns:
            logger.warning("user_id column not found in data for engagement analysis")
            return

        # Calculate engagement statistics
        engagement_stats = df.groupby('engagement_type')['user_id'].nunique()

        # Check if we have any engagement types
        if engagement_stats.empty:
            logger.warning("No engagement types found in data")
            return

        # Ensure output directory exists
        ensure_dir(OUTPUT_VISUALIZATIONS_DIR)

        # Engagement distribution visualization
        try:
            def plot_engagement_distribution():
                plt.figure(figsize=(8, 6))
                engagement_stats.plot(kind='pie', autopct='%1.1f%%', colormap=PIE_COLORMAP)
                plt.title('Distribution of User Engagement Types')
                plt.ylabel('')

            create_and_save_figure(
                plot_engagement_distribution,
                os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'user_engagement_distribution.png'),
                figsize=(8, 6)
            )
        except Exception as e:
            logger.error(f"Error creating engagement distribution visualization: {e}")

        # Activity metrics by engagement type
        metrics = ['STEPS', 'KCALORIES', 'DURATION']
        for metric in metrics:
            try:
                # Check if the metric column exists
                if metric not in df.columns:
                    logger.warning(f"Metric {metric} not found in data")
                    continue

                # Check if the metric has any non-zero values
                if df[metric].sum() == 0:
                    logger.warning(f"Metric {metric} has all zero values, skipping visualization")
                    continue

                def plot_metric_by_engagement():
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(data=df, x='engagement_type', y=metric, hue='engagement_type', palette=BOX_COLORMAP,
                                legend=False, dodge=False)
                    plt.title(f'{metric.capitalize()} Distribution by Engagement Type')
                    plt.xlabel('Engagement Type')
                    plt.ylabel(metric.capitalize())

                create_and_save_figure(
                    plot_metric_by_engagement,
                    os.path.join(OUTPUT_VISUALIZATIONS_DIR, f'{metric}_by_engagement.png'),
                    figsize=(10, 6)
                )
            except Exception as e:
                logger.error(f"Error creating {metric} by engagement visualization: {e}")

        # Statistical comparison
        for metric in metrics:
            try:
                # Check if the metric column exists
                if metric not in df.columns:
                    continue

                active_data = df[df['engagement_type'] == 'Active'][metric]
                passive_data = df[df['engagement_type'] == 'Passive'][metric]

                if len(active_data) > 0 and len(passive_data) > 0:
                    # Check if we have any non-zero values
                    if active_data.sum() == 0 and passive_data.sum() == 0:
                        logger.warning(
                            f"Both active and passive users have all zero values for {metric}, skipping statistical comparison")
                        continue

                    stat, pval = stats.mannwhitneyu(
                        active_data.dropna(),
                        passive_data.dropna(),
                        alternative='two-sided'
                    )

            except Exception as e:
                logger.error(f"Error performing statistical comparison for {metric}: {e}")
    except Exception as e:
        logger.error(f"Error in analyze_engagement_patterns: {e}")
        import traceback
        logger.error(f"Error traceback: {traceback.format_exc()}")


def analyze_visualizations_challenges_tasks(csv_data: Dict[str, pd.DataFrame]) -> None:
    """Analyze visualizations, challenges, and tasks data from campaign_desc.xlsx and campaign_data.xlsx

    This function analyzes the hierarchical structure of:
    - Visualizations
    - Challenges (assigned to visualizations)
    - Tasks (assigned to challenges)

    It generates visualizations and statistics about completion rates, points, etc.
    
    It also ranks completed tasks by:
    1. Activity type (e.g., walking, fruit intake)
    2. Data source (self-report, Garmin, Nutrida)
    3. Rewarded points

    Args:
        csv_data: Dictionary of DataFrames containing campaign data
    """

    # Check if required data exists
    required_sheets = ['desc_visualizations', 'desc_challenges', 'desc_tasks', 'activities']
    missing_sheets = [sheet for sheet in required_sheets if sheet not in csv_data]
    if missing_sheets:
        logger.error(f"Missing required data sheets: {missing_sheets}")
        return

    # Get the data
    visualizations_df = csv_data['desc_visualizations'].copy()
    challenges_df = csv_data['desc_challenges'].copy()
    # Tasks are taken from the campaign description file (config/campaign_desc.xlsx),
    # specifically the sheet named "tasks". When loaded by load_excel_files, sheets from
    # campaign_desc.xlsx are stored in csv_data with a desc_ prefix; thus the tasks sheet
    # is available as csv_data['desc_tasks'].
    tasks_df = csv_data['desc_tasks'].copy()
    # Activities used in timing/frequency plots (including activity_type_by_day) are taken
    # from the campaign data file (config/campaign_data.xlsx), sheet named "activities".
    # load_excel_files loads that sheet directly into csv_data under the key 'activities'.
    activities_df = csv_data['activities'].copy()

    # Create visualizations

    # 4. Task completion analysis (if activity data contains task completion information)
    try:
        if 'activities' in csv_data and 'type' in activities_df.columns:
            # Try to match activities with tasks

            # Get unique activity types
            activity_types = activities_df['type'].unique()

            # Count activities by type
            activity_counts = activities_df['type'].value_counts()

            def plot_activity_completion():
                plt.figure(figsize=(12, 8))

                # Create bar chart
                plt.bar(range(len(activity_counts)), activity_counts.values, color=SINGLE_SERIES_COLOR)

                # Add labels
                plt.xlabel('Activity Type')
                plt.ylabel('Count')
                plt.title('Activity Completion by Type')
                plt.xticks(range(len(activity_counts)), activity_counts.index, rotation=45, ha='right')
                plt.tight_layout()

            # Save the activity completion plot
            create_and_save_figure(
                plot_activity_completion,
                os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'activity_completion.png'),
                figsize=(14, 8)
            )

            # 5. Frequency, timing, and type of tasks completed analysis
            # Ensure activities_df has necessary columns for analysis
            if 'createdAt' not in activities_df.columns:
                logger.warning("No timestamp data available for frequency and timing analysis")
            else:
                # Convert createdAt to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(activities_df['createdAt']):
                    activities_df['createdAt'] = pd.to_datetime(activities_df['createdAt'])

                # Extract date and time components
                activities_df['date'] = activities_df['createdAt'].dt.date
                activities_df['hour'] = activities_df['createdAt'].dt.hour
                activities_df['day_of_week'] = activities_df['createdAt'].dt.day_name()

                # Define day order for consistent display
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

                # Calculate activity counts by date, hour, and day
                activities_by_date = activities_df.groupby('date').size()
                activities_by_hour = activities_df.groupby('hour').size()
                activities_by_day = activities_df.groupby('day_of_week').size()

                # Reindex days of week to ensure consistent order
                activities_by_day = activities_by_day.reindex(day_order)

                # Completion Rate Analysis by Activity Type/Task
                # For each activity type, analyze completion patterns

                # Create a heatmap of activity types by hour
                # Ensure all 24 hours (0-23) are represented, even if some have zero counts
                activity_hour_counts = pd.crosstab(activities_df['hour'], activities_df['type']).reindex(range(24),
                                                                                                         fill_value=0)

                def plot_activity_type_by_hour():
                    plt.figure(figsize=(14, 8))
                    sns.heatmap(
                        activity_hour_counts,
                        cmap=SEQUENTIAL_HEATMAP_COLORMAP,
                        annot=True,
                        fmt='.0f',
                        linewidths=.5,
                        annot_kws={'fontsize': 8}
                    )
                    plt.title('Activity Types by Hour of Day')
                    plt.xlabel('Activity Type')
                    plt.ylabel('Hour of Day')
                    plt.tight_layout()

                create_and_save_figure(
                    plot_activity_type_by_hour,
                    f'{OUTPUT_VISUALIZATIONS_DIR}/activity_type_by_hour.png',
                    figsize=(14, 8)
                )

                # Create a heatmap of activity types by day of week
                activity_day_counts = pd.crosstab(activities_df['day_of_week'], activities_df['type'])
                activity_day_counts = activity_day_counts.reindex(day_order).fillna(0)

                def plot_activity_type_by_day():
                    plt.figure(figsize=(14, 8))
                    sns.heatmap(
                        activity_day_counts,
                        cmap=SEQUENTIAL_HEATMAP_COLORMAP,
                        annot=True,
                        fmt='.0f',
                        linewidths=.5,
                        annot_kws={'fontsize': 8}
                    )
                    plt.title('Activity Types by Day of Week')
                    plt.xlabel('Activity Type')
                    plt.ylabel('Day of Week')
                    plt.tight_layout()

                create_and_save_figure(
                    plot_activity_type_by_day,
                    f'{OUTPUT_VISUALIZATIONS_DIR}/activity_type_by_day.png',
                    figsize=(14, 8)
                )

                # Count unique active users per day of the campaign
                active_users_per_day = activities_df.groupby('date')['pid'].nunique()

                def plot_active_users_per_day():
                    plt.figure(figsize=(12, 6))
                    # Plot by actual dates on the x-axis
                    series = active_users_per_day.sort_index()
                    date_labels = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in series.index]
                    x_pos = range(len(series))
                    plt.bar(x_pos, series.values, color=SINGLE_SERIES_COLOR, width=0.6)
                    plt.title('Number of Active Players per Day')
                    plt.xlabel('Date')
                    plt.ylabel('Number of Active Players')
                    ax = plt.gca()
                    ax.set_xticks(list(x_pos))
                    ax.set_xticklabels(date_labels)
                    ax.tick_params(axis='x', rotation=90)
                    for tick in ax.get_xticklabels():
                        tick.set_horizontalalignment('center')
                    plt.margins(x=0.01)
                    plt.tight_layout(pad=2.0)

                create_and_save_figure(
                    plot_active_users_per_day,
                    f'{OUTPUT_VISUALIZATIONS_DIR}/active_players_per_day.png',
                    figsize=(12, 6)
                )

        else:
            logger.warning("No activity data available for completion analysis")
    except Exception as e:
        logger.error(f"Error creating activity completion analysis: {e}")
        import traceback
        logger.error(f"Error traceback: {traceback.format_exc()}")

    # New plots: completed tasks analysis (users per task and completions per day)
    try:
        # Ensure we have necessary columns in activities_df
        required_cols = ['pid']
        missing = [c for c in required_cols if c not in activities_df.columns]
        if missing:
            logger.warning(f"Cannot compute completed tasks plots; missing columns: {missing}")
        else:
            # Use createdAt or date to determine completion date
            if 'createdAt' in activities_df.columns and not pd.api.types.is_datetime64_any_dtype(
                    activities_df['createdAt']):
                with pd.option_context('mode.chained_assignment', None):
                    activities_df['createdAt'] = pd.to_datetime(activities_df['createdAt'], errors='coerce')
            # Ensure date column exists for grouping
            if 'date' not in activities_df.columns:
                if 'createdAt' in activities_df.columns:
                    activities_df['date'] = activities_df['createdAt'].dt.date
                else:
                    logger.warning(
                        "Activities lack both 'date' and 'createdAt'; cannot compute per-day task completions")
            # Extract detailed rewards to find challenge IDs
            if 'detailed_rewards' not in activities_df.columns:
                if 'rewardedParticipations' in activities_df.columns:
                    activities_df['detailed_rewards'] = activities_df['rewardedParticipations'].apply(
                        lambda x: extract_detailed_rewards(x) if isinstance(x, str) else []
                    )
                else:
                    activities_df['detailed_rewards'] = [[] for _ in range(len(activities_df))]

            # Build completion records for both challenges and tasks (rules)
            challenge_completion_records = []
            task_completion_records = []
            for _, row in activities_df.iterrows():
                reward_list = row.get('detailed_rewards', []) if isinstance(row.get('detailed_rewards', []),
                                                                            list) else []
                if not reward_list:
                    continue
                for reward in reward_list:
                    if not isinstance(reward, dict):
                        continue
                    # Extract challenge info (may be nested under 'challenge')
                    challenge_id = None
                    challenge_name = None
                    if isinstance(reward.get('challenge'), dict):
                        ch = reward.get('challenge')
                        challenge_id = ch.get('xid') or ch.get('id')
                        challenge_name = ch.get('name') or ch.get('label')
                    else:
                        # Try common keys for challenge identifiers directly on reward
                        for key in ['challengeId', 'challenge_id', 'cid', 'id']:
                            if key in reward and pd.notna(reward[key]):
                                challenge_id = reward[key]
                                break
                        for key in ['challengeName', 'name', 'label']:
                            if key in reward and pd.notna(reward[key]):
                                challenge_name = reward[key]
                                break
                    # Extract task info: in campaign_data rewardedParticipations, 'rule' is the task
                    task_name = reward.get('rule')
                    # Append challenge completion
                    challenge_completion_records.append({
                        'pid': row['pid'],
                        'challenge_id': challenge_id,
                        'challenge_name': challenge_name,
                        'date': row['date'] if 'date' in row and pd.notna(row['date']) else (
                            row['createdAt'].date() if 'createdAt' in row and pd.notna(row['createdAt']) else None)
                    })
                    # Append task completion (use name as key; optionally map to desc_tasks later)
                    if pd.notna(task_name) and str(task_name).strip() != '':
                        task_completion_records.append({
                            'pid': row['pid'],
                            'task_name': str(task_name).strip(),
                            'activity_type': row.get('type'),
                            'date': row['date'] if 'date' in row and pd.notna(row['date']) else (
                                row['createdAt'].date() if 'createdAt' in row and pd.notna(row['createdAt']) else None)
                        })

            # DataFrames for completions
            challenge_completions_df = pd.DataFrame(challenge_completion_records)
            task_completions_df = pd.DataFrame(task_completion_records)

            # Optional: map tasks to desc_tasks if possible to standardize naming/ids
            if not task_completions_df.empty and 'desc_tasks' in csv_data:
                try:
                    # Attempt to find a column in desc_tasks that matches the task 'rule' text
                    possible_name_cols = [c for c in csv_data['desc_tasks'].columns if
                                          c.lower() in ['rule', 'task', 'name', 'label']]
                    if possible_name_cols:
                        name_col = possible_name_cols[0]
                        map_df = csv_data['desc_tasks'][[name_col]].copy()
                        map_df['task_name_norm'] = map_df[name_col].astype(str).str.strip().str.lower()
                        task_completions_df['task_name_norm'] = task_completions_df['task_name'].astype(
                            str).str.strip().str.lower()
                        task_completions_df = task_completions_df.merge(map_df[['task_name_norm']], on='task_name_norm',
                                                                        how='left')
                        task_completions_df.drop(columns=['task_name_norm'], inplace=True)
                except Exception as _e:
                    # Mapping is best-effort; continue without failing
                    pass

            # If we have no task records, still generate provider chart with zeros if desc_tasks has providers; otherwise skip other plots
            if task_completions_df.empty:
                logger.warning(
                    "No task completion records (rules) found in rewards; generating provider chart with zero counts if possible")
                try:
                    provider_counts = None
                    if 'desc_tasks' in csv_data and not csv_data['desc_tasks'].empty:
                        tasks_desc_df = csv_data['desc_tasks'].copy()
                        col_map = {c.lower(): c for c in tasks_desc_df.columns}
                        data_providers_col = col_map.get('dataproviders')
                        if data_providers_col is not None:
                            def _parse_providers(val):
                                if pd.isna(val):
                                    return []
                                if isinstance(val, list):
                                    return [str(x).strip() for x in val if str(x).strip()]
                                s = str(val).strip()
                                try:
                                    js = re.sub("'", '"', s)
                                    parsed = json.loads(js) if (js.startswith('[') and js.endswith(']')) else None
                                    if isinstance(parsed, list):
                                        return [str(x).strip() for x in parsed if str(x).strip()]
                                except Exception:
                                    pass
                                parts = re.split(r'[;,]', s)
                                return [p.strip() for p in parts if p.strip()]

                            tasks_desc_df['providers_list'] = tasks_desc_df[data_providers_col].apply(_parse_providers)
                            try:
                                all_providers = sorted(
                                    {str(p).strip() for p in tasks_desc_df.explode('providers_list')['providers_list']
                                     if pd.notna(p) and str(p).strip() != ''})
                            except Exception:
                                all_providers = []
                            if all_providers:
                                import pandas as _pd
                                provider_counts = _pd.Series([0] * len(all_providers), index=all_providers)
                    else:
                        logger.warning("desc_tasks sheet not found; cannot build provider chart")
                    if provider_counts is not None and not provider_counts.empty:
                        def plot_tasks_by_provider():
                            plt.figure(figsize=(12, 6))
                            _ = provider_counts.plot(kind='bar', colormap=BAR_COLORMAP)
                            plt.title('Tasks Performed by Users per Data Provider')
                            plt.xlabel('Data Provider')
                            plt.ylabel('Tasks Performed')
                            plt.xticks(rotation=45)

                        create_and_save_figure(
                            plot_tasks_by_provider,
                            os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'tasks_by_provider.png'),
                            figsize=(12, 6)
                        )
                        logger.info('Generated tasks_by_provider chart with zero counts for all providers')
                except Exception as _e:
                    logger.error(f"Error generating zero-count provider chart: {_e}")
            else:
                # Unique user x task (by task name) and per-day counts
                unique_task_completions = task_completions_df.dropna(subset=['task_name']).drop_duplicates(
                    subset=['pid', 'task_name'])

                # Tasks performed per data provider chart (matched via desc_tasks -> dataproviders by task name)
                try:
                    provider_counts = None
                    if 'desc_tasks' in csv_data and not csv_data['desc_tasks'].empty:
                        tasks_desc_df = csv_data['desc_tasks'].copy()
                        # Build case-insensitive column map
                        col_map = {c.lower(): c for c in tasks_desc_df.columns}
                        # Prefer 'name' as instructed; fallback to common alternatives if needed
                        name_col = col_map.get('name') or col_map.get('label') or col_map.get('rule') or col_map.get(
                            'task') or col_map.get('task_name')
                        data_providers_col = col_map.get('dataproviders')
                        if name_col is None or data_providers_col is None:
                            logger.warning(
                                "desc_tasks sheet present but missing 'name' or 'dataproviders' for provider mapping; skipping tasks_by_provider chart")
                        else:
                            def _norm_name(s):
                                return re.sub(r'\s+', ' ', str(s)).strip().lower()

                            def _parse_providers(val):
                                if pd.isna(val):
                                    return []
                                if isinstance(val, list):
                                    return [str(x).strip() for x in val if str(x).strip()]
                                s = str(val).strip()
                                try:
                                    js = re.sub("'", '"', s)
                                    parsed = json.loads(js) if (js.startswith('[') and js.endswith(']')) else None
                                    if isinstance(parsed, list):
                                        return [str(x).strip() for x in parsed if str(x).strip()]
                                except Exception:
                                    pass
                                parts = re.split(r'[;,]', s)
                                return [p.strip() for p in parts if p.strip()]

                            # Normalize names on both sides
                            tasks_desc_df['task_name_norm'] = tasks_desc_df[name_col].apply(_norm_name)
                            task_completions_df['task_name_norm'] = task_completions_df['task_name'].apply(_norm_name)
                            tasks_desc_df['providers_list'] = tasks_desc_df[data_providers_col].apply(_parse_providers)
                            # Build full provider list once
                            try:
                                all_providers = sorted(
                                    {str(p).strip() for p in tasks_desc_df.explode('providers_list')['providers_list']
                                     if pd.notna(p) and str(p).strip() != ''})
                            except Exception:
                                all_providers = []
                            # Map providers to completions by normalized name
                            providers_map = tasks_desc_df.set_index('task_name_norm')['providers_list'].to_dict()
                            task_completions_df['providers_list'] = task_completions_df['task_name_norm'].map(
                                providers_map)
                            mapped_df = task_completions_df.dropna(subset=['providers_list']).copy()
                            if not mapped_df.empty:
                                # Explode to provider rows; count occurrences (each reward entry)
                                exploded = mapped_df.explode('providers_list').rename(
                                    columns={'providers_list': 'provider'})
                                # Count occurrences per provider from completions
                                provider_counts = exploded.dropna(subset=['provider']).groupby('provider').size()
                                # Ensure all providers are present on x-axis (include zeros) and sort alphabetically
                                if all_providers:
                                    provider_counts = provider_counts.reindex(all_providers, fill_value=0)
                                else:
                                    provider_counts = provider_counts.sort_values(ascending=False)
                            else:
                                # No mapped completions; build zero series if providers exist
                                if all_providers:
                                    import pandas as _pd
                                    provider_counts = _pd.Series([0] * len(all_providers), index=all_providers)
                    else:
                        logger.warning("desc_tasks sheet not found; skipping tasks_by_provider chart")
                    # Plot if we have counts
                    if provider_counts is not None and not provider_counts.empty:
                        def plot_tasks_by_provider():
                            plt.figure(figsize=(12, 6))
                            _ = provider_counts.plot(kind='bar', colormap=BAR_COLORMAP)
                            plt.title('Tasks Performed by Users per Data Provider')
                            plt.xlabel('Data Provider')
                            plt.ylabel('Tasks Performed')
                            plt.xticks(rotation=45)

                        create_and_save_figure(
                            plot_tasks_by_provider,
                            os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'tasks_by_provider.png'),
                            figsize=(12, 6)
                        )
                        logger.info('Generated tasks_by_provider chart (including zero-count providers)')
                    else:
                        logger.warning('No provider counts available for tasks_by_provider plot')
                except Exception as _e:
                    logger.error(f"Error computing tasks_by_provider from desc_tasks mapping: {_e}")

                # Plot: Number of tasks completed per day (unique user-task per day)
                if 'date' in task_completions_df.columns and task_completions_df['date'].notna().any():
                    per_day = task_completions_df.dropna(subset=['date', 'task_name']).drop_duplicates(
                        subset=['pid', 'task_name', 'date']).groupby('date').size().sort_index()

                    def plot_tasks_completed_per_day():
                        plt.figure(figsize=(12, 6))
                        series = per_day.sort_index()
                        date_labels = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in series.index]
                        x_pos = range(len(series))
                        plt.bar(x_pos, series.values, color=SINGLE_SERIES_COLOR, width=0.6)
                        plt.title('Tasks Completed per Day')
                        plt.xlabel('Date')
                        plt.ylabel('Tasks Completed (unique user-task)')
                        ax = plt.gca()
                        ax.set_xticks(list(x_pos))
                        ax.set_xticklabels(date_labels)
                        ax.tick_params(axis='x', rotation=90)
                        for tick in ax.get_xticklabels():
                            tick.set_horizontalalignment('center')
                        plt.margins(x=0.01)
                        plt.tight_layout(pad=2.0)

                    create_and_save_figure(
                        plot_tasks_completed_per_day,
                        os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'tasks_completed_per_day.png'),
                        figsize=(12, 6)
                    )
                else:
                    logger.warning("Date information not available for per-day task completion plot")

                # Plot: Number of tasks completed per user (unique tasks per user)
                per_user = unique_task_completions.groupby('pid').size().sort_values(ascending=False)
                if not per_user.empty:
                    def plot_tasks_completed_per_user():
                        plt.figure(figsize=(12, 6))
                        plot_data = per_user
                        # Limit to top 50 players to keep plot readable
                        if len(plot_data) > 50:
                            plot_data = plot_data.head(50)
                            plt.title('Tasks Completed per Player (Top 50 Players)')
                        else:
                            plt.title('Tasks Completed per Player')
                        plot_data.plot(kind='bar', colormap=BAR_COLORMAP)
                        plt.xlabel('Player ID')
                        plt.ylabel('Tasks Completed (unique tasks)')
                        plt.xticks(rotation=45)

                    create_and_save_figure(
                        plot_tasks_completed_per_user,
                        os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'tasks_completed_per_player.png'),
                        figsize=(12, 6)
                    )
                else:
                    logger.warning("No data available for per-user task completion plot")

                # Compute participants per task for summary
                participants_per_task = unique_task_completions.groupby('task_name')['pid'].nunique().sort_values(
                    ascending=False)
                participants_plot_df = participants_per_task.reset_index().rename(
                    columns={'task_name': 'label', 'pid': 'participants'})

    except Exception as e:
        logger.error(f"Error generating completed tasks plots: {e}")


def analyze_geofence_data(json_data):
    """Analyze geofence data from JSON files

    This function analyzes geofence data from JSON files, including:
    - Speed distribution
    - Altitude distribution
    - Error margin analysis
    - Temporal patterns in geofence data
    - Movement trajectories
    - 3D visualizations
    - Correlation analyses

    Args:
        json_data: Dictionary of JSON data
    """

    # Combine geofence data from all users
    geofence_dfs = []

    for key, data in json_data.items():
        # Debug: Print key and data type
        if 'geofence' in key.lower() and isinstance(data, pd.DataFrame):
            # Extract player ID from the key if possible
            try:
                # Assuming key format is like "player_123_geofence" (also supports legacy "user_123_geofence")
                parts = key.split('_')

                if len(parts) >= 3 and parts[0] in ('player', 'user'):
                    player_id = parts[1]

                    # Add identifier columns to the DataFrame (both for backward compatibility)
                    data_copy = data.copy()
                    data_copy['player_id'] = player_id
                    data_copy['user_id'] = player_id

                    # Ensure all required columns exist
                    required_columns = ['LATITUDE', 'LONGITUDE', 'ALTITUDE', 'SPEED', 'ERROR', 'TIMESTAMP']
                    for col in required_columns:
                        if col not in data_copy.columns:
                            if col == 'TIMESTAMP':
                                # Use a default timestamp if missing
                                data_copy[col] = pd.NaT
                            else:
                                # Use 0 for missing numeric columns
                                data_copy[col] = 0

                    geofence_dfs.append(data_copy)
                else:
                    logger.warning(f"Key {key} does not match expected format 'player_X_geofence' (or legacy 'user_X_geofence')")
            except (IndexError, ValueError) as e:
                logger.warning(f"Error extracting player ID from key {key}: {e}")

    # Debug: Check if geofence_dfs is empty
    if not geofence_dfs:
        logger.warning("No geofence data found, geofence_dfs is empty")
        return

    # Combine all geofence dataframes
    all_geofence_data = pd.concat(geofence_dfs, ignore_index=True)

    # Convert TIMESTAMP to datetime if it's not already
    if 'TIMESTAMP' in all_geofence_data.columns:
        all_geofence_data['TIMESTAMP'] = pd.to_datetime(all_geofence_data['TIMESTAMP'], errors='coerce')

    # Ensure numeric dtypes for geofence measures to avoid errors during quantiles/plots
    for col in ['LATITUDE', 'LONGITUDE', 'ALTITUDE', 'SPEED', 'ERROR']:
        if col in all_geofence_data.columns:
            all_geofence_data[col] = pd.to_numeric(all_geofence_data[col], errors='coerce')

    # Drop rows with missing SPEED for outlier computation
    speed_series = all_geofence_data['SPEED'].dropna()
    if speed_series.empty:
        logger.warning("No valid SPEED values in geofence data; skipping geofence analysis")
        return

    # Clean data: Remove extreme outliers in SPEED
    # Calculate Q1, Q3 and IQR for SPEED
    Q1 = speed_series.quantile(0.25)
    Q3 = speed_series.quantile(0.75)
    IQR = Q3 - Q1

    # Define bounds for outliers (using 3*IQR as a more lenient threshold)
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR

    # Filter out extreme outliers
    filtered_geofence_data = all_geofence_data[
        (all_geofence_data['SPEED'] >= lower_bound) &
        (all_geofence_data['SPEED'] <= upper_bound)
        ].copy()

    # Log how many outliers were removed
    outliers_count = len(all_geofence_data) - len(filtered_geofence_data)
    if outliers_count > 0:
        # 6. Temporal analysis (if timestamp data is available)
        if 'TIMESTAMP' in filtered_geofence_data.columns and not filtered_geofence_data['TIMESTAMP'].isna().all():
            # Add hour of day column
            filtered_geofence_data['hour_of_day'] = filtered_geofence_data['TIMESTAMP'].dt.hour
            # Ensure hour_of_day is an ordered categorical with all hours 0-23 so plots show all ticks
            filtered_geofence_data['hour_of_day'] = pd.Categorical(
                filtered_geofence_data['hour_of_day'], categories=list(range(24)), ordered=True
            )

            def plot_hourly_activity():
                plt.figure(figsize=(12, 6))
                hourly_counts = filtered_geofence_data['hour_of_day'].value_counts().sort_index()
                # Ensure all hours 0-23 are represented, even if count is 0
                hourly_counts = hourly_counts.reindex(range(24), fill_value=0)
                sns.barplot(x=list(hourly_counts.index), y=list(hourly_counts.values))
                plt.title('Geofence Activity by Hour of Day')
                plt.xlabel('Hour of Day')
                plt.ylabel('Number of Records')
                plt.xticks(range(0, 24))
                plt.xlim(-0.5, 23.5)
                plt.grid(True, alpha=0.3)

            create_and_save_figure(
                plot_hourly_activity,
                os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'geofence_hourly_activity.png')
            )

            # Speed by hour of day
            def plot_speed_by_hour():
                plt.figure(figsize=(12, 6))
                sns.boxplot(data=filtered_geofence_data, x='hour_of_day', y='SPEED', order=list(range(24)))
                plt.title('Speed by Hour of Day')
                plt.xlabel('Hour of Day')
                plt.ylabel('Speed')
                plt.xticks(range(0, 24))
                plt.xlim(-0.5, 23.5)
                plt.grid(True, alpha=0.3)

            create_and_save_figure(
                plot_speed_by_hour,
                os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'geofence_speed_by_hour.png')
            )

        # 10. Movement trajectory visualization (if timestamps are available)
        if 'TIMESTAMP' in filtered_geofence_data.columns and not filtered_geofence_data['TIMESTAMP'].isna().all():
            try:
                # Sort data by timestamp
                trajectory_data = filtered_geofence_data.sort_values('TIMESTAMP')

                def plot_movement_trajectory():
                    plt.figure(figsize=(12, 8))
                    plt.plot(trajectory_data['LONGITUDE'], trajectory_data['LATITUDE'], 'b-', alpha=0.7)
                    plt.scatter(trajectory_data['LONGITUDE'], trajectory_data['LATITUDE'],
                                c=trajectory_data['SPEED'], cmap='viridis', alpha=0.8)
                    plt.colorbar(label='Speed')
                    plt.title('Movement Trajectory (Color indicates Speed)')
                    plt.xlabel('Longitude')
                    plt.ylabel('Latitude')
                    plt.grid(True, alpha=0.3)

                create_and_save_figure(
                    plot_movement_trajectory,
                    os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'geofence_movement_trajectory.png')
                )
            except Exception as e:
                logger.warning(f"Could not create movement trajectory plot: {e}")

        # 12. 3D visualization of geofence data
        try:
            from mpl_toolkits.mplot3d import Axes3D

            def plot_3d_visualization():
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                scatter = ax.scatter(filtered_geofence_data['LONGITUDE'],
                                     filtered_geofence_data['LATITUDE'],
                                     filtered_geofence_data['ALTITUDE'],
                                     c=filtered_geofence_data['SPEED'],
                                     cmap='viridis',
                                     alpha=0.6)
                plt.colorbar(scatter, label='Speed')
                ax.set_title('3D Geofence Visualization')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.set_zlabel('Altitude')

            create_and_save_figure(
                plot_3d_visualization,
                os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'geofence_3d_visualization.png')
            )
        except Exception as e:
            logger.warning(f"Could not create 3D visualization: {e}")
    else:
        logger.warning("No geofence data available for analysis")


def analyze_day_aggregate_steps(json_data: Dict[str, Union[pd.DataFrame, Dict]],
                                campaign_metrics: Optional[Dict] = None) -> Optional[pd.DataFrame]:
    """
	Parse DAY_AGGREGATE JSON files (and RUN/WALK variants) to compute and plot steps trend per user and overall.

	Sources supported:
	- Separate files like 'user_<id>_day_aggregate*.json' loaded as DataFrames in json_data.
	- Combined 'user_<id>_all_data.json' dicts containing lists under keys 'day_aggregate', 'day_aggregate_walk', 'day_aggregate_run'.

	Outputs:
	- Saves one figure under OUTPUT_VISUALIZATIONS_DIR:
	  * steps_trend.png          (total steps across users per day over full campaign date range)
	- Returns the combined DataFrame with columns: ['date', 'user_id', 'steps']
	"""
    try:
        records: List[pd.DataFrame] = []

        def _extract_from_df(df: pd.DataFrame, user_id: str) -> Optional[pd.DataFrame]:
            if df is None or df.empty:
                return None
            local = df.copy()
            # Determine date column preference: X_DATE, DATE, START_DATE, then END_DATE
            date_candidates = ['X_DATE', 'DATE', 'START_DATE', 'END_DATE']
            date_col = next((c for c in date_candidates if c in local.columns), None)
            if date_col is None:
                return None
            # Parse to datetime.date
            local['__date'] = pd.to_datetime(local[date_col], errors='coerce').dt.date
            # Fallback: if parsing failed and there is an END_DATE, try it
            if local['__date'].isna().all() and 'END_DATE' in local.columns and date_col != 'END_DATE':
                local['__date'] = pd.to_datetime(local['END_DATE'], errors='coerce').dt.date
            # Determine steps column (support STEPS_SUM, STEP_SUM, or STEPS)
            steps_candidates = ['STEPS_SUM', 'STEP_SUM', 'STEPS']
            steps_col = next((c for c in steps_candidates if c in local.columns), None)
            if steps_col is None:
                return None
            out = local[['__date', steps_col]].dropna(subset=['__date']).rename(columns={steps_col: 'steps'})
            # Coerce steps to numeric to avoid string concatenation during sum
            out['steps'] = pd.to_numeric(out['steps'], errors='coerce').fillna(0)
            if out.empty:
                return None
            out['user_id'] = str(user_id)
            return out

        for key, data in json_data.items():
            k_lower = key.lower()
            # Try to extract player id from key names like 'player_443_day_aggregate' or legacy 'user_443_all_data'
            m = re.match(r"(?:player|user)_(\d+)_", key)
            user_id = m.group(1) if m else key

            if isinstance(data, pd.DataFrame):
                if 'day_aggregate' in k_lower:
                    df_out = _extract_from_df(data, user_id)
                    if df_out is not None:
                        records.append(df_out)
            elif isinstance(data, dict) and (k_lower.startswith('player_') or k_lower.startswith('user_')) and k_lower.endswith('_all_data'):
                for section in ['day_aggregate', 'day_aggregate_walk', 'day_aggregate_run']:
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
            logger.warning("No DAY_AGGREGATE step data found in JSON inputs; skipping daily steps plots")
            return None

        combined = pd.concat(records, ignore_index=True)
        # Aggregate across potential duplicates and variants (run/walk/base)
        combined = combined.groupby(['__date', 'user_id'], as_index=False)['steps'].sum()
        combined = combined.rename(columns={'__date': 'date'})
        # Ensure numeric dtype for steps
        combined['steps'] = pd.to_numeric(combined['steps'], errors='coerce').fillna(0)
        # Sort by date then user
        combined = combined.sort_values(['date', 'user_id'])

        # Build full campaign date range
        def _parse_ts(val):
            return pd.to_datetime(val, errors='coerce')

        inferred = combined.groupby('date', as_index=False)['steps'].sum()
        min_ts = _parse_ts(inferred['date']).min()
        max_ts = _parse_ts(inferred['date']).max()
        start_ts = None
        end_ts = None
        if campaign_metrics and isinstance(campaign_metrics, dict):
            start_ts = _parse_ts(campaign_metrics.get('start_date'))
            end_ts = _parse_ts(campaign_metrics.get('end_date'))
        # Fallbacks if campaign dates missing
        if pd.isna(start_ts) or start_ts is None:
            start_ts = min_ts
        if pd.isna(end_ts) or end_ts is None:
            end_ts = max_ts
        # Ensure valid range
        if pd.isna(start_ts) or pd.isna(end_ts) or start_ts > end_ts:
            start_ts, end_ts = min_ts, max_ts
        full_range = pd.date_range(start=start_ts.normalize(), end=end_ts.normalize(), freq='D')

        # Plot: total steps across users per day (reindexed to full range)
        total = inferred.copy()
        total['date'] = pd.to_datetime(total['date'])
        total = total.set_index('date').reindex(full_range).fillna(0).rename_axis('date').reset_index()

        def plot_total():
            ax = plt.gca()
            ax.plot(total['date'], total['steps'], marker='o')
            ax.set_title('Steps Trend (All Users)')
            ax.set_xlabel('Date')
            ax.set_ylabel('Steps')
            # Explicitly show every campaign date on x-axis
            try:
                # ticks as full daily range
                ticks = pd.to_datetime(list(full_range))
                labels = [d.strftime('%Y-%m-%d') for d in ticks]
                ax.set_xticks(ticks)
                ax.set_xticklabels(labels, rotation=90, ha='center')
                ax.set_xlim(ticks[0], ticks[-1])
            except Exception:
                plt.xticks(rotation=90)

        create_and_save_figure(
            plot_total,
            os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'steps_trend.png'),
            figsize=PLOT_FIGSIZE_ACTIVITY_TYPES_STACKED
        )

        return combined
    except Exception as e:
        logger.error(f"Error analyzing DAY_AGGREGATE steps: {e}")
        return None


def generate_analysis_report(csv_data, json_data, activities, campaign_metrics, dropout_metrics, joining_metrics=None):
    """
    Generate a report of which analyses were performed and which weren't, with reasons.

    Args:
        csv_data: Dictionary of DataFrames from Excel files
        json_data: Dictionary of data from JSON files
        activities: DataFrame of activities
        campaign_metrics: Dictionary of campaign metrics
        dropout_metrics: Dictionary of dropout metrics
        joining_metrics: Dictionary of joining metrics

    Returns:
        str: Path to the generated report file.
    """

    # Ensure data_analysis directory exists
    data_analysis_dir = ensure_dir(os.path.join(PROJECT_ROOT, 'data_analysis'))
    report_path = os.path.join(data_analysis_dir, 'analysis_report.txt')

    # Lists to track performed and skipped analyses
    performed_analyses = []
    skipped_analyses = []

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("GameBus Data Analysis Report\n")
        f.write("===========================\n\n")

        f.write("1. Data Loading\n")
        f.write("---------------\n")
        if csv_data:
            f.write(f"✓ Excel data loaded successfully: {len(csv_data)} sheets\n")
            performed_analyses.append("Excel data loading")
        else:
            f.write("✗ No Excel data loaded\n")
            skipped_analyses.append(("Excel data loading", "No Excel files found or error during loading"))

        if json_data:
            f.write(f"✓ JSON data loaded successfully: {len(json_data)} files\n")
            performed_analyses.append("JSON data loading")
        else:
            f.write("✗ No JSON data loaded\n")
            skipped_analyses.append(("JSON data loading", "No JSON files found or error during loading"))

        # 2. Campaign Summary
        f.write("\n2. Campaign Summary\n")
        f.write("--------------------\n")
        try:
            # Safe access to campaign_metrics
            campaign_name = (campaign_metrics or {}).get('name', 'GameBus Campaign')
            campaign_abbr = (campaign_metrics or {}).get('abbreviation', '')
            start_date = (campaign_metrics or {}).get('start_date', 'Unknown')
            end_date = (campaign_metrics or {}).get('end_date', 'Unknown')
            length_days = (campaign_metrics or {}).get('length_days', 0)
            unique_users = (campaign_metrics or {}).get('unique_users', 0)

            # Header: campaign name (with abbreviation if available)
            if campaign_abbr:
                f.write(f"{campaign_name} ({campaign_abbr})\n\n")
            else:
                f.write(f"{campaign_name}\n\n")

            # Core metrics
            f.write(f"Start Date: {start_date}\n")
            f.write(f"End Date: {end_date}\n")
            f.write(f"Length: {length_days} days\n")

            # Determine total number of users strictly from campaign_data.xlsx 'aggregation' sheet using 'pid'
            agg_total_users = None
            try:
                if isinstance(csv_data, dict):
                    # Find the sheet named exactly 'aggregation' (case-insensitive)
                    agg_key = next((k for k in csv_data.keys() if isinstance(k, str) and k.lower() == 'aggregation'),
                                   None)
                    if agg_key:
                        df_agg = csv_data[agg_key]
                        if isinstance(df_agg, pd.DataFrame) and not df_agg.empty:
                            cols = list(df_agg.columns)
                            # Count unique user ids from column 'pid' (each row corresponds to a user)
                            pid_candidates = [c for c in cols if isinstance(c, str) and c.lower() == 'pid']
                            if pid_candidates:
                                pid_col = pid_candidates[0]
                                agg_total_users = int(pd.Series(df_agg[pid_col]).nunique())
                            else:
                                logger.warning("Aggregation sheet found but no 'pid' column present.")
            except Exception as e:
                logger.warning(f"Could not determine total users from aggregation sheet: {e}")

            # Prefer aggregation-derived total users when available, otherwise use unique active users
            total_users_for_report = agg_total_users if isinstance(agg_total_users,
                                                                   int) and agg_total_users > 0 else unique_users
            f.write(f"Total number of Players in the Campaign: {total_users_for_report}\n")

            # Active/Passive users summary (by rewards)
            active_users_count = (campaign_metrics or {}).get('active_users_count', None)
            passive_users_count = (campaign_metrics or {}).get('passive_users_count', None)
            if active_users_count is not None and passive_users_count is not None:
                f.write(f"Active Players (rewarded): {active_users_count}\n")
                f.write(f"Passive Players (enrolled, never rewarded): {passive_users_count}\n")
                passive_ids = (campaign_metrics or {}).get('passive_user_ids', [])
                if isinstance(passive_ids, list) and len(passive_ids) > 0:
                    preview = ', '.join(map(str, passive_ids[:10]))
                    f.write(f"Passive Player IDs: {preview}\n")
        except Exception as e:
            logger.error(f"Error writing campaign summary to analysis report: {e}")

    return report_path


def create_complete_report(csv_data, json_data, activities=None):
    """
    Create a complete analysis report with all required sections.

    Args:
        csv_data: Dictionary of DataFrames from Excel files
        json_data: Dictionary of data from JSON files
        activities: Either a DataFrame of activities or the full result tuple from analyze_activities

    Returns:
        str: Path to the generated report file.
    """

    # Unpack inputs depending on what was provided
    acts = None
    campaign_metrics = None
    dropout_metrics = None
    joining_metrics = None

    if isinstance(activities, tuple) and len(activities) >= 5:
        # analyze_activities returns: (activities, unique_users_count, dropout_metrics, joining_metrics, campaign_metrics)
        acts, _unique_users_count, dropout_metrics, joining_metrics, campaign_metrics = activities
    else:
        acts = activities

    # Generate and return the report path
    return generate_analysis_report(csv_data, json_data, acts, campaign_metrics, dropout_metrics, joining_metrics)

def main():
    """Main function to run the analysis"""
    try:
        # Ensure output directories exist
        ensure_output_dirs()

        # Load data
        try:
            csv_data = load_excel_files()
            if not csv_data:
                logger.warning("No Excel data loaded")
            else:
                logger.info(f"Loaded {len(csv_data)} Excel sheets")
        except Exception as e:
            logger.error(f"Error loading Excel files: {e}")
            csv_data = {}

        try:
            json_data = load_json_files()
            if not json_data:
                logger.warning("No JSON data loaded")
            else:
                logger.info(f"Loaded {len(json_data)} JSON files")
        except Exception as e:
            logger.error(f"Error loading JSON files: {e}")
            json_data = {}

        # Analyze activities data
        try:
            result = analyze_activities(csv_data, json_data)
            if result:
                activities, unique_users_count, dropout_metrics, joining_metrics, campaign_metrics = result
            else:
                logger.warning("Activities analysis returned no results")
                activities = None
                unique_users_count = 0
                dropout_metrics = None
                joining_metrics = None
                campaign_metrics = None
        except Exception as e:
            logger.error(f"Error analyzing activities data: {e}")
            activities = None
            unique_users_count = 0
            dropout_metrics = None
            joining_metrics = None
            campaign_metrics = None

        # Analyze geofence data
        try:
            # Debug: Check if json_data is empty
            analyze_geofence_data(json_data)
        except Exception as e:
            logger.error(f"Error analyzing geofence data: {e}")
            # Debug: Print the full error traceback
            import traceback
            logger.error(f"Error traceback: {traceback.format_exc()}")

        # Analyze daily steps from DAY_AGGREGATE files
        try:
            analyze_day_aggregate_steps(json_data, campaign_metrics)
        except Exception as e:
            logger.error(f"Error analyzing daily steps from DAY_AGGREGATE: {e}")

        # Analyze visualizations, challenges, and tasks
        try:
            analyze_visualizations_challenges_tasks(csv_data)
        except Exception as e:
            logger.error(f"Error analyzing visualizations, challenges, and tasks: {e}")

        # Generate analysis report
        try:
            # Pass the entire result tuple to create_complete_report
            if result:
                report_path = create_complete_report(csv_data, json_data, result)
            else:
                report_path = create_complete_report(csv_data, json_data, activities)
        except Exception as e:
            logger.error(f"Error generating complete report: {e}")
            report_path = None

        # Print summary
        logger.info("Analysis complete")
        print("\nAnalysis complete.")
        print(f"- Visualizations saved to '{OUTPUT_VISUALIZATIONS_DIR}' directory.")

    except Exception as e:
        logger.error(f"Unexpected error in main function: {e}")
        print(f"An error occurred during analysis. See log file for details.")

if __name__ == "__main__":
    main()
