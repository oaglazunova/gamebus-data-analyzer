"""
GameBus Health Behavior Mining Data Analysis Script

This script analyzes data from the GameBus Health Behavior Mining project, including:
1. Excel files in the config directory containing campaign, and activity data
2. JSON files in the data_raw directory containing player-specific data like activity types, and mood logs

The script generates:
1. Descriptive statistics tables for each variable, providing insights into:
   - Central tendency (mean, median)
   - Dispersion (standard deviation, min, max)
   - Distribution characteristics (quartiles, skewness)

2. Various visualizations to provide insights into:
   - Activity patterns and distributions
   - Player engagement and participation
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
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import tabulate
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

# Add console handler
console_handler = logging.StreamHandler()
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

# Colormaps for different types of visualizations
BAR_COLORMAP = 'viridis'  # Perceptually uniform, good for categorical data
PIE_COLORMAP = 'tab10'    # Distinct colors for categorical data
CORRELATION_HEATMAP_COLORMAP = 'coolwarm'  # Diverging colormap for correlation matrices
SEQUENTIAL_HEATMAP_COLORMAP = 'turbo'  # Sequential colormap for heatmaps showing magnitude
LINE_COLORMAP = 'tab10'   # Good for distinguishing multiple lines
HISTOGRAM_COLORMAP = 'Blues'  # Sequential colormap for frequency distributions
BOX_COLORMAP = 'Set2'     # Categorical colormap for categorical comparisons
REGRESSION_COLORMAP = 'Blues'  # Sequential colormap for showing relationships

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
OUTPUT_VISUALIZATIONS_DIR = os.path.join(PROJECT_ROOT, 'data_analysis', 'visualizations')
OUTPUT_STATISTICS_DIR = os.path.join(PROJECT_ROOT, 'data_analysis', 'statistics')
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data_raw')
# No hardcoded assumptions as per requirements

# Small utilities for filesystem setup

def ensure_dir(path: str) -> str:
	"""Create directory if it doesn't exist and return the same path."""
	os.makedirs(path, exist_ok=True)
	return path


def ensure_output_dirs() -> None:
	"""Ensure all output directories used by this module exist."""
	# Parent data_analysis directory
	ensure_dir(os.path.join(PROJECT_ROOT, 'data_analysis'))
	# Subdirectories
	ensure_dir(OUTPUT_VISUALIZATIONS_DIR)
	ensure_dir(OUTPUT_STATISTICS_DIR)

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
	plt.tight_layout()
	plt.savefig(filename, bbox_inches='tight')
	plt.close()  # Close the figure to free memory
	logger.info(f"Figure saved to: {filename}")

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

def perform_bivariate_analysis(df, title, filename=None):
	"""
    Compute pairwise relationships between numeric variables in a DataFrame and return the
    correlation matrix. This function no longer renders or saves any plots.

    What it does now:
    - Selects numeric columns (int64, float64) from the provided DataFrame.
    - If at least two numeric columns are present, computes the Pearson correlation
      matrix between those columns and rounds the coefficients to two decimals.
    - Returns the correlation matrix for further use by callers.

    Args:
        df: pandas.DataFrame to analyze.
        title: Unused for plotting; retained for backward compatibility and potential logging.
        filename: Unused parameter kept for backward compatibility (ignored).

    Returns:
        pandas.DataFrame | None: The rounded Pearson correlation matrix if computed; otherwise None.
    """
	# Select numerical columns
	numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

	# Correlation analysis for numerical variables
	if len(numerical_cols) >= 2:
		corr_matrix = df[numerical_cols].corr(method='pearson')
		# Round correlation values to 2 decimal places
		corr_matrix = corr_matrix.round(2)
		logger.info(f"Computed correlation matrix for bivariate analysis on '{title}' with {len(numerical_cols)} numeric columns")
		return corr_matrix
	else:
		logger.info("perform_bivariate_analysis: fewer than two numeric columns; skipping correlation computation")
		return None

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
				logger.info(f"Loaded sheet '{sheet_name}' from {path} with {len(df)} rows and {len(df.columns)} columns")
			except Exception as e:
				logger.error(f"Error loading sheet '{sheet_name}' from {path}: {e}")
	except Exception as e:
		logger.error(f"Error opening {path}: {e}")
	return sheets


def load_excel_files() -> Dict[str, pd.DataFrame]:
	"""
    Load campaign data from Excel files in the config directory.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary of DataFrames, keyed by sheet name
    """
	# Ensure config directory exists
	ensure_dir(CONFIG_DIR)

	data_dict: Dict[str, pd.DataFrame] = {}
	data_dict.update(_load_excel_sheets(os.path.join(CONFIG_DIR, 'campaign_data.xlsx')))
	data_dict.update(_load_excel_sheets(os.path.join(CONFIG_DIR, 'campaign_desc.xlsx'), prefix="desc_"))

	if not data_dict:
		logger.warning("No data loaded from Excel files")

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

	logger.info(f"Found {len(json_files)} JSON files in {RAW_DATA_DIR}")

	for file in json_files:
		# Extract player ID and data type from filename
		filename = os.path.basename(file)
		key = filename.split('.')[0]  # Remove .json extension

		# Debug: Print filename and key
		logger.info(f"Processing file: {filename}, key: {key}")

		try:
			with open(file, 'r') as f:
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

				df = pd.DataFrame(data)
				if df.empty:
					logger.warning(f"Empty DataFrame created from {filename}, skipping")
					continue

				data_dict[key] = df
				logger.info(f"Loaded {filename} with {len(df)} rows and {len(df.columns)} columns")
			else:
				if not data:  # Skip empty dictionaries
					logger.warning(f"Empty data dictionary in {filename}, skipping")
					continue

				data_dict[key] = data
				logger.info(f"Loaded {filename} (not converted to DataFrame)")
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


def analyze_activities(csv_data: Dict[str, pd.DataFrame], json_data: Dict[str, Union[pd.DataFrame, Dict]] = None) -> Optional[Tuple]:
	"""
    Analyze activities data from the campaign data.

    This function analyzes activity data, including:
    - Activity types and frequencies
    - Points distribution
    - Player engagement patterns
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
            - usage_metrics dictionary
            - dropout_metrics dictionary
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

	logger.info(f"Analyzing activities data with {len(activities)} rows")

	# Process the data
	try:
		# Convert createdAt to datetime
		activities['createdAt'] = pd.to_datetime(activities['createdAt'])
		activities['date'] = activities['createdAt'].dt.date
		activities['hour'] = activities['createdAt'].dt.hour

		# Extract points from rewardedParticipations column
		activities['points'] = activities['rewardedParticipations'].apply(extract_points)

		# Extract detailed reward information
		logger.info("Extracting detailed reward information")
		activities['detailed_rewards'] = activities['rewardedParticipations'].apply(extract_detailed_rewards)

		# Create a column with the number of rewards per activity
		activities['num_rewards'] = activities['detailed_rewards'].apply(len)
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
			logger.info("Dropped 'properties' column for statistics generation")

		generate_descriptive_stats(activities_for_stats, "Activities Data", None)
		logger.info("Generated descriptive statistics for activities data")

		# Skip bivariate analysis for activities data to avoid ANOVA test warnings
		# perform_bivariate_analysis(activities_for_stats, "Activities Data", None)

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
			logger.info(f"Created 'wave' column with values from 1 to {activities['wave'].max()}")

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
		logger.info("Created wave comparisons barplots")

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
				cmap = INDEPENDENT_COLORMAP(player_wave_counts.shape[1]) if callable(INDEPENDENT_COLORMAP) else INDEPENDENT_COLORMAP
				ax = player_wave_counts.plot(kind='bar', figsize=(12, 6), colormap=cmap)
				plt.title('Activities per Wave by Players')
				plt.xlabel('Wave')
				plt.ylabel('Number of Activities')
				# Place legend below the plot
				handles, labels = ax.get_legend_handles_labels()
				ncols = min(6, max(1, len(labels)))
				legend = plt.legend(handles, labels, title='Player ID', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=ncols, frameon=False)
				# Add a bit of extra bottom margin so the legend fits nicely
				plt.subplots_adjust(bottom=0.2)
				plt.tight_layout()

			create_and_save_figure(
				plot_wave_comparisons_by_player,
				f'{OUTPUT_VISUALIZATIONS_DIR}/wave_comparisons_by_player.png',
				figsize=(12, 8)
			)
			logger.info("Created wave comparisons by player barplots")

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
			cmap = INDEPENDENT_COLORMAP(type_wave_counts.shape[1]) if callable(INDEPENDENT_COLORMAP) else INDEPENDENT_COLORMAP
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
		logger.info("Created wave comparisons by activity type barplots")

		# Create wave comparisons of average points by activity type
		def plot_wave_points_by_activity_type():
			# Group by wave and type, calculate mean points
			# Note: Using all activity types may reduce readability if there are many types
			# Consider adding filtering if the visualization becomes too cluttered
			wave_type_points = activities.groupby(['wave', 'type'])['points'].mean().unstack()

			# Plot the data
			cmap = INDEPENDENT_COLORMAP(wave_type_points.shape[1]) if callable(INDEPENDENT_COLORMAP) else INDEPENDENT_COLORMAP
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
		logger.info("Created wave comparisons of points by activity type barplots")
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
			logger.info("Created activities over time visualization")
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
			logger.info("Created points by activity type visualization")
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
					plt.title('Total Rewarded Points by User (Top 50 Users)')
				else:
					plt.title('Total Rewarded Points by User')

				plot_data.plot(kind='bar', colormap=BAR_COLORMAP)
				plt.xlabel('User ID')
				plt.ylabel('Total Rewarded Points')
				plt.xticks(rotation=45)

			create_and_save_figure(
				plot_points_by_user,
				f'{OUTPUT_VISUALIZATIONS_DIR}/points_by_user.png',
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
			logger.info("Created rewards by activity type visualization (average points)")
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
			logger.info("Created points over time visualization")
	except Exception as e:
		logger.error(f"Error creating points over time visualization: {e}")
		# Continue with other visualizations even if this one fails

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
					plt.title('Number of Activities by User (Top 50 Users)')
				else:
					plt.title('Number of Activities by User')

				plot_data.plot(kind='bar', colormap=BAR_COLORMAP)
				plt.xlabel('User ID')
				plt.ylabel('Number of Activities')

			create_and_save_figure(
				plot_user_activity_distribution,
				f'{OUTPUT_VISUALIZATIONS_DIR}/user_activity_distribution.png',
				figsize=(12, 6)
			)
			logger.info("Created user activity distribution visualization")
	except Exception as e:
		logger.error(f"Error creating user activity distribution visualization: {e}")
		# Continue with other visualizations even if this one fails


	# 5. Activity Type Distribution by User (Bivariate Analysis)
	try:
		# This visualization shows how different users engage with different activity types
		# Helps identify user preferences and potential targeting opportunities
		if len(activities['pid'].unique()) < 2 or len(activities['type'].unique()) < 2:
			logger.warning("Insufficient data for activity type by user visualization (need at least 2 users and 2 activity types)")
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
						logger.warning("Some users have zero activities, removing them from normalization")
						# Filter out rows with zero sum
						user_type_counts = user_type_counts[row_sums > 0]
						row_sums = user_type_counts.sum(axis=1)

					user_type_counts_norm = user_type_counts.div(row_sums, axis=0)

					# Use all users as per requirements
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
							logger.info(f"Limited activity type heatmap to top 15 activity types out of {user_type_counts.shape[1]}")

						def plot_activity_type_by_user():
							# Adjust font size for annotations based on number of users
							# Start with font size 10 for <= 10 users, decrease as user count increases
							annot_fontsize = max(6, 10 - (len(valid_users) - 10) / 10)

							sns.heatmap(user_type_heatmap, cmap=SEQUENTIAL_HEATMAP_COLORMAP, 
									   annot=True, fmt='.2f', linewidths=0.5, 
									   annot_kws={'fontsize': annot_fontsize})
							plt.title(f'Activity Type Distribution by User (All {len(valid_users)} Users)')
							plt.xlabel('Activity Type')
							plt.ylabel('User ID')

						# Adjust figure size based on number of users
						# Base size is 14x8, but add 0.5 height for each additional 5 users beyond 10
						height = 8 + max(0, (len(valid_users) - 10) / 5 * 0.5)
						# Cap the height at a reasonable maximum
						height = min(height, 20)

						create_and_save_figure(
							plot_activity_type_by_user,
							f'{OUTPUT_VISUALIZATIONS_DIR}/activity_type_by_user.png',
							figsize=(14, height)
						)
						logger.info("Created activity type by user visualization")
				except Exception as e:
					logger.error(f"Error in normalization or heatmap creation: {e}")
	except Exception as e:
		logger.error(f"Error creating activity type by user visualization: {e}")
		# Continue with other visualizations even if this one fails

	# Calculate campaign metrics
	try:
		# Count unique users in the campaign
		unique_users_count = activities['pid'].nunique()
		logger.info(f"Number of unique users in the campaign: {unique_users_count}")

		# Calculate campaign length (days between first and last activity)
		campaign_start = activities['createdAt'].min()
		campaign_end = activities['createdAt'].max()
		campaign_length_days = (campaign_end - campaign_start).days

		logger.info(f"Campaign length: {campaign_length_days} days (from {campaign_start.date()} to {campaign_end.date()})")

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
						logger.info(f"Found campaign name: {campaign_name}")

						# Try to get the abbreviation from the same row if available
						if 'abbreviation' in df.columns:
							# Get the abbreviation from the same row as the name
							row_idx = names.index[0]
							if row_idx in df.index and not pd.isna(df.loc[row_idx, 'abbreviation']):
								campaign_abbr = df.loc[row_idx, 'abbreviation']
								logger.info(f"Found campaign abbreviation: {campaign_abbr}")
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
			'abbreviation': campaign_abbr
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
			logger.warning("Found negative dropout days, setting them to 0")
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
				sns.histplot(user_dropout['dropout_days'], kde=True, bins=20, palette=HISTOGRAM_COLORMAP)
				plt.title('Distribution of User Drop-out Rates')
				plt.xlabel('Time Between First and Last Activity (days)')
				plt.ylabel('Number of Users')

			create_and_save_figure(
				plot_dropout_rates_histogram,
				f'{OUTPUT_VISUALIZATIONS_DIR}/dropout_rates_distribution.png',
				figsize=(10, 6)
			)
			logger.info("Created dropout rates histogram visualization")
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
				sns.histplot(user_dropout['joining_days'], kde=True, bins=20, palette=HISTOGRAM_COLORMAP)
				plt.title('Distribution of User Joining Rates')
				plt.xlabel('Days Between Campaign Start and First Activity')
				plt.ylabel('Number of Users')

			create_and_save_figure(
				plot_joining_rates_histogram,
				f'{OUTPUT_VISUALIZATIONS_DIR}/joining_rates_distribution.png',
				figsize=(10, 6)
			)
			logger.info("Created joining rates histogram visualization")

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
			logger.info("Created combined dropout and joining rates visualization (KDE)")

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
				sns.boxplot(x='metric', y='days', data=combined_df, palette=['blue', 'red'])
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
			logger.info("Created combined dropout and joining rates boxplot visualization")
	except Exception as e:
		logger.error(f"Error calculating or visualizing joining rates: {e}")
		joining_metrics = None


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

					# Check if we have data for all hours
					all_hours = set(range(24))
					missing_hours = all_hours - set(activity_heatmap_data.columns)
					if missing_hours:
						logger.info(f"Missing hours in heatmap: {sorted(missing_hours)}")
						# Add missing hours with count 0
						for hour in missing_hours:
							activity_heatmap_data[hour] = 0
						# Sort columns
						activity_heatmap_data = activity_heatmap_data.reindex(sorted(activity_heatmap_data.columns), axis=1)

					if activity_heatmap_data.empty or activity_heatmap_data.sum().sum() == 0:
						logger.warning("Empty heatmap data, skipping visualization")
					else:
						def plot_activity_heatmap_by_time():
							# Use a fixed format string for the heatmap
							# We'll use '.0f' for all values since most will be integers
							sns.heatmap(activity_heatmap_data, cmap=SEQUENTIAL_HEATMAP_COLORMAP, annot=True,
										fmt='.1f', linewidths=0.5)
							plt.title('Activity Heatmap by Day of Week and Hour of Day')
							plt.xlabel('Hour of Day')
							plt.ylabel('Day of Week')

						create_and_save_figure(
							plot_activity_heatmap_by_time,
							f'{OUTPUT_VISUALIZATIONS_DIR}/activity_heatmap_by_time.png',
							figsize=(15, 8)
						)
						logger.info("Created activity heatmap by time visualization")
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
							logger.info(f"Limiting stacked bar chart to top {top_count} activity types out of {len(type_counts)}")
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
								activity_types_by_date_filtered.index = pd.to_datetime(activity_types_by_date_filtered.index, errors='coerce')
								activity_types_by_date_filtered = activity_types_by_date_filtered.sort_index()
							except Exception:
								pass

							# 2) Reindex to full daily date range and fill missing with zeros
							try:
								if len(activity_types_by_date_filtered.index) > 0:
									full_range = pd.date_range(start=activity_types_by_date_filtered.index.min(), end=activity_types_by_date_filtered.index.max(), freq='D')
									activity_types_by_date_filtered = activity_types_by_date_filtered.reindex(full_range).fillna(0)
							except Exception:
								pass

							# Keep actual dates on the x-axis (no conversion to campaign day numbers)
							# Prepare labels as date strings for clarity
							try:
								date_labels = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in activity_types_by_date_filtered.index]
							except Exception:
								date_labels = [str(d) for d in activity_types_by_date_filtered.index]

							# Set labels and title for dates
							title = 'Activity Types Distribution by Date'
							xlabel = 'Date'

							# Plot stacked bar chart
							plt.figure(figsize=PLOT_FIGSIZE_ACTIVITY_TYPES_STACKED)
							# Determine independent colormap (function or instance)
							cmap = INDEPENDENT_COLORMAP(activity_types_by_date_filtered.shape[1]) if callable(INDEPENDENT_COLORMAP) else INDEPENDENT_COLORMAP
							activity_types_by_date_filtered.plot(kind='bar', stacked=True, colormap=cmap, width=1, ax=plt.gca())
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
							plt.tight_layout()
							plt.savefig(f"{OUTPUT_VISUALIZATIONS_DIR}/activity_types_stacked_by_date.png", bbox_inches='tight')
							plt.close()
							logger.info("Created activity types stacked by date visualization")
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
						# Use all users for the heatmap as per requirements
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
									user_activity_heatmap_filtered.columns = pd.to_datetime(user_activity_heatmap_filtered.columns)
								except Exception:
									pass

							# Format column labels as 'YYYY-MM-DD' for readability
							try:
								user_activity_heatmap_filtered.columns = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in user_activity_heatmap_filtered.columns]
							except Exception:
								user_activity_heatmap_filtered.columns = [str(d) for d in user_activity_heatmap_filtered.columns]

							title = f'User Engagement Heatmap by Day (All {len(top_users)} Users)'
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
									plt.ylabel('User ID')
									# Add explanation of how engagement is calculated
									plt.figtext(0.5, 0.01,
												"Note: Engagement is calculated as the total number of activities per user. " +
												"Colors represent activity count.",
												ha='center', fontsize=9, wrap=True)

								create_and_save_figure(
									plot_user_engagement_heatmap,
									f'{OUTPUT_VISUALIZATIONS_DIR}/user_engagement_heatmap.png',
									figsize=(16, 10)
								)
								logger.info("Created user engagement heatmap visualization")
			except Exception as e:
				logger.error(f"Error processing data for user engagement heatmap: {e}")
	except Exception as e:
		logger.error(f"Error creating user engagement heatmap: {e}")
		# Continue with other visualizations even if this one fails

	# Prepare return values
	logger.info("Preparing return values from analyze_activities")

	# Usage metrics have been removed as per requirements


	# Ensure dropout_metrics is properly defined if it wasn't created earlier
	if 'dropout_metrics' not in locals() or dropout_metrics is None:
		logger.warning("Dropout metrics not calculated, creating empty dictionary")
		dropout_metrics = {
			'avg_dropout_days': 0,
			'median_dropout_days': 0,
			'min_dropout_days': 0,
			'max_dropout_days': 0
		}

	# Ensure campaign_metrics is properly defined if it wasn't created earlier
	if 'campaign_metrics' not in locals() or campaign_metrics is None:
		logger.warning("Campaign metrics not calculated, creating empty dictionary")
		campaign_metrics = {
			'unique_users': unique_users_count if 'unique_users_count' in locals() else 0,
			'length_days': campaign_length_days if 'campaign_length_days' in locals() else 0,
			'start_date': campaign_start.date() if 'campaign_start' in locals() and campaign_start is not None else None,
			'end_date': campaign_end.date() if 'campaign_end' in locals() and campaign_end is not None else None,
			'name': "GameBus Campaign",  # Default campaign name
			'abbreviation': ""  # Default empty abbreviation
		}

	# Log summary of metrics
	logger.info(f"Analysis complete: {unique_users_count} users, " +
				f"{campaign_metrics['length_days']} days")

	# Save statistics to files
	try:
		# Ensure statistics directory exists
		ensure_dir(OUTPUT_STATISTICS_DIR)
		logger.info(f"Saving statistics to {OUTPUT_STATISTICS_DIR}")

		# Save campaign summary
		try:
			campaign_summary_path = os.path.join(OUTPUT_STATISTICS_DIR, 'campaign_summary.txt')
			with open(campaign_summary_path, 'w') as f:
				f.write(f"Campaign Summary\n")
				f.write(f"========================\n\n")

				# Use safe access to get campaign name and abbreviation
				campaign_name = campaign_metrics.get('name', "GameBus Campaign")
				campaign_abbr = campaign_metrics.get('abbreviation', "")

				# Write campaign name and abbreviation if available
				if campaign_abbr:
					f.write(f"{campaign_name} ({campaign_abbr})\n\n")
				else:
					f.write(f"{campaign_name}\n\n")

				# Use safe access to campaign_metrics
				start_date = campaign_metrics.get('start_date', 'Unknown')
				end_date = campaign_metrics.get('end_date', 'Unknown')
				length_days = campaign_metrics.get('length_days', 0)
				unique_users = campaign_metrics.get('unique_users', 0)

				f.write(f"Start Date: {start_date}\n")
				f.write(f"End Date: {end_date}\n")
				f.write(f"Length: {length_days} days\n")
				f.write(f"Number of Participants: {unique_users}\n")

			logger.info(f"Saved campaign summary to {campaign_summary_path}")
		except Exception as e:
			logger.error(f"Error saving campaign summary: {e}")

	except Exception as e:
		logger.error(f"Error creating statistics directory or saving files: {e}")

	# Create a separate file for drop-out rates analysis
	try:
		dropout_analysis_path = os.path.join(OUTPUT_STATISTICS_DIR, 'dropout_analysis.txt')
		with open(dropout_analysis_path, 'w') as f:
			f.write(f"Drop-out Rates Analysis\n")
			f.write(f"=====================\n\n")
			f.write(f"Definition:\n")
			f.write(f"- Drop-out Rate: Time between first and last recorded activity for each user\n\n")

			# Check if dropout_stats is available
			if dropout_metrics is None or 'dropout_stats' not in locals() or dropout_stats is None:
				logger.warning("No dropout statistics available for analysis file")
				f.write(f"No dropout statistics available for this dataset.\n")
			else:
				try:
					# Create a table with dropout statistics using safe access
					table_data = [
						["Metric", "Value (days)"],
						["Average drop-out time", f"{dropout_stats.get('mean', 0):.1f}"],
						["Median drop-out time", f"{dropout_stats.get('50%', 0):.1f}"],
						["Minimum drop-out time", f"{dropout_stats.get('min', 0):.1f}"],
						["Maximum drop-out time", f"{dropout_stats.get('max', 0):.1f}"]
					]

					# Add percentiles if available
					if '25%' in dropout_stats:
						table_data.append(["25th percentile", f"{dropout_stats['25%']:.1f}"])
					if '75%' in dropout_stats:
						table_data.append(["75th percentile", f"{dropout_stats['75%']:.1f}"])
					if '90%' in dropout_stats:
						table_data.append(["90th percentile", f"{dropout_stats['90%']:.1f}"])

					# Format the table using tabulate
					table = tabulate.tabulate(table_data, headers="firstrow", tablefmt="grid")
					f.write(table)

					# Add additional statistical analysis
					f.write(f"\n\nStatistical Analysis:\n")

					# Add standard deviation if available
					if 'std' in dropout_stats:
						f.write(f"- Standard deviation: {dropout_stats['std']:.1f} days\n")

					# Add count if available
					if 'count' in dropout_stats:
						f.write(f"- Total number of users analyzed: {dropout_stats['count']:.0f}\n")

					# Calculate IQR if both percentiles are available
					if '75%' in dropout_stats and '25%' in dropout_stats:
						iqr = dropout_stats['75%'] - dropout_stats['25%']
						f.write(f"- Interquartile range (IQR): {iqr:.1f} days\n")

					# Calculate range if both min and max are available
					if 'max' in dropout_stats and 'min' in dropout_stats:
						range_val = dropout_stats['max'] - dropout_stats['min']
						f.write(f"- Range: {range_val:.1f} days\n\n")

					f.write(f"Distribution Analysis:\n")
					f.write(f"- The distribution of drop-out rates shows how long users remain engaged with the app.\n")
					f.write(f"- A higher average drop-out time indicates better user retention.\n")
					f.write(f"- The difference between mean and median can indicate skewness in the distribution.\n")
					f.write(f"- Large standard deviation suggests high variability in user engagement patterns.\n")
				except Exception as e:
					logger.error(f"Error generating dropout statistics table: {e}")
					f.write(f"Error generating dropout statistics: {e}\n")

		logger.info(f"Saved dropout analysis to {dropout_analysis_path}")
	except Exception as e:
		logger.error(f"Error saving dropout analysis: {e}")

	# Analyze completed tasks
	try:
		task_completion_path = os.path.join(OUTPUT_STATISTICS_DIR, 'task_completion_analysis.txt')
		with open(task_completion_path, 'w') as f:
			f.write(f"Task Completion Analysis\n")
			f.write(f"======================\n\n")
			f.write(f"Definition:\n")
			f.write(f"- Completed Task: An activity that has been recorded and potentially rewarded with points\n\n")

			# Check if activities data is available
			if activities is None or activities.empty:
				logger.warning("No activities data available for task completion analysis")
				f.write(f"No activities data available for this dataset.\n")
			else:
				try:
					# 1. Frequency of completed tasks
					f.write(f"1. Task Frequency Analysis\n")
					f.write(f"------------------------\n\n")

					# Calculate tasks per day
					try:
						tasks_per_day = activities.groupby('date').size()
						avg_tasks_per_day = tasks_per_day.mean()
						median_tasks_per_day = tasks_per_day.median()
						max_tasks_per_day = tasks_per_day.max()
						min_tasks_per_day = tasks_per_day.min()

						# Calculate tasks per user
						tasks_per_user = activities.groupby('pid').size()
						avg_tasks_per_user = tasks_per_user.mean()
						median_tasks_per_user = tasks_per_user.median()
						max_tasks_per_user = tasks_per_user.max()
						min_tasks_per_user = tasks_per_user.min()

						# Create a table with frequency statistics
						table_data = [
							["Metric", "Per Day", "Per User"],
							["Average tasks", f"{avg_tasks_per_day:.1f}", f"{avg_tasks_per_user:.1f}"],
							["Median tasks", f"{median_tasks_per_day:.1f}", f"{median_tasks_per_user:.1f}"],
							["Maximum tasks", f"{max_tasks_per_day}", f"{max_tasks_per_user}"],
							["Minimum tasks", f"{min_tasks_per_day}", f"{min_tasks_per_user}"]
						]

						# Format the table using tabulate
						table = tabulate.tabulate(table_data, headers="firstrow", tablefmt="grid")
						f.write(table)
						f.write("\n\n")

						logger.info("Generated task frequency analysis")
					except Exception as e:
						logger.error(f"Error calculating task frequency statistics: {e}")
						f.write(f"Error calculating task frequency statistics: {e}\n\n")
				except Exception as e:
					logger.error(f"Error generating task completion analysis: {e}")
					f.write(f"Error generating task completion analysis: {e}\n")

					# 2. Timing of completed tasks
					try:
						f.write(f"2. Task Timing Analysis\n")
						f.write(f"---------------------\n\n")

						# Analyze tasks by hour of day
						tasks_by_hour = activities.groupby('hour').size()
						if tasks_by_hour.empty:
							logger.warning("No hourly task data available")
							f.write("No hourly task data available.\n\n")
						else:
							peak_hour = tasks_by_hour.idxmax()

							# Analyze tasks by day of week
							if 'day_of_week' not in activities.columns:
								activities['day_of_week'] = pd.to_datetime(activities['date']).dt.day_name()

							day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
							tasks_by_day = activities.groupby('day_of_week').size().reindex(day_order)

							if tasks_by_day.empty or tasks_by_day.isnull().all():
								logger.warning("No daily task data available")
								f.write("No daily task data available.\n\n")
							else:
								# Fill NaN values with 0
								tasks_by_day = tasks_by_day.fillna(0)
								peak_day = tasks_by_day.idxmax()

								# Calculate percentages safely
								hour_sum = tasks_by_hour.sum()
								hour_pct = (tasks_by_hour[peak_hour] / hour_sum * 100) if hour_sum > 0 else 0

								day_sum = tasks_by_day.sum()
								day_pct = (tasks_by_day[peak_day] / day_sum * 100) if day_sum > 0 else 0

								table_data = [
									["Time Period", "Task Count", "Percentage"],
									["Peak hour", f"{peak_hour}:00 - {peak_hour+1}:00", f"{hour_pct:.1f}%"],
									["Peak day", f"{peak_day}", f"{day_pct:.1f}%"]
								]

								# Add rows for each hour with significant activity (>5% of total)
								for hour, count in tasks_by_hour.items():
									if hour == peak_hour:
										continue  # Skip peak hour as it's already in the table

									percentage = (count / hour_sum * 100) if hour_sum > 0 else 0
									if percentage >= 5.0:
										table_data.append(
											["Active hour", f"{hour}:00 - {hour+1}:00", f"{percentage:.1f}%"]
										)

								# Add rows for each day with significant activity (>10% of total)
								for day, count in tasks_by_day.items():
									if day == peak_day:
										continue  # Skip peak day as it's already in the table

									percentage = (count / day_sum * 100) if day_sum > 0 else 0
									if percentage >= 10.0:
										table_data.append(
											["Active day", f"{day}", f"{percentage:.1f}%"]
										)

								# Format the table using tabulate
								table = tabulate.tabulate(table_data, headers="firstrow", tablefmt="grid")
								f.write(table)
								f.write("\n\n")

								logger.info("Generated task timing analysis")
					except Exception as e:
						logger.error(f"Error calculating task timing statistics: {e}")
						f.write(f"Error calculating task timing statistics: {e}\n\n")

					# 3. Types of completed tasks
					try:
						f.write(f"3. Task Types Analysis\n")
						f.write(f"-------------------\n\n")

						# Analyze task types
						task_types = activities['type'].value_counts()

						if task_types.empty:
							logger.warning("No task types data available")
							f.write("No task types data available.\n\n")
						else:
							total_tasks = task_types.sum()

							table_data = [
								["Task Type", "Count", "Percentage"]
							]

							# Add rows for each task type
							for task_type, count in task_types.items():
								percentage = (count / total_tasks * 100) if total_tasks > 0 else 0
								table_data.append([f"{task_type}", f"{count}", f"{percentage:.1f}%"])

							# Format the table using tabulate
							table = tabulate.tabulate(table_data, headers="firstrow", tablefmt="grid")
							f.write(table)
							f.write("\n\n")

							logger.info("Generated task types analysis")
					except Exception as e:
						logger.error(f"Error calculating task types statistics: {e}")
						f.write(f"Error calculating task types statistics: {e}\n\n")

					# 4. Completion rates by activity type/task
					try:
						f.write(f"4. Completion Rates by Activity Type\n")
						f.write(f"--------------------------------\n\n")
						f.write(f"Note: Completion rate is calculated as the average number of tasks per user per day\n\n")

						# Get campaign metrics safely
						users_count = campaign_metrics.get('unique_users', 0) if campaign_metrics else 0
						days_count = campaign_metrics.get('length_days', 0) if campaign_metrics else 0

						if users_count <= 0 or days_count <= 0:
							logger.warning("Invalid campaign metrics for completion rate calculation")
							f.write("Cannot calculate completion rates: missing campaign metrics.\n\n")
						else:
							try:
								# Calculate completion rate as the number of tasks per user per day
								completion_rate_by_type = activities.groupby('type').apply(
									lambda x: len(x) / (users_count * days_count),
									include_groups=False
								)

								if completion_rate_by_type.empty:
									logger.warning("No completion rate data available")
									f.write("No completion rate data available.\n\n")
								else:
									table_data = [
										["Activity Type", "Completion Rate (tasks/user/day)"]
									]

									# Add rows for each activity type
									for activity_type, rate in completion_rate_by_type.items():
										table_data.append([f"{activity_type}", f"{rate:.3f}"])

									# Format the table using tabulate
									table = tabulate.tabulate(table_data, headers="firstrow", tablefmt="grid")
									f.write(table)
									f.write("\n\n")

									logger.info("Generated completion rates analysis")
							except Exception as e:
								logger.error(f"Error calculating completion rates: {e}")
								f.write(f"Error calculating completion rates: {e}\n\n")
					except Exception as e:
						logger.error(f"Error generating completion rates analysis: {e}")
						f.write(f"Error generating completion rates analysis: {e}\n\n")

					# 5. Activity provider/source analysis
					try:
						f.write(f"5. Activity Provider/Source Analysis\n")
						f.write(f"---------------------------------\n\n")
						f.write(f"Note: Provider information is inferred from activity types\n\n")

						if task_types is None or task_types.empty:
							logger.warning("No task types data available for provider analysis")
							f.write("No task types data available for provider analysis.\n\n")
						else:
							try:
								# Group activities by inferred provider
								# This is a simplification - in a real scenario, you would extract actual provider information
								inferred_providers = {
									'Self-reported': ['NUTRITION_DIARY', 'DRINKING_DIARY', 'PLAN_MEAL'],
									'Physical activities': ['PHYSICAL_ACTIVITY'],
									'App interaction': ['VISIT_APP_PAGE', 'NAVIGATE_APP'],
									'General activities': ['GENERAL_ACTIVITY'],
									'Other': []  # Will catch any types not in the above categories
								}

								provider_counts = {provider: 0 for provider in inferred_providers}

								# Count activities by inferred provider
								for activity_type, count in task_types.items():
									assigned = False
									for provider, types in inferred_providers.items():
										if activity_type in types:
											provider_counts[provider] += count
											assigned = True
											break
									if not assigned:
										provider_counts['Other'] += count

								# Calculate total tasks
								total_tasks = sum(provider_counts.values())

								if total_tasks <= 0:
									logger.warning("No tasks found for provider analysis")
									f.write("No tasks found for provider analysis.\n\n")
								else:
									table_data = [
										["Provider/Source", "Task Count", "Percentage"]
									]

									# Add rows for each provider
									for provider, count in provider_counts.items():
										if count > 0:  # Only include providers with activities
											percentage = (count / total_tasks * 100) if total_tasks > 0 else 0
											table_data.append([f"{provider}", f"{count}", f"{percentage:.1f}%"])

									# Format the table using tabulate
									table = tabulate.tabulate(table_data, headers="firstrow", tablefmt="grid")
									f.write(table)
									f.write("\n\n")

									logger.info("Generated activity provider analysis")
							except Exception as e:
								logger.error(f"Error calculating provider statistics: {e}")
								f.write(f"Error calculating provider statistics: {e}\n\n")
					except Exception as e:
						logger.error(f"Error generating provider analysis: {e}")
						f.write(f"Error generating provider analysis: {e}\n\n")

					# 6. Rewarded points analysis
					try:
						f.write(f"6. Rewarded Points Analysis\n")
						f.write(f"------------------------\n\n")

						if 'points' not in activities.columns:
							logger.warning("No points column found in activities data")
							f.write("No points data available for analysis.\n\n")
						else:
							try:
								# Analyze points awarded for completed tasks
								total_points = activities['points'].sum()

								if total_points <= 0:
									logger.warning("No points awarded in the dataset")
									f.write("No points awarded in the dataset.\n\n")
								else:
									avg_points_per_task = activities['points'].mean()

									# Calculate points by activity type
									points_by_type = activities.groupby('type')['points'].agg(['sum', 'mean', 'count'])

									if points_by_type.empty:
										logger.warning("No points by activity type data available")
										f.write("No points by activity type data available.\n\n")
									else:
										points_by_type = points_by_type.sort_values('sum', ascending=False)

										table_data = [
											["Metric", "Value"],
											["Total points awarded", f"{total_points:.0f}"],
											["Average points per task", f"{avg_points_per_task:.1f}"]
										]

										# Add most/least rewarding activity types if available
										if len(points_by_type) > 0:
											table_data.append(["Most rewarding activity type",
															   f"{points_by_type.index[0]} ({points_by_type['mean'].iloc[0]:.1f} points/task)"])

										if len(points_by_type) > 1:
											table_data.append(["Least rewarding activity type",
															   f"{points_by_type.index[-1]} ({points_by_type['mean'].iloc[-1]:.1f} points/task)"])

										# Format the table using tabulate
										table = tabulate.tabulate(table_data, headers="firstrow", tablefmt="grid")
										f.write(table)
										f.write("\n\n")

										# Add detailed points by activity type
										f.write(f"Points by Activity Type:\n\n")

										table_data = [
											["Activity Type", "Total Points", "Average Points/Task", "Task Count"]
										]

										# Add rows for each activity type
										for activity_type in points_by_type.index:
											total = points_by_type.loc[activity_type, 'sum']
											average = points_by_type.loc[activity_type, 'mean']
											count = points_by_type.loc[activity_type, 'count']
											table_data.append([f"{activity_type}", f"{total:.0f}", f"{average:.1f}", f"{count}"])

										# Format the table using tabulate
										table = tabulate.tabulate(table_data, headers="firstrow", tablefmt="grid")
										f.write(table)
										f.write("\n\n")

										logger.info("Generated rewarded points analysis")
							except Exception as e:
								logger.error(f"Error calculating points statistics: {e}")
								f.write(f"Error calculating points statistics: {e}\n\n")
					except Exception as e:
						logger.error(f"Error generating rewarded points analysis: {e}")
						f.write(f"Error generating rewarded points analysis: {e}\n\n")

					# Add summary and insights
					try:
						f.write(f"Summary and Insights:\n")
						f.write(f"-------------------\n")

						# Add insights based on available data
						insights = []

						# Tasks per user insight
						if 'tasks_per_user' in locals() and not tasks_per_user.empty:
							avg_tasks_per_user_val = tasks_per_user.mean()
							insights.append(f"- Users completed an average of {avg_tasks_per_user_val:.1f} tasks during the campaign.")

						# Peak hour and day insight
						if 'peak_hour' in locals() and 'peak_day' in locals():
							insights.append(f"- The most active time for task completion was {peak_hour}:00 - {peak_hour+1}:00 on {peak_day}s.")

						# Most common task type insight
						if 'task_types' in locals() and not task_types.empty and 'total_tasks' in locals() and total_tasks > 0:
							most_common_type = task_types.index[0]
							most_common_pct = (task_types.iloc[0] / total_tasks * 100)
							insights.append(f"- The most common task type was {most_common_type} ({most_common_pct:.1f}% of all tasks).")

						# Most rewarding activity type insight
						if 'points_by_type' in locals() and not points_by_type.empty and len(points_by_type) > 0:
							most_rewarding_type = points_by_type.index[0]
							most_rewarding_points = points_by_type['mean'].iloc[0]
							insights.append(f"- The most rewarding activity type was {most_rewarding_type}, with an average of {most_rewarding_points:.1f} points per task.")

						# Highest completion rate insight
						if 'completion_rate_by_type' in locals() and not completion_rate_by_type.empty:
							highest_rate_type = completion_rate_by_type.idxmax()
							highest_rate = completion_rate_by_type.max()
							insights.append(f"- The highest completion rate was for {highest_rate_type} activities at {highest_rate:.3f} tasks per user per day.")

						# Write insights or a message if none are available
						if insights:
							for insight in insights:
								f.write(f"{insight}\n")
						else:
							f.write("- Not enough data available to generate insights.\n")

						logger.info("Added summary and insights to task completion analysis")
					except Exception as e:
						logger.error(f"Error generating summary and insights: {e}")
						f.write(f"Error generating summary and insights: {e}\n")

		logger.info(f"Saved task completion analysis to {task_completion_path}")
	except Exception as e:
		logger.error(f"Error saving task completion analysis: {e}")

	logger.info(f"Campaign metrics and task completion analysis saved to {OUTPUT_STATISTICS_DIR} folder")

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
		logger.info(f"\nUser engagement distribution:\n{engagement_stats}")

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
			logger.info("Created user engagement distribution visualization")
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
					sns.boxplot(data=df, x='engagement_type', y=metric, palette=BOX_COLORMAP)
					plt.title(f'{metric.capitalize()} Distribution by Engagement Type')
					plt.xlabel('Engagement Type')
					plt.ylabel(metric.capitalize())

				create_and_save_figure(
					plot_metric_by_engagement,
					os.path.join(OUTPUT_VISUALIZATIONS_DIR, f'{metric}_by_engagement.png'),
					figsize=(10, 6)
				)
				logger.info(f"Created {metric} by engagement visualization")
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
						logger.warning(f"Both active and passive users have all zero values for {metric}, skipping statistical comparison")
						continue

					stat, pval = stats.mannwhitneyu(
						active_data.dropna(),
						passive_data.dropna(),
						alternative='two-sided'
					)

					logger.info(f"\n{metric.capitalize()} comparison:")
					logger.info(f"Active users mean: {active_data.mean():.2f}")
					logger.info(f"Passive users mean: {passive_data.mean():.2f}")
					logger.info(f"Mann-Whitney U p-value: {pval:.4f}")
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
	logger.info("Analyzing visualizations, challenges, and tasks data...")

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

	logger.info(f"Loaded data: {len(visualizations_df)} visualizations, {len(challenges_df)} challenges, {len(tasks_df)} tasks")

	# Generate descriptive statistics
	generate_descriptive_stats(visualizations_df, "Visualizations Data", None)
	generate_descriptive_stats(challenges_df, "Challenges Data", None)
	generate_descriptive_stats(tasks_df, "Tasks Data", None)	

	# Prepare summary file path for later appends (ensure it exists)
	ensure_dir(OUTPUT_STATISTICS_DIR)
	summary_file_path = os.path.join(OUTPUT_STATISTICS_DIR, 'visualization_challenge_task_summary.txt')
	if not os.path.exists(summary_file_path):
		with open(summary_file_path, 'w') as f:
			f.write("# Visualization, Challenge, and Task Analysis\n\n")

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
				plt.bar(range(len(activity_counts)), activity_counts.values, color='mediumpurple')

				# Add labels
				plt.xlabel('Activity Type')
				plt.ylabel('Count')
				plt.title('Activity Completion by Type')
				plt.xticks(range(len(activity_counts)), activity_counts.index, rotation=45, ha='right')
				plt.tight_layout()

			create_and_save_figure(
				plot_activity_completion,
				f'{OUTPUT_VISUALIZATIONS_DIR}/activity_completion.png',
				figsize=(14, 8)
			)
			logger.info("Created activity completion visualization")

			# Add activity completion information to the summary file
			with open(summary_file_path, 'a') as f:
				f.write("\n## Activity Completion Analysis\n\n")
				f.write(f"Total activities recorded: {len(activities_df)}\n")
				f.write("\nActivity counts by type:\n")
				for activity_type, count in activity_counts.items():
					f.write(f"- {activity_type}: {count}\n")

			# 5. Frequency, timing, and type of tasks completed analysis
			logger.info("Analyzing frequency, timing, and type of tasks completed...")

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
				activity_hour_counts = pd.crosstab(activities_df['hour'], activities_df['type'])

				def plot_activity_type_by_hour():
					plt.figure(figsize=(14, 8))
					sns.heatmap(activity_hour_counts, cmap=SEQUENTIAL_HEATMAP_COLORMAP, annot=False, linewidths=.5)
					plt.title('Activity Types by Hour of Day')
					plt.xlabel('Activity Type')
					plt.ylabel('Hour of Day')
					plt.tight_layout()

				create_and_save_figure(
					plot_activity_type_by_hour,
					f'{OUTPUT_VISUALIZATIONS_DIR}/activity_type_by_hour.png',
					figsize=(14, 8)
				)
				logger.info("Created activity types by hour heatmap")

				# Create a heatmap of activity types by day of week
				activity_day_counts = pd.crosstab(activities_df['day_of_week'], activities_df['type'])
				activity_day_counts = activity_day_counts.reindex(day_order)

				def plot_activity_type_by_day():
					plt.figure(figsize=(14, 8))
					sns.heatmap(activity_day_counts, cmap=SEQUENTIAL_HEATMAP_COLORMAP, annot=False, linewidths=.5)
					plt.title('Activity Types by Day of Week')
					plt.xlabel('Activity Type')
					plt.ylabel('Day of Week')
					plt.tight_layout()

				create_and_save_figure(
					plot_activity_type_by_day,
					f'{OUTPUT_VISUALIZATIONS_DIR}/activity_type_by_day.png',
					figsize=(14, 8)
				)
				logger.info("Created activity types by day heatmap")
				
				# Count unique active users per day of the campaign
				active_users_per_day = activities_df.groupby('date')['pid'].nunique()
				
				def plot_active_users_per_day():
					plt.figure(figsize=(12, 6))
					# Plot by actual dates on the x-axis
					series = active_users_per_day.sort_index()
					date_labels = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in series.index]
					x_pos = range(len(series))
					plt.bar(x_pos, series.values, color='skyblue', width=0.6)
					plt.title('Number of Active Users per Day')
					plt.xlabel('Date')
					plt.ylabel('Number of Active Users')
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
					f'{OUTPUT_VISUALIZATIONS_DIR}/active_users_per_day.png',
					figsize=(12, 6)
				)
				logger.info("Created active users per day chart")

				# Add frequency, timing, and completion rate information to the summary file
				with open(summary_file_path, 'a') as f:
					f.write("\n## Frequency, Timing, and Completion Rate Analysis\n\n")

					# Frequency summary
					f.write("### Activity Frequency\n\n")
					f.write(f"Average activities per day: {activities_by_date.mean():.2f}\n")
					f.write(f"Maximum activities in a day: {activities_by_date.max()} (on {activities_by_date.idxmax()})\n")
					f.write(f"Minimum activities in a day: {activities_by_date.min()} (on {activities_by_date.idxmin()})\n\n")
					
					# Add information about active users per day
					f.write("### Active Users per Day\n\n")
					f.write(f"Average active users per day: {active_users_per_day.mean():.2f}\n")
					f.write(f"Maximum active users in a day: {active_users_per_day.max()} (on {active_users_per_day.idxmax()})\n")
					f.write(f"Minimum active users in a day: {active_users_per_day.min()} (on {active_users_per_day.idxmin()})\n")
					f.write(f"A chart showing the number of active users per day has been generated in the visualizations directory.\n\n")

					# Timing summary
					f.write("### Activity Timing\n\n")
					f.write("Most active hours of the day:\n")
					top_hours = activities_by_hour.nlargest(3)
					for hour, count in top_hours.items():
						f.write(f"- Hour {hour}: {count} activities\n")

					f.write("\nMost active days of the week:\n")
					top_days = activities_by_day.nlargest(3)
					for day, count in top_days.items():
						f.write(f"- {day}: {count} activities\n")

					# Completion rate by activity type
					f.write("\n### Completion Rates by Activity Type\n\n")

					# Calculate completion rates if we have task data to compare against
					if 'desc_tasks' in csv_data and not csv_data['desc_tasks'].empty:
						tasks_df = csv_data['desc_tasks']
						f.write("Activity completion rates (activities recorded vs. tasks defined):\n\n")

						# This is a simplified approach - in a real scenario, you'd need more specific matching logic
						# between activities and tasks
						for activity_type, count in activity_counts.items():
							f.write(f"- {activity_type}: {count} activities completed\n")
					else:
						f.write("Activity completion counts by type:\n\n")
						for activity_type, count in activity_counts.items():
							f.write(f"- {activity_type}: {count} activities\n")

					# Add activity type by time analysis
					f.write("\n### Activity Types by Time\n\n")
					f.write("Peak hours for each activity type:\n\n")

					for activity_type in activity_types:
						if activity_type in activity_hour_counts.columns:
							peak_hour = activity_hour_counts[activity_type].idxmax()
							f.write(f"- {activity_type}: Peak at hour {peak_hour}\n")

					f.write("\nPreferred days for each activity type:\n\n")
					for activity_type in activity_types:
						if activity_type in activity_day_counts.columns:
							peak_day = activity_day_counts[activity_type].idxmax()
							f.write(f"- {activity_type}: Most common on {peak_day}\n")
					
					# Ranking completed tasks by activity type, data source, and rewarded points
					# This section implements the feature to rank completed tasks by:
					# 1. Activity type (e.g., walking, fruit intake)
					# 2. Data source (self-report, Garmin, Nutrida)
					# 3. Rewarded points
					# It first checks if the necessary data is available, extracts or infers it if needed,
					# then groups and sorts the tasks according to these criteria.
					f.write("\n### Ranked Completed Tasks\n\n")
					f.write("Tasks ranked by: 1) activity type, 2) data source, 3) rewarded points\n\n")
					
					# Check if we have the necessary data for ranking
					has_points = 'points' in activities_df.columns or 'rewardedParticipations' in activities_df.columns
					has_data_source = 'data_provider_name' in activities_df.columns
					
					if not has_points:
						logger.warning("No points data available for ranking tasks")
						f.write("Points data not available for ranking tasks.\n\n")
					
					# Create a copy of activities_df for ranking
					ranking_df = activities_df.copy()
					
					# Extract points if not already available
					if 'points' not in ranking_df.columns and 'rewardedParticipations' in ranking_df.columns:
						logger.info("Extracting points from rewardedParticipations for ranking")
						ranking_df['points'] = ranking_df['rewardedParticipations'].apply(
							lambda x: extract_points(x) if isinstance(x, str) else 0
						)
						has_points = True
					
					# Infer data source if not already available
					if not has_data_source:
						logger.info("Inferring data source from activity types for ranking")
						# Define mapping of activity types to data sources
						inferred_providers = {
							'Self-report': ['NUTRITION_DIARY', 'DRINKING_DIARY', 'PLAN_MEAL'],
							'Garmin': ['PHYSICAL_ACTIVITY', 'STEPS', 'DISTANCE', 'HEART_RATE'],
							'Nutrida': ['NUTRITION_DIARY', 'FOOD_INTAKE', 'FRUIT_INTAKE'],
							'Other': []  # Will catch any types not in the above categories
						}
						
						# Function to infer provider from activity type
						def infer_provider(activity_type):
							for provider, types in inferred_providers.items():
								if activity_type in types:
									return provider
							return 'Other'
						
						ranking_df['data_provider_name'] = ranking_df['type'].apply(infer_provider)
						has_data_source = True
					
					if has_points and has_data_source:
						# Group by activity type and data source, then sort by points
						ranked_tasks = ranking_df.groupby(['type', 'data_provider_name'])['points'].agg(['sum', 'mean', 'count']).reset_index()
						ranked_tasks = ranked_tasks.sort_values(['type', 'data_provider_name', 'sum'], ascending=[True, True, False])
						
						if not ranked_tasks.empty:
							# Create a table for the ranked tasks
							table_data = [
								["Activity Type", "Data Source", "Total Points", "Avg Points", "Count"]
							]
							
							for _, row in ranked_tasks.iterrows():
								table_data.append([
									row['type'],
									row['data_provider_name'],
									f"{row['sum']:.0f}",
									f"{row['mean']:.1f}",
									f"{row['count']}"
								])
							
							# Format the table using tabulate
							table = tabulate.tabulate(table_data, headers="firstrow", tablefmt="grid")
							f.write(table)
							f.write("\n\n")
							
							logger.info("Generated ranked completed tasks analysis")
						else:
							logger.warning("No data available for ranked tasks analysis")
							f.write("No data available for ranked tasks analysis.\n\n")
					else:
						logger.warning("Insufficient data for ranking tasks by all criteria")
						f.write("Insufficient data for ranking tasks by all criteria.\n\n")
		else:
			logger.warning("No activity data available for completion analysis")
	except Exception as e:
		logger.error(f"Error creating activity completion analysis: {e}")
		import traceback
		logger.error(f"Error traceback: {traceback.format_exc()}")

	# New plots: completed tasks analysis (participants per task and completions per day)
	try:
		# Ensure we have necessary columns in activities_df
		required_cols = ['pid']
		missing = [c for c in required_cols if c not in activities_df.columns]
		if missing:
			logger.warning(f"Cannot compute completed tasks plots; missing columns: {missing}")
		else:
			# Use createdAt or date to determine completion date
			if 'createdAt' in activities_df.columns and not pd.api.types.is_datetime64_any_dtype(activities_df['createdAt']):
				with pd.option_context('mode.chained_assignment', None):
					activities_df['createdAt'] = pd.to_datetime(activities_df['createdAt'], errors='coerce')
			# Ensure date column exists for grouping
			if 'date' not in activities_df.columns:
				if 'createdAt' in activities_df.columns:
					activities_df['date'] = activities_df['createdAt'].dt.date
				else:
					logger.warning("Activities lack both 'date' and 'createdAt'; cannot compute per-day task completions")
			# Extract detailed rewards to find challenge IDs
			if 'detailed_rewards' not in activities_df.columns:
				if 'rewardedParticipations' in activities_df.columns:
					logger.info("Extracting detailed_rewards from rewardedParticipations for task completion analysis")
					activities_df['detailed_rewards'] = activities_df['rewardedParticipations'].apply(
						lambda x: extract_detailed_rewards(x) if isinstance(x, str) else []
					)
				else:
					activities_df['detailed_rewards'] = [[] for _ in range(len(activities_df))]

			# Build completion records for both challenges and tasks (rules)
			challenge_completion_records = []
			task_completion_records = []
			for _, row in activities_df.iterrows():
				reward_list = row.get('detailed_rewards', []) if isinstance(row.get('detailed_rewards', []), list) else []
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
						'date': row['date'] if 'date' in row and pd.notna(row['date']) else (row['createdAt'].date() if 'createdAt' in row and pd.notna(row['createdAt']) else None)
					})
					# Append task completion (use name as key; optionally map to desc_tasks later)
					if pd.notna(task_name) and str(task_name).strip() != '':
						task_completion_records.append({
							'pid': row['pid'],
							'task_name': str(task_name).strip(),
							'date': row['date'] if 'date' in row and pd.notna(row['date']) else (row['createdAt'].date() if 'createdAt' in row and pd.notna(row['createdAt']) else None)
						})

			# DataFrames for completions
			challenge_completions_df = pd.DataFrame(challenge_completion_records)
			task_completions_df = pd.DataFrame(task_completion_records)

			# Optional: map tasks to desc_tasks if possible to standardize naming/ids
			if not task_completions_df.empty and 'desc_tasks' in csv_data:
				try:
					# Attempt to find a column in desc_tasks that matches the task 'rule' text
					possible_name_cols = [c for c in csv_data['desc_tasks'].columns if c.lower() in ['rule', 'task', 'name', 'label']]
					if possible_name_cols:
						name_col = possible_name_cols[0]
						map_df = csv_data['desc_tasks'][[name_col]].copy()
						map_df['task_name_norm'] = map_df[name_col].astype(str).str.strip().str.lower()
						task_completions_df['task_name_norm'] = task_completions_df['task_name'].astype(str).str.strip().str.lower()
						task_completions_df = task_completions_df.merge(map_df[['task_name_norm']], on='task_name_norm', how='left')
						task_completions_df.drop(columns=['task_name_norm'], inplace=True)
				except Exception as _e:
					# Mapping is best-effort; continue without failing
					pass

			# If we have no task records, skip plots
			if task_completions_df.empty:
				logger.warning("No task completion records (rules) found in rewards; skipping completion plots")
			else:
				# Unique user x task (by task name) and per-day counts
				unique_task_completions = task_completions_df.dropna(subset=['task_name']).drop_duplicates(subset=['pid', 'task_name'])

				# Plot: Number of tasks completed per day (unique user-task per day)
				if 'date' in task_completions_df.columns and task_completions_df['date'].notna().any():
					per_day = task_completions_df.dropna(subset=['date', 'task_name']).drop_duplicates(subset=['pid', 'task_name', 'date']).groupby('date').size().sort_index()
					def plot_tasks_completed_per_day():
						plt.figure(figsize=(12, 6))
						series = per_day.sort_index()
						date_labels = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in series.index]
						x_pos = range(len(series))
						plt.bar(x_pos, series.values, color='seagreen', width=0.6)
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
					logger.info("Created tasks completed per day visualization")
				else:
					logger.warning("Date information not available for per-day task completion plot")

				# Plot: Number of tasks completed per user (unique tasks per user)
				per_user = unique_task_completions.groupby('pid').size().sort_values(ascending=False)
				if not per_user.empty:
					def plot_tasks_completed_per_user():
						plt.figure(figsize=(12, 6))
						plot_data = per_user
						# Limit to top 50 users to keep plot readable
						if len(plot_data) > 50:
							plot_data = plot_data.head(50)
							plt.title('Tasks Completed per User (Top 50 Users)')
						else:
							plt.title('Tasks Completed per User')
						plot_data.plot(kind='bar', colormap=BAR_COLORMAP)
						plt.xlabel('User ID')
						plt.ylabel('Tasks Completed (unique tasks)')
						plt.xticks(rotation=45)
					create_and_save_figure(
						plot_tasks_completed_per_user,
						os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'tasks_completed_per_user.png'),
						figsize=(12, 6)
					)
					logger.info("Created tasks completed per user visualization")
				else:
					logger.warning("No data available for per-user task completion plot")

				# Compute participants per task for summary
				participants_per_task = unique_task_completions.groupby('task_name')['pid'].nunique().sort_values(ascending=False)
				participants_plot_df = participants_per_task.reset_index().rename(columns={'task_name': 'label', 'pid': 'participants'})

				# Append summary to the text report
				with open(summary_file_path, 'a') as f:
					f.write("\n## Task Completion Summary\n\n")
					f.write(f"Total unique task completions (user x task): {len(unique_task_completions)}\n")
					f.write("\nTop tasks by participants completing:\n")
					for _, r in participants_plot_df.head(10).iterrows():
						f.write(f"- {r['label']}: {int(r['participants'])} participants\n")
					# Add top users by tasks completed
					if not per_user.empty:
						f.write("\nTop users by tasks completed:\n")
						for pid, count in per_user.head(10).items():
							f.write(f"- User {pid}: {int(count)} tasks\n")
	except Exception as e:
		logger.error(f"Error generating completed tasks plots: {e}")

logger.info("Completed analysis of visualizations, challenges, and tasks")


def analyze_task_points_levels_impact(csv_data: Dict[str, pd.DataFrame], activities_df: Optional[pd.DataFrame] = None, campaign_metrics: Optional[Dict] = None) -> None:
	"""
	Analyze the impact of task types, points, and levels on engagement and adherence.

	Definitions used (robust fallbacks are applied if exact columns are missing):
	- Engagement (user-level):
	  * engagement_count = number of activities a user performed.
	  * engagement_active = Active/Passive (Active if any of STEPS/KCALORIES/DURATION > 0 if available; falls back to engagement_count > 0).
	- Adherence (user-level):
	  * active_days_ratio = number of unique active days / campaign_length_days (if campaign dates available),
	    otherwise normalized by each user's observed span (min 1).
	  * dropout_days = (last_activity - first_activity) in days.
	- Task type: uses activities_df['type'] (activity/task kind).
	- Points: uses activities_df['points'] (parsed from rewardedParticipations).
	- Level: uses desc_tasks['level'] if available; otherwise derives bins from points per activity (Low/Medium/High by tertiles).

	Outputs:
	- statistics/tasks_points_levels_impact.txt (summary tables and correlation results)
	- visualizations:
	  * engagement_by_activity_type.png
	  * adherence_by_activity_type.png
	  * engagement_vs_avg_points.png
	  * adherence_vs_avg_points.png
	  * engagement_by_level.png (if levels available/derived)
	  * adherence_by_level.png (if levels available/derived)
	"""
	try:
		# Acquire activities data
		if activities_df is None:
			activities_df = csv_data.get('activities')
		if activities_df is None or activities_df.empty:
			logger.warning("No activities data available for tasks/points/levels impact analysis")
			return

		# Ensure required basic columns
		required_cols = ['pid', 'type', 'createdAt']
		missing = [c for c in required_cols if c not in activities_df.columns]
		if missing:
			logger.warning(f"Activities missing required columns for impact analysis: {missing}")
			# We can still proceed if we at least have pid and type
			if not set(['pid', 'type']).issubset(activities_df.columns):
				return

		# Work on a copy
		df = activities_df.copy()

		# Parse dates
		if 'createdAt' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['createdAt']):
			with pd.option_context('mode.chained_assignment', None):
				df['createdAt'] = pd.to_datetime(df['createdAt'], errors='coerce')
		# Derive date column for daily aggregation
		if 'date' not in df.columns and 'createdAt' in df.columns:
			with pd.option_context('mode.chained_assignment', None):
				df['date'] = df['createdAt'].dt.date

		# Ensure points column exists (parsed earlier in analyze_activities)
		if 'points' not in df.columns and 'rewardedParticipations' in df.columns:
			with pd.option_context('mode.chained_assignment', None):
				df['points'] = df['rewardedParticipations'].apply(extract_points)
		elif 'points' not in df.columns:
			with pd.option_context('mode.chained_assignment', None):
				df['points'] = 0

		# Compute user-level engagement metrics
		user_metrics = df.groupby('pid').agg(
			engagement_count=('pid', 'size'),
			first_activity=('createdAt', 'min'),
			last_activity=('createdAt', 'max'),
			active_days=('date', lambda x: x.nunique() if x is not None else 0),
			total_points=('points', 'sum'),
			avg_points=('points', 'mean')
		).reset_index()

		# Handle NaT values gracefully
		user_metrics['first_activity'] = pd.to_datetime(user_metrics['first_activity'], errors='coerce')
		user_metrics['last_activity'] = pd.to_datetime(user_metrics['last_activity'], errors='coerce')
		user_metrics['span_days'] = (user_metrics['last_activity'] - user_metrics['first_activity']).dt.days
		user_metrics['span_days'] = user_metrics['span_days'].fillna(0).clip(lower=0)

		# Determine campaign length for adherence normalization
		campaign_length_days = None
		if campaign_metrics and isinstance(campaign_metrics.get('length_days', None), (int, float)) and campaign_metrics.get('length_days', 0) > 0:
			campaign_length_days = int(campaign_metrics['length_days'])
		elif 'date' in df.columns and df['date'].notna().any():
			# Fallback to observed campaign span
			dates = pd.to_datetime(df['date'], errors='coerce')
			campaign_length_days = int((dates.max() - dates.min()).days) if dates.notna().any() else None

		# Adherence metrics
		if campaign_length_days and campaign_length_days > 0:
			user_metrics['active_days_ratio'] = (user_metrics['active_days'] / campaign_length_days).clip(upper=1)
		else:
			# Normalize by own span, avoid division by zero
			user_metrics['active_days_ratio'] = user_metrics.apply(lambda r: (r['active_days'] / max(1, r['span_days'])) if pd.notna(r['span_days']) and r['span_days'] > 0 else 1.0 if r['active_days'] > 0 else 0.0, axis=1)
		# Dropout days (as inverse of adherence)
		user_metrics['dropout_days'] = user_metrics['span_days']

		# Engagement active/passive flag if activity metrics exist
		metric_cols = [c for c in ['STEPS', 'KCALORIES', 'DURATION'] if c in df.columns]
		if metric_cols:
			agg = df.groupby('pid')[metric_cols].sum().reset_index()
			user_metrics = user_metrics.merge(agg, on='pid', how='left')
			user_metrics['engagement_active'] = (user_metrics[metric_cols].fillna(0).sum(axis=1) > 0).map({True: 'Active', False: 'Passive'})
		else:
			user_metrics['engagement_active'] = (user_metrics['engagement_count'] > 0).map({True: 'Active', False: 'Passive'})

		# Map user-level metrics back to activity types presence
		# Determine which users performed each activity type at least once
		user_types = df.groupby('pid')['type'].apply(lambda s: sorted(pd.Series(s).dropna().unique().tolist())).reset_index().rename(columns={'type': 'types_list'})
		user_metrics = user_metrics.merge(user_types, on='pid', how='left')

		# Helper to compute metric by membership in type
		def metric_by_type(metric_col: str) -> pd.DataFrame:
			rows = []
			for act_type, _ in df['type'].value_counts().items():
				mask = user_metrics['types_list'].apply(lambda lst: isinstance(lst, list) and act_type in lst)
				vals = user_metrics.loc[mask, metric_col]
				if len(vals) == 0:
					continue
				rows.append({'activity_type': act_type, 'count_users': int(mask.sum()), 'mean': float(pd.Series(vals).mean()), 'median': float(pd.Series(vals).median())})
			return pd.DataFrame(rows).sort_values('mean', ascending=False)

		engagement_by_type = metric_by_type('engagement_count')
		adherence_by_type = metric_by_type('active_days_ratio')

		# Derive or fetch levels
		levels_available = False
		level_map = None
		if 'desc_tasks' in csv_data and not csv_data['desc_tasks'].empty:
			tasks_df = csv_data['desc_tasks']
			possible_level_cols = [c for c in tasks_df.columns if c.lower() in ['level', 'difficulty', 'tier', 'stage', 'lvl']]
			name_cols = [c for c in tasks_df.columns if c.lower() in ['rule', 'task', 'name', 'label']]
			if possible_level_cols and name_cols:
				lvl_col = possible_level_cols[0]
				name_col = name_cols[0]
				level_map = tasks_df[[name_col, lvl_col]].dropna()
				level_map.columns = ['task_name', 'level']
				levels_available = True
		# If no explicit level, derive from points per activity (tertiles)
		if not levels_available:
			# Build per-activity points distribution
			pts = df[['points']].copy()
			if not pts.empty and pts['points'].notna().any():
				q = pts['points'].quantile([0.33, 0.66]).fillna(0)
				low, mid = q.iloc[0], q.iloc[1]
				with pd.option_context('mode.chained_assignment', None):
					df['derived_level'] = pd.cut(df['points'], bins=[-1e9, low, mid, 1e12], labels=['Low', 'Medium', 'High'])
				levels_available = True
				level_source = 'derived_level'
			else:
				level_source = None
		else:
			level_source = 'mapped_level'

		# If level_map exists, attempt to map from rewardedParticipations 'rule' field
		if levels_available and level_map is not None:
			# Ensure detailed_rewards exists
			if 'detailed_rewards' not in df.columns and 'rewardedParticipations' in df.columns:
				with pd.option_context('mode.chained_assignment', None):
					df['detailed_rewards'] = df['rewardedParticipations'].apply(extract_detailed_rewards)
			# Extract rule/task name when present
			def get_rule_name(reward_list):
				if not isinstance(reward_list, list):
					return None
				for r in reward_list:
					if isinstance(r, dict) and 'rule' in r and pd.notna(r['rule']):
						return str(r['rule']).strip()
				return None
			with pd.option_context('mode.chained_assignment', None):
				df['task_name'] = df.get('detailed_rewards', [[]]).apply(get_rule_name)
			mapped = df.merge(level_map, on='task_name', how='left')
			with pd.option_context('mode.chained_assignment', None):
				mapped['level'] = mapped['level'].fillna('Unknown')
			level_source = 'level'
		else:
			mapped = df.copy()
			if levels_available and level_source == 'derived_level':
				mapped['level'] = mapped['derived_level']

		# Build per-user level assignments (most frequent level)
		levels_summary = None
		if levels_available and 'level' in mapped.columns:
			lvl_user = mapped.dropna(subset=['level']).groupby(['pid', 'level']).size().reset_index(name='n')
			# pick most frequent level per user
			idx = lvl_user.groupby('pid')['n'].idxmax()
			user_level = lvl_user.loc[idx, ['pid', 'level']]
			user_metrics = user_metrics.merge(user_level, on='pid', how='left')
			levels_summary = user_metrics.groupby('level').agg(
				users=('pid', 'nunique'),
				mean_engagement=('engagement_count', 'mean'),
				median_engagement=('engagement_count', 'median'),
				mean_adherence=('active_days_ratio', 'mean'),
				median_adherence=('active_days_ratio', 'median')
			).reset_index()

		# Correlations with points
		corr_spear_eng = stats.spearmanr(user_metrics['avg_points'].fillna(0), user_metrics['engagement_count'].fillna(0)) if len(user_metrics) > 1 else (None, None)
		corr_spear_adh = stats.spearmanr(user_metrics['avg_points'].fillna(0), user_metrics['active_days_ratio'].fillna(0)) if len(user_metrics) > 1 else (None, None)

		# Save statistics report
		ensure_dir(OUTPUT_STATISTICS_DIR)
		report_path = os.path.join(OUTPUT_STATISTICS_DIR, 'tasks_points_levels_impact.txt')
		with open(report_path, 'w') as f:
			f.write("Analysis: Impact of Task Types, Points, and Levels on Engagement and Adherence\n\n")
			f.write("User-level metric definitions:\n")
			f.write("- engagement_count: total activities per user\n")
			f.write("- active_days_ratio: active (unique) days divided by campaign length or user span\n")
			f.write("\n-- By Activity Type --\n")
			if engagement_by_type is not None and not engagement_by_type.empty:
				f.write("Top activity types by mean engagement_count:\n")
				f.write(engagement_by_type.to_string(index=False))
				f.write("\n\n")
			if adherence_by_type is not None and not adherence_by_type.empty:
				f.write("Top activity types by mean active_days_ratio:\n")
				f.write(adherence_by_type.to_string(index=False))
				f.write("\n\n")
			f.write("-- Correlation with Points --\n")
			if corr_spear_eng[0] is not None:
				f.write(f"Spearman rho (avg_points vs engagement_count): {corr_spear_eng.correlation:.3f}, p={corr_spear_eng.pvalue:.4f}\n")
				f.write(f"Spearman rho (avg_points vs active_days_ratio): {corr_spear_adh.correlation:.3f}, p={corr_spear_adh.pvalue:.4f}\n")
			else:
				f.write("Insufficient data for correlation analysis.\n")
			if levels_summary is not None and not levels_summary.empty:
				f.write("\n-- By Level --\n")
				f.write(levels_summary.to_string(index=False))
		logger.info(f"Saved tasks/points/levels impact report to {report_path}")

		# Visualizations
		ensure_dir(OUTPUT_VISUALIZATIONS_DIR)
		# Engagement by activity type (top N to keep readable)
		try:
			if engagement_by_type is not None and not engagement_by_type.empty:
				plot_df = engagement_by_type.head(20)
				def plot_engagement_by_type():
					plt.bar(plot_df['activity_type'], plot_df['mean'], color='steelblue')
					plt.xticks(rotation=45, ha='right')
					plt.ylabel('Mean engagement_count')
					plt.title('Engagement by Activity Type (Top 20)')
				create_and_save_figure(
					plot_engagement_by_type,
					os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'engagement_by_activity_type.png'),
					figsize=(14, 8)
				)
		except Exception as e:
			logger.error(f"Error plotting engagement_by_activity_type: {e}")
		# Adherence by activity type
		try:
			if adherence_by_type is not None and not adherence_by_type.empty:
				plot_df = adherence_by_type.head(20)
				def plot_adherence_by_type():
					plt.bar(plot_df['activity_type'], plot_df['mean'], color='seagreen')
					plt.xticks(rotation=45, ha='right')
					plt.ylabel('Mean active_days_ratio')
					plt.title('Adherence by Activity Type (Top 20)')
				create_and_save_figure(
					plot_adherence_by_type,
					os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'adherence_by_activity_type.png'),
					figsize=(14, 8)
				)
		except Exception as e:
			logger.error(f"Error plotting adherence_by_activity_type: {e}")

		# Scatter: avg_points vs engagement/adherence
		try:
			if not user_metrics.empty:
				def plot_scatter(x, y, x_label, y_label, title):
					plt.scatter(user_metrics[x], user_metrics[y], alpha=0.6, c=user_metrics.get('engagement_count', 1), cmap=REGRESSION_COLORMAP)
					plt.xlabel(x_label)
					plt.ylabel(y_label)
					plt.title(title)
				create_and_save_figure(
					lambda: plot_scatter('avg_points', 'engagement_count', 'Avg points per activity', 'Engagement count', 'Engagement vs Avg Points'),
					os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'engagement_vs_avg_points.png'),
					figsize=(8, 6)
				)
				create_and_save_figure(
					lambda: plot_scatter('avg_points', 'active_days_ratio', 'Avg points per activity', 'Active days ratio', 'Adherence vs Avg Points'),
					os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'adherence_vs_avg_points.png'),
					figsize=(8, 6)
				)
		except Exception as e:
			logger.error(f"Error plotting points scatter: {e}")

		# By level (if available)
		try:
			if levels_summary is not None and not levels_summary.empty:
				def plot_by_level(metric_col, ylabel, title, fname):
					plt.bar(levels_summary['level'].astype(str), levels_summary[metric_col], color='mediumpurple')
					plt.xlabel('Level')
					plt.ylabel(ylabel)
					plt.title(title)
				create_and_save_figure(
					lambda: plot_by_level('mean_engagement', 'Mean engagement_count', 'Engagement by Level', os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'engagement_by_level.png')),
					os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'engagement_by_level.png'),
					figsize=(8, 6)
				)
				create_and_save_figure(
					lambda: plot_by_level('mean_adherence', 'Mean active_days_ratio', 'Adherence by Level', os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'adherence_by_level.png')),
					os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'adherence_by_level.png'),
					figsize=(8, 6)
				)
		except Exception as e:
			logger.error(f"Error plotting by level: {e}")

		logger.info("Completed impact analysis of task types, points, and levels")
	except Exception as e:
		logger.error(f"Error in analyze_task_points_levels_impact: {e}")
		import traceback
		logger.error(f"Traceback: {traceback.format_exc()}")


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
		logger.info(f"Processing key: {key}, data type: {type(data)}")

		if 'geofence' in key.lower() and isinstance(data, pd.DataFrame):
			logger.info(f"Found geofence data in key: {key}")
			# Extract user ID from the key if possible
			try:
				# Assuming key format is like "user_123_geofence"
				parts = key.split('_')
				logger.info(f"Split key into parts: {parts}")

				if len(parts) >= 3 and parts[0] == 'user':
					user_id = parts[1]
					logger.info(f"Extracted user_id: {user_id}")

					# Add user_id column to the DataFrame
					data_copy = data.copy()
					data_copy['user_id'] = user_id

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
					logger.info(f"Added geofence data for user {user_id} to geofence_dfs")
				else:
					logger.warning(f"Key {key} does not match expected format 'user_X_geofence'")
			except (IndexError, ValueError) as e:
				logger.warning(f"Error extracting user ID from key {key}: {e}")

	# Debug: Check if geofence_dfs is empty
	logger.info(f"geofence_dfs has {len(geofence_dfs)} dataframes")
	if not geofence_dfs:
		logger.warning("No geofence data found, geofence_dfs is empty")

		# Debug: Check each key in json_data to see why no geofence data was found
		for key, data in json_data.items():
			logger.info(f"Key: {key}, Type: {type(data)}")
			if isinstance(data, pd.DataFrame):
				logger.info(f"  DataFrame columns: {list(data.columns)}")
			elif isinstance(data, dict):
				logger.info(f"  Dictionary keys: {list(data.keys())}")
			else:
				logger.info(f"  Data type: {type(data)}")

		return

	# Combine all geofence dataframes
	all_geofence_data = pd.concat(geofence_dfs, ignore_index=True)

	# Convert TIMESTAMP to datetime if it's not already
	if 'TIMESTAMP' in all_geofence_data.columns:
		all_geofence_data['TIMESTAMP'] = pd.to_datetime(all_geofence_data['TIMESTAMP'], errors='coerce')

	# Clean data: Remove extreme outliers in SPEED
	# Calculate Q1, Q3 and IQR for SPEED
	Q1 = all_geofence_data['SPEED'].quantile(0.25)
	Q3 = all_geofence_data['SPEED'].quantile(0.75)
	IQR = Q3 - Q1

	# Define bounds for outliers (using 3*IQR as a more lenient threshold)
	lower_bound = Q1 - 3 * IQR
	upper_bound = Q3 + 3 * IQR

	# Filter out extreme outliers
	filtered_geofence_data = all_geofence_data[
		(all_geofence_data['SPEED'] >= lower_bound) & 
		(all_geofence_data['SPEED'] <= upper_bound)
	]

	# Log how many outliers were removed
	outliers_count = len(all_geofence_data) - len(filtered_geofence_data)
	if outliers_count > 0:
		logger.info(f"Removed {outliers_count} outliers from geofence data")

		# Generate descriptive statistics for geofence data
		stats_file = os.path.join(OUTPUT_STATISTICS_DIR, 'geofence_statistics.txt')
		with open(stats_file, 'w') as f:
			f.write("Geofence Data Statistics\n")
			f.write("=======================\n\n")

			# Overall statistics
			f.write("Overall Statistics:\n")
			f.write(f"Total records: {len(filtered_geofence_data)}\n")
			f.write(f"Number of users: {filtered_geofence_data['user_id'].nunique()}\n\n")

			# Statistics for each numeric column
			numeric_cols = ['LATITUDE', 'LONGITUDE', 'ALTITUDE', 'SPEED', 'ERROR']
			for col in numeric_cols:
				f.write(f"{col} Statistics:\n")
				f.write(f"  Min: {filtered_geofence_data[col].min():.2f}\n")
				f.write(f"  Max: {filtered_geofence_data[col].max():.2f}\n")
				f.write(f"  Mean: {filtered_geofence_data[col].mean():.2f}\n")
				f.write(f"  Median: {filtered_geofence_data[col].median():.2f}\n")
				f.write(f"  Std Dev: {filtered_geofence_data[col].std():.2f}\n\n")

		logger.info(f"Geofence statistics saved to {stats_file}")

		# Speed Distribution visualization removed as requested

		# Altitude Distribution visualization removed as requested

		# Error Margin Analysis visualization removed as requested

		# Speed vs. Error Correlation visualization removed as requested

		# Location Distribution removed as requested

		# 6. Temporal analysis (if timestamp data is available)
		if 'TIMESTAMP' in filtered_geofence_data.columns and not filtered_geofence_data['TIMESTAMP'].isna().all():
			# Add hour of day column
			filtered_geofence_data['hour_of_day'] = filtered_geofence_data['TIMESTAMP'].dt.hour

			def plot_hourly_activity():
				plt.figure(figsize=(12, 6))
				hourly_counts = filtered_geofence_data['hour_of_day'].value_counts().sort_index()
				sns.barplot(x=hourly_counts.index, y=hourly_counts.values)
				plt.title('Geofence Activity by Hour of Day')
				plt.xlabel('Hour of Day')
				plt.ylabel('Number of Records')
				plt.xticks(range(0, 24))
				plt.grid(True, alpha=0.3)

			create_and_save_figure(
				plot_hourly_activity,
				os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'geofence_hourly_activity.png')
			)

			# Speed by hour of day
			def plot_speed_by_hour():
				plt.figure(figsize=(12, 6))
				sns.boxplot(data=filtered_geofence_data, x='hour_of_day', y='SPEED')
				plt.title('Speed by Hour of Day')
				plt.xlabel('Hour of Day')
				plt.ylabel('Speed')
				plt.xticks(range(0, 24))
				plt.grid(True, alpha=0.3)

			create_and_save_figure(
				plot_speed_by_hour,
				os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'geofence_speed_by_hour.png')
			)

			# Day of week analysis visualization removed as requested

			# Monthly activity patterns visualization removed as requested

			# Speed vs. Altitude correlation visualization removed as requested

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

		# Location Heatmap visualization removed as requested

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

		# 13. Try to create an interactive map using folium if available
		try:
			import folium
			from folium.plugins import HeatMap

			# Create a map centered at the mean of the coordinates
			center_lat = filtered_geofence_data['LATITUDE'].mean()
			center_lon = filtered_geofence_data['LONGITUDE'].mean()

			m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

			# Add points to the map
			for _, row in filtered_geofence_data.iterrows():
				folium.CircleMarker(
					location=[row['LATITUDE'], row['LONGITUDE']],
					radius=5,
					color='blue',
					fill=True,
					fill_color='blue',
					fill_opacity=0.7,
					popup=f"Speed: {row['SPEED']}, Altitude: {row['ALTITUDE']}"
				).add_to(m)

			# Add a heatmap layer
			heat_data = [[row['LATITUDE'], row['LONGITUDE']] for _, row in filtered_geofence_data.iterrows()]
			HeatMap(heat_data).add_to(m)

			# Save the map
			map_file = os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'geofence_interactive_map.html')
			m.save(map_file)
			logger.info(f"Interactive map saved to {map_file}")
		except ImportError:
			logger.warning("Folium not available, skipping interactive map visualization")
		except Exception as e:
			logger.warning(f"Could not create interactive map: {e}")

		logger.info("Completed geofence data analysis")
	else:
		logger.warning("No geofence data available for analysis")

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
	logger.info("generate_analysis_report invoked")
	# Ensure data_analysis directory exists
	data_analysis_dir = ensure_dir(os.path.join(PROJECT_ROOT, 'data_analysis'))
	report_path = os.path.join(data_analysis_dir, 'analysis_report.txt')
	logger.info(f"Generating analysis report at {report_path}")


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

		f.write("\n2. Activities Analysis\n")
		f.write("---------------------\n")
		if activities is not None:
			f.write("✓ Activities analysis performed successfully\n")
			performed_analyses.append("Activities analysis")



			# Check if dropout analysis was performed
			if dropout_metrics:
				f.write("✓ User dropout analysis performed\n")
				performed_analyses.append("User dropout analysis")
			else:
				reason = "Insufficient activity data to calculate dropout rates"
				f.write("✗ User dropout analysis NOT performed\n")
				f.write(f"   - Reason: {reason}\n")
				skipped_analyses.append(("User dropout analysis", reason))

			# Check if joining analysis was performed
			if joining_metrics:
				f.write("✓ User joining rates analysis performed\n")
				performed_analyses.append("User joining rates analysis")
			else:
				reason = "Insufficient activity data or missing campaign start date to calculate joining rates"
				f.write("✗ User joining rates analysis NOT performed\n")
				f.write(f"   - Reason: {reason}\n")
				skipped_analyses.append(("User joining rates analysis", reason))

			# Check if campaign metrics analysis was performed
			if campaign_metrics:
				f.write("✓ Campaign metrics analysis performed\n")
				performed_analyses.append("Campaign metrics analysis")
			else:
				reason = "Insufficient data to determine campaign start and end dates"
				f.write("✗ Campaign metrics analysis NOT performed\n")
				f.write(f"   - Reason: {reason}\n")
				skipped_analyses.append(("Campaign metrics analysis", reason))
		else:
			reason = "No activity data available or error during analysis"
			f.write("✗ Activities analysis NOT performed\n")
			f.write(f"   - Reason: {reason}\n")
			skipped_analyses.append(("Activities analysis", reason))

		f.write("\n3. User Data Analysis\n")
		f.write("--------------------\n")
		if json_data:
			performed_analyses.append("User data analysis")

			# Check if user data contains specific types of data
			has_movement_data = False

			# First check if any filenames contain movement descriptors
			movement_descriptors = ['walk', 'run', 'cycle', 'bike']
			for key in json_data.keys():
				key_lower = key.lower()
				if any(descriptor in key_lower for descriptor in movement_descriptors):
					has_movement_data = True
					break

			# If no movement data found in filenames, check in the data structure
			if not has_movement_data:
				for user_data in json_data.values():
					if 'activities' in user_data:
						for activity in user_data['activities']:
							if activity.get('gameDescriptor') in ['WALK', 'RUN', 'CYCLE']:
								has_movement_data = True

			if has_movement_data:
				f.write("✓ Movement data analysis performed (steps, calories, etc.)\n")
				performed_analyses.append("Movement data analysis")
			else:
				reason = "No movement data (WALK, RUN, CYCLE) found in user activities"
				f.write("✗ Movement data analysis NOT performed\n")
				f.write(f"   - Reason: {reason}\n")
				skipped_analyses.append(("Movement data analysis", reason))

		else:
			reason = "No JSON user data available"
			f.write("✗ User data analysis NOT performed\n")
			f.write(f"   - Reason: {reason}\n")
			skipped_analyses.append(("User data analysis", reason))


		f.write("\n4. Summary\n")
		f.write("---------\n")
		f.write("Analysis completed. Results saved to:\n")
		f.write(f"- Visualizations: {OUTPUT_VISUALIZATIONS_DIR}\n")
		f.write(f"- Statistics: {OUTPUT_STATISTICS_DIR}\n")
		f.write(f"- This report: {report_path}\n")

		# Add list of performed analyses
		f.write("\n5. Analyses Performed\n")
		f.write("-------------------\n")

		# Check for statistics files
		statistics_files = []
		if os.path.exists(OUTPUT_STATISTICS_DIR):
			statistics_files = [f for f in os.listdir(OUTPUT_STATISTICS_DIR) if os.path.isfile(os.path.join(OUTPUT_STATISTICS_DIR, f))]

		# Check for visualization files
		visualization_files = []
		if os.path.exists(OUTPUT_VISUALIZATIONS_DIR):
			visualization_files = [f for f in os.listdir(OUTPUT_VISUALIZATIONS_DIR) if os.path.isfile(os.path.join(OUTPUT_VISUALIZATIONS_DIR, f))]

		# Write statistics files
		if statistics_files:
			f.write("Statistics:\n")
			for stat_file in statistics_files:
				f.write(f"- {stat_file}\n")

		# Write visualization files (group by type)
		if visualization_files:
			f.write("\nVisualizations:\n")

			# Group visualizations by type
			# First identify combined visualizations
			combined_viz = [v for v in visualization_files if 'combined' in v]

			# Then group the rest, excluding combined visualizations
			remaining_viz = [v for v in visualization_files if v not in combined_viz]
			activity_viz = [v for v in remaining_viz if v.startswith('activity_')]
			user_viz = [v for v in remaining_viz if v.startswith('user_')]
			movement_viz = [v for v in remaining_viz if 'steps' in v or 'calories' in v or 'movement' in v]
			dropout_viz = [v for v in remaining_viz if 'dropout' in v]
			joining_viz = [v for v in remaining_viz if 'joining' in v]
			other_viz = [v for v in remaining_viz if not any([
				v.startswith('activity_'),
				v.startswith('user_'),
				'steps' in v,
				'calories' in v,
				'movement' in v,
				'dropout' in v,
				'joining' in v
			])]

			if combined_viz:
				f.write("  Combined Metrics Analysis:\n")
				for viz in combined_viz:
					f.write(f"  - {viz}\n")

			if activity_viz:
				f.write("  Activity Analysis:\n")
				for viz in activity_viz:
					f.write(f"  - {viz}\n")

			# Group rewarded points visualizations
			points_viz = [v for v in remaining_viz if 'points' in v or 'rewards' in v]
			if points_viz:
				f.write("  Rewarded Points Analysis:\n")
				for viz in points_viz:
					f.write(f"  - {viz}\n")

			if user_viz:
				f.write("  User Analysis:\n")
				for viz in user_viz:
					f.write(f"  - {viz}\n")


			if movement_viz:
				f.write("  Movement Analysis:\n")
				for viz in movement_viz:
					f.write(f"  - {viz}\n")

			if dropout_viz:
				f.write("  Dropout Analysis:\n")
				for viz in dropout_viz:
					f.write(f"  - {viz}\n")

			if joining_viz:
				f.write("  Joining Rates Analysis:\n")
				for viz in joining_viz:
					f.write(f"  - {viz}\n")

			if other_viz:
				f.write("  Other Visualizations:\n")
				for viz in other_viz:
					f.write(f"  - {viz}\n")

		if not statistics_files and not visualization_files:
			f.write("No analyses were performed.\n")

		# Add list of skipped analyses with reasons
		f.write("\n7. Analyses Skipped\n")
		f.write("------------------\n")

		# List expected analyses that weren't found
		expected_statistics = [
			"campaign_summary.txt",
			"dropout_analysis.txt",
			"task_completion_analysis.txt",
			"tasks_points_levels_impact.txt"
		]

		missing_statistics = [s for s in expected_statistics if s not in statistics_files]

		if skipped_analyses or missing_statistics:
			# First list the skipped analyses from the original list
			for analysis, reason in skipped_analyses:
				f.write(f"- {analysis}: {reason}\n")

			# Then list any expected statistics files that weren't found
			for missing_stat in missing_statistics:
				f.write(f"- {missing_stat}: Required data not available or analysis failed\n")
		else:
			f.write("No analyses were skipped.\n")

	logger.info(f"Analysis report generated at {report_path}")
	return report_path

def create_complete_report(csv_data, json_data, activities=None):
	"""
    Create a complete analysis report with all required sections.

    Args:
        csv_data: Dictionary of DataFrames from Excel files
        json_data: Dictionary of data from JSON files
        activities: DataFrame of activities or tuple from analyze_activities

    Returns:
        str: Path to the generated report file.
    """
	# Unpack results if activities is a tuple from analyze_activities
	activities_df = activities
	dropout_metrics = None
	joining_metrics = None
	campaign_metrics = None
	if isinstance(activities, tuple) and len(activities) >= 5:
		activities_df, _, dropout_metrics, joining_metrics, campaign_metrics = activities

	# Generate and return the report path
	return generate_analysis_report(csv_data, json_data, activities_df, campaign_metrics, dropout_metrics, joining_metrics)

def main():
	"""Main function to run the analysis"""
	try:
		logger.info("Starting data analysis")

		# Ensure output directories exist
		ensure_output_dirs()
		logger.info(f"Output directories ready: {OUTPUT_VISUALIZATIONS_DIR} and {OUTPUT_STATISTICS_DIR}")

		# Load data
		try:
			logger.info("Loading Excel files...")
			csv_data = load_excel_files()
			if not csv_data:
				logger.warning("No Excel data loaded")
			else:
				logger.info(f"Loaded {len(csv_data)} Excel sheets")
		except Exception as e:
			logger.error(f"Error loading Excel files: {e}")
			csv_data = {}

		try:
			logger.info("Loading JSON files...")
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
			logger.info("Analyzing activities data...")
			result = analyze_activities(csv_data, json_data)
			if result:
				activities, unique_users_count, dropout_metrics, joining_metrics, campaign_metrics = result
				logger.info(f"Activities analysis complete: {unique_users_count} users")
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
			logger.info("Analyzing geofence data...")
			# Debug: Check if json_data is empty
			logger.info(f"json_data has {len(json_data)} items before calling analyze_geofence_data")
			analyze_geofence_data(json_data)
			logger.info("Geofence data analysis complete")
		except Exception as e:
			logger.error(f"Error analyzing geofence data: {e}")
			# Debug: Print the full error traceback
			import traceback
			logger.error(f"Error traceback: {traceback.format_exc()}")

		# Analyze visualizations, challenges, and tasks
		try:
			logger.info("Analyzing visualizations, challenges, and tasks...")
			analyze_visualizations_challenges_tasks(csv_data)
			logger.info("Visualizations, challenges, and tasks analysis complete")
		except Exception as e:
			logger.error(f"Error analyzing visualizations, challenges, and tasks: {e}")

		# Analyze impact of task types, points, and levels on engagement and adherence
		try:
			logger.info("Analyzing impact of task types, points, and levels on engagement and adherence...")
			activities_df_to_pass = activities if isinstance(activities, pd.DataFrame) else (activities[0] if isinstance(activities, tuple) and len(activities) > 0 else None)
			campaign_metrics_to_pass = campaign_metrics
			analyze_task_points_levels_impact(csv_data, activities_df_to_pass, campaign_metrics_to_pass)
			logger.info("Impact analysis complete")
		except Exception as e:
			logger.error(f"Error analyzing task types/points/levels impact: {e}")

		# Generate analysis report
		try:
			logger.info("Calling create_complete_report function")
			# Pass the entire result tuple to create_complete_report
			if result:
				report_path = create_complete_report(csv_data, json_data, result)
			else:
				report_path = create_complete_report(csv_data, json_data, activities)
			logger.info(f"Report generation completed, report saved to {report_path}")
		except Exception as e:
			logger.error(f"Error generating complete report: {e}")
			report_path = None

		# Print summary
		logger.info("Analysis complete")
		print("\nAnalysis complete.")
		print(f"- Visualizations saved to '{OUTPUT_VISUALIZATIONS_DIR}' directory.")
		print(f"- Statistics saved to '{OUTPUT_STATISTICS_DIR}' directory.")
		print(f"- Analysis report saved to '{report_path}'")

		if unique_users_count > 0:
			print(f"- Number of unique users in the campaign: {unique_users_count}")

		if campaign_metrics:
			try:
				print(f"- Campaign length: {campaign_metrics['length_days']} days (from {campaign_metrics['start_date']} to {campaign_metrics['end_date']})")
			except (KeyError, TypeError) as e:
				logger.error(f"Error accessing campaign metrics: {e}")

		if dropout_metrics:
			try:
				print(f"- User Drop-out Rates (time between first and last activity):")
				print(f"  * Average drop-out time: {dropout_metrics['avg_dropout_days']:.0f} days")
				print(f"  * Median drop-out time: {dropout_metrics['median_dropout_days']:.0f} days")
				print(f"  * Range: {dropout_metrics['min_dropout_days']:.0f} to {dropout_metrics['max_dropout_days']:.0f} days")
			except (KeyError, TypeError) as e:
				logger.error(f"Error accessing dropout metrics: {e}")

		if joining_metrics:
			try:
				print(f"- User Joining Rates (days between campaign start and first activity):")
				print(f"  * Average joining time: {joining_metrics['avg_joining_days']:.0f} days")
				print(f"  * Median joining time: {joining_metrics['median_joining_days']:.0f} days")
				print(f"  * Range: {joining_metrics['min_joining_days']:.0f} to {joining_metrics['max_joining_days']:.0f} days")
			except (KeyError, TypeError) as e:
				logger.error(f"Error accessing joining metrics: {e}")

	except Exception as e:
		logger.error(f"Unexpected error in main function: {e}")
		print(f"An error occurred during analysis. See log file for details.")

	# Log the completion of the analysis
	logger.info("Analysis complete with visualizations and statistics generated")

	# Only print a summary of the visualizations to keep the output concise
	print("\nVisualization categories:")
	print(f"   - Activities Analysis: {OUTPUT_VISUALIZATIONS_DIR}/activity_*.png")
	print(f"   - User Engagement: {OUTPUT_VISUALIZATIONS_DIR}/user_*.png, {OUTPUT_VISUALIZATIONS_DIR}/*_usage_*.png")

	print("\nFor a complete list of generated visualizations, check the log file or the visualizations directory.")

if __name__ == "__main__":
	main()
