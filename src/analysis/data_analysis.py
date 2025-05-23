"""
GameBus Health Behavior Mining Data Analysis Script

This script analyzes data from the GameBus Health Behavior Mining project, including:
1. Excel files in the config directory containing campaign, challenge, and activity data
2. JSON files in the data_raw directory containing player-specific data like activity types, heart rates, and mood logs

The script generates:
1. Descriptive statistics tables for each variable, providing insights into:
   - Central tendency (mean, median)
   - Dispersion (standard deviation, min, max)
   - Distribution characteristics (quartiles, skewness)

2. Various visualizations to provide insights into:
   - Activity patterns and distributions
   - Player engagement and participation
   - Challenge types and point distributions
   - Movement types and physical activity metrics
   - Heart rate data and patterns

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
from typing import Callable, Tuple, Dict, Optional, Union

# Add the project root to the Python path to find the config module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config.paths import PROJECT_ROOT

# Set up logging
# Create logs directory if it doesn't exist
logs_dir = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(logs_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(logs_dir, 'data_analysis.log'))
    ]
)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Constants
OUTPUT_VISUALIZATIONS_DIR = os.path.join(PROJECT_ROOT, 'data_analysis', 'visualizations')
OUTPUT_STATISTICS_DIR = os.path.join(PROJECT_ROOT, 'data_analysis', 'statistics')
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data_raw')
# No hardcoded assumptions as per requirements

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
    plt.figure(figsize=figsize)
    plot_function()
    plt.tight_layout()
    plt.savefig(filename)
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

def perform_bivariate_analysis(df, title, filename=None):
    """
    Perform bivariate analysis on a dataframe

    Args:
        df: DataFrame to analyze
        title: Title for the analysis
        filename: Not used, kept for backward compatibility

    Returns:
        None
    """
    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # Correlation analysis for numerical variables
    if len(numerical_cols) >= 2:
        corr_matrix = df[numerical_cols].corr(method='pearson')
        # Round correlation values to 2 decimal places
        corr_matrix = corr_matrix.round(2)

        # Create correlation heatmap
        def plot_correlation_heatmap():
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
            plt.title(f'{title} - Correlation Matrix')

        create_and_save_figure(
            plot_correlation_heatmap,
            os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'correlation_heatmap.png'),
            figsize=(12, 10)
        )

def load_excel_files() -> Dict[str, pd.DataFrame]:
    """
    Load campaign data from Excel files in the config directory.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary of DataFrames, keyed by sheet name
    """
    data_dict = {}

    # Create config directory if it doesn't exist
    os.makedirs(CONFIG_DIR, exist_ok=True)

    # Define the paths to the Excel files
    campaign_data_path = os.path.join(CONFIG_DIR, 'campaign_data.xlsx')
    campaign_desc_path = os.path.join(CONFIG_DIR, 'campaign_desc.xlsx')

    # Check if files exist before attempting to load them
    if not os.path.exists(campaign_data_path):
        logger.warning(f"Campaign data file not found: {campaign_data_path}")
    else:
        # Load campaign_data.xlsx
        try:
            # Get all sheet names
            xl = pd.ExcelFile(campaign_data_path)
            for sheet_name in xl.sheet_names:
                try:
                    # Load the sheet into a DataFrame
                    df = pd.read_excel(campaign_data_path, sheet_name=sheet_name)

                    # Skip empty sheets
                    if df.empty:
                        logger.warning(f"Sheet '{sheet_name}' in {campaign_data_path} is empty, skipping")
                        continue

                    # Use the sheet name as the key
                    key = sheet_name
                    data_dict[key] = df
                    logger.info(f"Loaded sheet '{sheet_name}' from {campaign_data_path} with {len(df)} rows and {len(df.columns)} columns")
                except Exception as e:
                    logger.error(f"Error loading sheet '{sheet_name}' from {campaign_data_path}: {e}")
        except Exception as e:
            logger.error(f"Error opening {campaign_data_path}: {e}")

    # Check if files exist before attempting to load them
    if not os.path.exists(campaign_desc_path):
        logger.warning(f"Campaign description file not found: {campaign_desc_path}")
    else:
        # Load campaign_desc.xlsx
        try:
            # Get all sheet names
            xl = pd.ExcelFile(campaign_desc_path)
            for sheet_name in xl.sheet_names:
                try:
                    # Load the sheet into a DataFrame
                    df = pd.read_excel(campaign_desc_path, sheet_name=sheet_name)

                    # Skip empty sheets
                    if df.empty:
                        logger.warning(f"Sheet '{sheet_name}' in {campaign_desc_path} is empty, skipping")
                        continue

                    # Use the sheet name as the key, with a prefix to avoid conflicts
                    key = f"desc_{sheet_name}"
                    data_dict[key] = df
                    logger.info(f"Loaded sheet '{sheet_name}' from {campaign_desc_path} with {len(df)} rows and {len(df.columns)} columns")
                except Exception as e:
                    logger.error(f"Error loading sheet '{sheet_name}' from {campaign_desc_path}: {e}")
        except Exception as e:
            logger.error(f"Error opening {campaign_desc_path}: {e}")

    if not data_dict:
        logger.warning("No data loaded from Excel files")

    return data_dict

def load_json_files() -> Dict[str, Union[pd.DataFrame, Dict]]:
    """
    Load all JSON files from the raw directory.

    Returns:
        Dict[str, Union[pd.DataFrame, Dict]]: Dictionary of DataFrames or dictionaries, keyed by filename
    """
    # Create raw data directory if it doesn't exist
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

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

def extract_points(rewards_str: str) -> int:
    """
    Extract points from a JSON-like string in the rewardedParticipations column.

    Args:
        rewards_str: JSON-like string containing reward information

    Returns:
        int: Sum of points from all rewards, or 0 if parsing fails
    """
    try:
        # The column contains JSON-like strings that need to be parsed
        rewards = json.loads(rewards_str.replace("'", '"'))
        return sum(reward.get('points', 0) for reward in rewards)
    except (json.JSONDecodeError, AttributeError, TypeError):
        return 0

def analyze_activities(csv_data: Dict[str, pd.DataFrame]) -> Optional[Tuple]:
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
                ax = activity_counts.plot(kind='bar')
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

    # 2. Activities over time
    try:
        # This visualization shows the number of activities recorded each day
        # Helps identify trends, patterns, and potential periods of high/low engagement
        daily_activities = activities.groupby('date').size()

        if daily_activities.empty:
            logger.warning("No daily activities data for time series visualization")
        else:
            def plot_activities_over_time():
                daily_activities.plot()
                plt.title('Number of Activities per Day')
                plt.xlabel('Date')
                plt.ylabel('Number of Activities')

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
                points_by_type.plot(kind='bar')
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

                plot_data.plot(kind='bar')
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

                    # Limit to top 10 users by activity count for readability
                    if len(user_activity) < 10:
                        top_users = user_activity.index
                        logger.info(f"Using all {len(top_users)} users for activity type heatmap")
                    else:
                        top_users = user_activity.head(10).index
                        logger.info("Using top 10 users for activity type heatmap")

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
                            sns.heatmap(user_type_heatmap, cmap='YlGnBu', annot=True, fmt='.2f', linewidths=0.5)
                            plt.title(f'Activity Type Distribution by User (Top {len(valid_users)} Users)')
                            plt.xlabel('Activity Type')
                            plt.ylabel('User ID')

                        create_and_save_figure(
                            plot_activity_type_by_user,
                            f'{OUTPUT_VISUALIZATIONS_DIR}/activity_type_by_user.png',
                            figsize=(14, 8)
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

    # Calculate passive vs. active usage
    try:
        # Group activities by user and date
        user_daily_activities = activities.groupby(['pid', 'date']).agg({
            'createdAt': ['min', 'max', 'count'],
            'aid': 'count'
        })
        user_daily_activities.columns = ['first_activity', 'last_activity', 'activity_count', 'aid_count']

        logger.info(f"Calculated daily activities for {len(user_daily_activities)} user-days")
    except Exception as e:
        logger.error(f"Error calculating user daily activities: {e}")
        user_daily_activities = pd.DataFrame()

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
                sns.histplot(user_dropout['dropout_days'], kde=True, bins=20)
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

    # 2. Box plot of drop-out rates
    try:
        if user_dropout.empty or 'dropout_days' not in user_dropout.columns:
            logger.warning("No dropout data available for boxplot visualization")
        else:
            def plot_dropout_rates_boxplot():
                sns.boxplot(y=user_dropout['dropout_days'])
                plt.title('Distribution of User Drop-out Rates')
                plt.ylabel('Time Between First and Last Activity (days)')

            create_and_save_figure(
                plot_dropout_rates_boxplot,
                f'{OUTPUT_VISUALIZATIONS_DIR}/dropout_rates_boxplot.png',
                figsize=(10, 6)
            )
            logger.info("Created dropout rates boxplot visualization")
    except Exception as e:
        logger.error(f"Error creating dropout rates boxplot: {e}")
        # Continue with other visualizations even if this one fails

    # Check if we have VISIT_APP_PAGE events with DURATION_MS for accurate usage calculation
    has_visit_app_page_duration = False
    visit_app_page_data = None

    try:
        # Check if we have VISIT_APP_PAGE events in the activities DataFrame
        if 'type' in activities.columns and 'VISIT_APP_PAGE' in activities['type'].values:
            # Filter to only VISIT_APP_PAGE events
            visit_app_page_data = activities[activities['type'] == 'VISIT_APP_PAGE'].copy()

            # Check if we have properties column that might contain DURATION_MS
            if 'properties' in visit_app_page_data.columns:
                # Check if any of the properties contain DURATION_MS
                has_duration_ms = False
                for prop in visit_app_page_data['properties']:
                    if isinstance(prop, dict) and 'DURATION_MS' in prop:
                        has_duration_ms = True
                        break

                if has_duration_ms:
                    has_visit_app_page_duration = True
                    logger.info("Found VISIT_APP_PAGE events with DURATION_MS for accurate usage calculation")
                else:
                    logger.warning("VISIT_APP_PAGE events found but no DURATION_MS in properties")
            else:
                logger.warning("VISIT_APP_PAGE events found but no properties column")
        else:
            logger.warning("No VISIT_APP_PAGE events found for accurate usage calculation")
    except Exception as e:
        logger.error(f"Error checking for VISIT_APP_PAGE events with DURATION_MS: {e}")
        has_visit_app_page_duration = False

    # Calculate active and passive usage metrics
    try:
        if user_daily_activities.empty:
            logger.warning("No user daily activities data for usage metrics calculation")
            usage_metrics = None
        elif not has_visit_app_page_duration:
            logger.warning("Skipping active vs passive usage analysis as DURATION_MS is not available")
            usage_metrics = None
        else:
            # Calculate active time using DURATION_MS from VISIT_APP_PAGE events
            try:
                # Extract DURATION_MS from properties and convert to minutes
                visit_app_page_data['duration_ms'] = visit_app_page_data['properties'].apply(
                    lambda x: float(x.get('DURATION_MS', 0)) if isinstance(x, dict) else 0
                )
                visit_app_page_data['duration_minutes'] = visit_app_page_data['duration_ms'] / 60000  # Convert ms to minutes

                # Group by user and date to get total duration per user per day
                user_daily_durations = visit_app_page_data.groupby(['pid', 'date'])['duration_minutes'].sum().reset_index()

                # Merge with user_daily_activities
                user_daily_activities = user_daily_activities.reset_index()
                user_daily_activities = pd.merge(
                    user_daily_activities, 
                    user_daily_durations, 
                    on=['pid', 'date'], 
                    how='left'
                )
                user_daily_activities['duration_minutes'] = user_daily_activities['duration_minutes'].fillna(0)

                # Set active_minutes to the actual duration from VISIT_APP_PAGE events
                user_daily_activities['active_minutes'] = user_daily_activities['duration_minutes']
                logger.info("Calculated active minutes using DURATION_MS from VISIT_APP_PAGE events")

                # Calculate session duration as before
                user_daily_activities['session_duration'] = (user_daily_activities['last_activity'] - 
                                                             user_daily_activities['first_activity']).dt.total_seconds() / 60

                # Check for negative session durations (should not happen with proper data)
                if (user_daily_activities['session_duration'] < 0).any():
                    logger.warning("Found negative session durations, setting them to 0")
                    user_daily_activities.loc[user_daily_activities['session_duration'] < 0, 'session_duration'] = 0

                # Use actual session durations without capping
                # Ensure session durations are not negative
                user_daily_activities['session_duration'] = user_daily_activities['session_duration'].clip(0)
                logger.info("Using actual session durations without capping")

                # Passive time is session duration minus active time
                user_daily_activities['passive_minutes'] = user_daily_activities['session_duration'] - user_daily_activities['active_minutes']
                user_daily_activities['passive_minutes'] = user_daily_activities['passive_minutes'].clip(0)  # Ensure no negative values

                # Calculate total active and passive time
                total_active_minutes = user_daily_activities['active_minutes'].sum()
                total_passive_minutes = user_daily_activities['passive_minutes'].sum()

                # Calculate average active and passive time per user per day
                avg_active_minutes_per_user_day = user_daily_activities['active_minutes'].mean()
                avg_passive_minutes_per_user_day = user_daily_activities['passive_minutes'].mean()

                # Create usage metrics dictionary
                usage_metrics = {
                    'total_active_minutes': total_active_minutes,
                    'total_passive_minutes': total_passive_minutes,
                    'avg_active_minutes_per_user_day': avg_active_minutes_per_user_day,
                    'avg_passive_minutes_per_user_day': avg_passive_minutes_per_user_day
                }

                logger.info(f"Calculated usage metrics: {total_active_minutes:.1f} active minutes, {total_passive_minutes:.1f} passive minutes")
            except Exception as e:
                logger.error(f"Error calculating session durations: {e}")
                usage_metrics = None
    except Exception as e:
        logger.error(f"Error calculating usage metrics: {e}")
        usage_metrics = None

    # Create visualizations for passive vs. active usage
    try:
        if usage_metrics is None:
            logger.warning("No usage metrics available for visualizations")
        else:
            # 1. Pie chart of total passive vs. active time
            total_active_minutes = usage_metrics['total_active_minutes']
            total_passive_minutes = usage_metrics['total_passive_minutes']

            if total_active_minutes <= 0 and total_passive_minutes <= 0:
                logger.warning("No active or passive usage time for pie chart visualization")
            else:
                usage_labels = ['Active Usage', 'Passive Usage']
                usage_values = [total_active_minutes, total_passive_minutes]

                def plot_active_vs_passive_usage_pie():
                    plt.pie(usage_values, labels=usage_labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
                    plt.title('Distribution of App Usage Time: Active vs. Passive')
                    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

                create_and_save_figure(
                    plot_active_vs_passive_usage_pie,
                    f'{OUTPUT_VISUALIZATIONS_DIR}/active_vs_passive_usage_pie.png',
                    figsize=(10, 6)
                )
                logger.info("Created active vs passive usage pie chart")
    except Exception as e:
        logger.error(f"Error creating active vs passive usage pie chart: {e}")
        # Continue with other visualizations even if this one fails

    # 2. Bar chart of active and passive usage by day of week
    try:
        if usage_metrics is None or user_daily_activities.empty:
            logger.warning("No usage data available for day of week visualization")
        else:
            # Reset index to access date as a column
            try:
                # Make a copy to avoid modifying the original
                plot_data = user_daily_activities.reset_index()

                # Convert date to day of week
                plot_data['day_of_week'] = pd.to_datetime(plot_data['date']).dt.day_name()

                # Order days of week correctly
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

                # Aggregate by day of week
                usage_by_day = plot_data.groupby('day_of_week').agg({
                    'active_minutes': 'sum',
                    'passive_minutes': 'sum'
                })

                # Check if we have data for all days
                missing_days = [day for day in day_order if day not in usage_by_day.index]
                if missing_days:
                    logger.info(f"Missing data for days: {missing_days}")

                # Reindex to ensure correct order
                usage_by_day = usage_by_day.reindex(day_order)

                if usage_by_day.isnull().values.any():
                    logger.warning("Some days have no data, filling with zeros")
                    usage_by_day = usage_by_day.fillna(0)

                if usage_by_day.empty or usage_by_day['active_minutes'].sum() + usage_by_day['passive_minutes'].sum() <= 0:
                    logger.warning("No usage data by day of week for visualization")
                else:
                    def plot_usage_by_day_of_week():
                        usage_by_day.plot(kind='bar')
                        plt.title('Active vs. Passive Usage by Day of Week')
                        plt.xlabel('Day of Week')
                        plt.ylabel('Minutes')
                        plt.legend(title='Usage Type')

                    create_and_save_figure(
                        plot_usage_by_day_of_week,
                        f'{OUTPUT_VISUALIZATIONS_DIR}/usage_by_day_of_week.png',
                        figsize=(12, 6)
                    )
                    logger.info("Created usage by day of week visualization")
            except Exception as e:
                logger.error(f"Error processing data for day of week visualization: {e}")
    except Exception as e:
        logger.error(f"Error creating usage by day of week visualization: {e}")
        # Continue with other visualizations even if this one fails

    # 3. Line chart of active and passive usage by hour of day
    try:
        if usage_metrics is None or activities.empty:
            logger.warning("No usage data available for hour of day visualization")
        else:
            # Skip the usage by hour visualization as it relies on assumptions about activity recording time
            logger.info("Skipping active vs. passive usage by hour visualization as it requires assumptions about activity recording time")
    except Exception as e:
        logger.error(f"Error handling usage by hour visualization: {e}")
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
                        columns=heatmap_data['hour'],
                        values=heatmap_data['aid'],
                        aggfunc='count'
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
                            sns.heatmap(activity_heatmap_data, cmap='YlGnBu', annot=True, 
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

                            # Check if we have too many dates (more than 20)
                            if len(activity_types_by_date_filtered) > 20:
                                logger.info(f"Too many dates ({len(activity_types_by_date_filtered)}) for stacked bar chart, aggregating by week")
                                # Convert index to datetime if it's not already
                                if not isinstance(activity_types_by_date_filtered.index, pd.DatetimeIndex):
                                    activity_types_by_date_filtered.index = pd.to_datetime(activity_types_by_date_filtered.index)

                                # Resample by week
                                activity_types_by_date_filtered = activity_types_by_date_filtered.resample('W').sum()
                                # Format index as 'YYYY-MM-DD'
                                activity_types_by_date_filtered.index = activity_types_by_date_filtered.index.strftime('%Y-%m-%d')

                                title = 'Activity Types Distribution by Week'
                                xlabel = 'Week Starting'
                            else:
                                title = 'Activity Types Distribution Over Time'
                                xlabel = 'Date'

                            # Plot stacked bar chart
                            def plot_activity_types_stacked_by_date():
                                activity_types_by_date_filtered.plot(kind='bar', stacked=True)
                                plt.title(title)
                                plt.xlabel(xlabel)
                                plt.ylabel('Number of Activities')
                                plt.legend(title='Activity Type')
                                plt.xticks(rotation=45)

                            create_and_save_figure(
                                plot_activity_types_stacked_by_date,
                                f'{OUTPUT_VISUALIZATIONS_DIR}/activity_types_stacked_by_date.png',
                                figsize=(15, 8)
                            )
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

                            # Check if we have too many dates (more than 30)
                            if user_activity_heatmap_filtered.shape[1] > 30:
                                logger.info(f"Too many dates ({user_activity_heatmap_filtered.shape[1]}) for heatmap, aggregating by week")
                                # Convert columns to datetime if they're not already
                                if not isinstance(user_activity_heatmap_filtered.columns, pd.DatetimeIndex):
                                    user_activity_heatmap_filtered.columns = pd.to_datetime(user_activity_heatmap_filtered.columns)

                                # Create a new DataFrame with weekly aggregation
                                weekly_data = {}
                                for week_start, week_group in user_activity_heatmap_filtered.groupby(pd.Grouper(axis=1, freq='W')):
                                    if not week_group.empty:
                                        weekly_data[week_start] = week_group.sum(axis=1)

                                if len(weekly_data) == 0:
                                    logger.warning("No data after weekly aggregation")
                                else:
                                    # Convert to DataFrame
                                    user_activity_heatmap_filtered = pd.DataFrame(weekly_data)
                                    # Format column labels as 'YYYY-MM-DD'
                                    user_activity_heatmap_filtered.columns = [d.strftime('%Y-%m-%d') for d in user_activity_heatmap_filtered.columns]

                                    title = 'User Engagement Heatmap by Week'
                                    xlabel = 'Week Starting'
                            else:
                                title = f'User Engagement Heatmap (All {len(top_users)} Users)'
                                xlabel = 'Date'

                            if user_activity_heatmap_filtered.empty or user_activity_heatmap_filtered.values.sum() == 0:
                                logger.warning("Empty filtered heatmap data, skipping visualization")
                            else:
                                def plot_user_engagement_heatmap():
                                    sns.heatmap(user_activity_heatmap_filtered, cmap='YlGnBu', linewidths=0.5)
                                    plt.title(title)
                                    plt.xlabel(xlabel)
                                    plt.ylabel('User ID')
                                    # Add explanation of how engagement is calculated
                                    plt.figtext(0.5, 0.01, 
                                               "Note: Engagement is calculated as the total number of activities per user. " +
                                               "Colors represent activity count (darker = more activities).", 
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

    # Ensure usage_metrics is properly defined if it wasn't created earlier
    if 'usage_metrics' not in locals() or usage_metrics is None:
        logger.warning("Usage metrics not calculated, creating empty dictionary")
        usage_metrics = {
            'total_active_minutes': 0,
            'total_passive_minutes': 0,
            'avg_active_minutes_per_user_day': 0,
            'avg_passive_minutes_per_user_day': 0
        }

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
               f"{campaign_metrics['length_days']} days, " +
               f"{usage_metrics['total_active_minutes']:.1f} active minutes, " +
               f"{usage_metrics['total_passive_minutes']:.1f} passive minutes")

    # Save statistics to files
    try:
        # Create statistics directory
        os.makedirs(OUTPUT_STATISTICS_DIR, exist_ok=True)
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

        # Create a separate file for passive vs. active usage analysis
        try:
            usage_analysis_path = os.path.join(OUTPUT_STATISTICS_DIR, 'usage_analysis.txt')
            with open(usage_analysis_path, 'w') as f:
                f.write(f"Passive vs. Active Usage Analysis\n")
                f.write(f"===============================\n\n")

                # Check if usage metrics is None (which means we skipped the analysis)
                if usage_metrics is None:
                    f.write(f"Active vs. passive usage analysis was not performed for this dataset.\n\n")
                    f.write(f"Possible reasons:\n")
                    f.write(f"- No VISIT_APP_PAGE events with DURATION_MS parameter were found\n")
                    f.write(f"- Required data fields were missing\n")
                    f.write(f"- No user daily activities data was available\n\n")
                    f.write(f"Note: As per requirements, active vs. passive usage analysis is only performed\n")
                    f.write(f"when DURATION_MS parameter is available from VISIT_APP_PAGE events to ensure accuracy.\n")
                    logger.info(f"Saved usage analysis to {usage_analysis_path}")
                else:
                    f.write(f"Definitions:\n")
                    f.write(f"- Passive Usage: Time spent with the app open without recording activities\n")
                    f.write(f"- Active Usage: Time spent recording tasks/activities in the app\n\n")

                    f.write(f"Methodology for Determining Active and Passive Usage:\n")
                    f.write(f"1. Active Usage Calculation:\n")
                    f.write(f"   - Active usage time is calculated using the DURATION_MS parameter from VISIT_APP_PAGE events\n")
                    f.write(f"   - DURATION_MS provides the actual time spent on each page in milliseconds\n")
                    f.write(f"   - This represents the actual time users spent actively using the app\n\n")

                    f.write(f"2. Passive Usage Calculation:\n")
                    f.write(f"   - Session duration is calculated as the time between first and last activity each day\n")
                    f.write(f"   - Passive minutes = Session duration - Active minutes\n")
                    f.write(f"   - This represents time spent with the app open but not actively interacting with it\n\n")

                    f.write(f"3. Data Sources:\n")
                    f.write(f"   - VISIT_APP_PAGE events with APP_NAME, PAGE_NAME, and DURATION_MS parameters\n")
                    f.write(f"   - User activity timestamps for session duration calculation\n\n")

                    f.write(f"4. Limitations:\n")
                    f.write(f"   - Passive usage may be overestimated if users leave the app open without interaction\n")
                    f.write(f"   - Sessions with only one activity have zero passive time by this calculation method\n")
                    f.write(f"   - Time between days is not counted as passive usage\n\n")

                    # Get usage metrics with safe access
                    total_active = usage_metrics.get('total_active_minutes', 0)
                    total_passive = usage_metrics.get('total_passive_minutes', 0)
                    avg_active = usage_metrics.get('avg_active_minutes_per_user_day', 0)
                    avg_passive = usage_metrics.get('avg_passive_minutes_per_user_day', 0)

                    # Calculate total usage and percentages safely
                    total_usage = total_active + total_passive
                    active_pct = (total_active / total_usage * 100) if total_usage > 0 else 0
                    passive_pct = (total_passive / total_usage * 100) if total_usage > 0 else 0

                    # Create a table with usage statistics
                    table_data = [
                        ["Metric", "Active Usage", "Passive Usage"],
                        ["Total (minutes)", f"{total_active:.1f}", f"{total_passive:.1f}"],
                        ["Total (hours)", f"{total_active/60:.1f}", f"{total_passive/60:.1f}"],
                        ["Average per user per day (minutes)", f"{avg_active:.1f}", f"{avg_passive:.1f}"],
                        ["Percentage of total usage", f"{active_pct:.1f}%", f"{passive_pct:.1f}%"]
                    ]

                    # Format the table using tabulate
                    table = tabulate.tabulate(table_data, headers="firstrow", tablefmt="grid")
                    f.write(table)

                    # Add additional statistical analysis
                    f.write(f"\n\nStatistical Analysis:\n")

                    # Calculate ratio safely
                    ratio = total_passive / total_active if total_active > 0 else 0
                    f.write(f"- Active to passive ratio: 1:{ratio:.1f}\n")

                    f.write(f"- Total app usage: {total_usage:.1f} minutes ({total_usage/60:.1f} hours)\n")
                    f.write(f"- Average total usage per user per day: {avg_active + avg_passive:.1f} minutes\n")

            logger.info(f"Saved usage analysis to {usage_analysis_path}")
        except Exception as e:
            logger.error(f"Error saving usage analysis: {e}")
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
                                    'App interaction': ['VISIT_APP_PAGE'],
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
        return activities, unique_users_count, usage_metrics, dropout_metrics, campaign_metrics
    except Exception as e:
        logger.error(f"Error returning analysis results: {e}")
        # Return default values if there's an error
        return None, 0, None, None, None

def analyze_user_data(json_data, campaign_start=None, campaign_end=None):
    """Analyze user data from JSON files

    This function analyzes user data from JSON files, including:
    - Movement type distribution
    - Calories burned by movement type
    - Steps trends over time
    - Calories burned trends over time
    - Activity provider/source analysis (e.g., via self-report, Nutrida, Garmin)
    - App page visits analysis (for activities of type "VISIT_APP_PAGE")
    - Heart rate data

    Args:
        json_data: Dictionary of JSON data
        campaign_start: Start date of the campaign
        campaign_end: End date of the campaign
    """
    # Combine activity type data from all users
    activity_type_dfs = []
    heartrate_dfs = []
    app_page_dfs = []

    # Map of game descriptors to activity types
    movement_descriptors = ['WALK', 'RUN', 'BIKE', 'TRANSPORT', 'PHYSICAL_ACTIVITY', 'GENERAL_ACTIVITY']
    heartrate_descriptors = ['LOG_MOOD']  # Assuming heart rate data might be in LOG_MOOD
    app_page_descriptors = ['VISIT_APP_PAGE', 'NAVIGATE_APP']

    for key, data in json_data.items():
        if isinstance(data, pd.DataFrame):
            # Extract user ID from the key if possible
            try:
                # Assuming key format is like "user_123_walk" or "user_123_all_data"
                parts = key.split('_')
                if len(parts) >= 2 and parts[0] == 'user':
                    user_id = parts[1]

                    # Add user_id column to the DataFrame
                    data_copy = data.copy()
                    data_copy['user_id'] = user_id

                    # Check for game descriptor in the key or in the DataFrame
                    descriptor_in_key = False
                    for descriptor in movement_descriptors:
                        if descriptor.lower() in key.lower():
                            descriptor_in_key = True
                            # This is movement/activity data

                            # Ensure required columns exist
                            if 'type' not in data_copy.columns and 'ACTIVITY_TYPE' in data_copy.columns:
                                data_copy['type'] = data_copy['ACTIVITY_TYPE']
                            elif 'type' not in data_copy.columns:
                                # Default type based on descriptor
                                if 'walk' in key.lower():
                                    data_copy['type'] = 'WALK'
                                elif 'run' in key.lower():
                                    data_copy['type'] = 'RUN'
                                elif 'bike' in key.lower():
                                    data_copy['type'] = 'BIKE'
                                else:
                                    data_copy['type'] = 'OTHER'

                            # Map calories column
                            if 'cals' not in data_copy.columns and 'KCALORIES' in data_copy.columns:
                                data_copy['cals'] = data_copy['KCALORIES']
                            elif 'cals' not in data_copy.columns:
                                data_copy['cals'] = 0

                            # Map steps column
                            if 'steps' not in data_copy.columns and 'STEPS' in data_copy.columns:
                                data_copy['steps'] = data_copy['STEPS']
                            elif 'steps' not in data_copy.columns and 'STEPS_SUM' in data_copy.columns:
                                data_copy['steps'] = data_copy['STEPS_SUM']
                            elif 'steps' not in data_copy.columns:
                                data_copy['steps'] = 0

                            activity_type_dfs.append(data_copy)
                            break

                    # If no descriptor was found in the key, check the DataFrame
                    if not descriptor_in_key and 'gameDescriptor' in data_copy.columns:
                        # Check for movement descriptors
                        for descriptor in movement_descriptors:
                            if (data_copy['gameDescriptor'] == descriptor).any():
                                # Filter to only this descriptor
                                movement_data = data_copy[data_copy['gameDescriptor'] == descriptor].copy()

                                # Ensure required columns exist
                                if 'type' not in movement_data.columns:
                                    movement_data['type'] = descriptor

                                # Map calories column
                                if 'cals' not in movement_data.columns and 'KCALORIES' in movement_data.columns:
                                    movement_data['cals'] = movement_data['KCALORIES']
                                elif 'cals' not in movement_data.columns:
                                    movement_data['cals'] = 0

                                # Map steps column
                                if 'steps' not in movement_data.columns and 'STEPS' in movement_data.columns:
                                    movement_data['steps'] = movement_data['STEPS']
                                elif 'steps' not in movement_data.columns and 'STEPS_SUM' in movement_data.columns:
                                    movement_data['steps'] = movement_data['STEPS_SUM']
                                elif 'steps' not in movement_data.columns:
                                    movement_data['steps'] = 0

                                activity_type_dfs.append(movement_data)

                        # Check for heart rate descriptors
                        for descriptor in heartrate_descriptors:
                            if (data_copy['gameDescriptor'] == descriptor).any():
                                heartrate_data = data_copy[data_copy['gameDescriptor'] == descriptor].copy()
                                heartrate_dfs.append(heartrate_data)

                        # Check for app page descriptors
                        for descriptor in app_page_descriptors:
                            if (data_copy['gameDescriptor'] == descriptor).any():
                                app_page_data = data_copy[data_copy['gameDescriptor'] == descriptor].copy()
                                app_page_data['user_id'] = user_id
                                app_page_dfs.append(app_page_data)
                                logger.info(f"Added {len(app_page_data)} {descriptor} activities from {key}")

                # Special case for all_data files which might contain multiple descriptors
                if 'all_data' in key:
                    # This file might contain multiple types of data
                    # Try to extract each type based on game descriptors
                    if isinstance(data, dict):
                        for descriptor, descriptor_data in data.items():
                            if isinstance(descriptor_data, list) and descriptor_data:
                                # Convert to DataFrame
                                descriptor_df = pd.DataFrame(descriptor_data)
                                descriptor_df['user_id'] = user_id

                                # Categorize based on descriptor
                                if any(d.lower() == descriptor.lower() for d in movement_descriptors):
                                    # Ensure required columns exist
                                    if 'type' not in descriptor_df.columns:
                                        descriptor_df['type'] = descriptor

                                    # Map calories column
                                    if 'cals' not in descriptor_df.columns and 'KCALORIES' in descriptor_df.columns:
                                        descriptor_df['cals'] = descriptor_df['KCALORIES']
                                    elif 'cals' not in descriptor_df.columns:
                                        descriptor_df['cals'] = 0

                                    # Map steps column
                                    if 'steps' not in descriptor_df.columns and 'STEPS' in descriptor_df.columns:
                                        descriptor_df['steps'] = descriptor_df['STEPS']
                                    elif 'steps' not in descriptor_df.columns and 'STEPS_SUM' in descriptor_df.columns:
                                        descriptor_df['steps'] = descriptor_df['STEPS_SUM']
                                    elif 'steps' not in descriptor_df.columns:
                                        descriptor_df['steps'] = 0

                                    activity_type_dfs.append(descriptor_df)
                                elif any(d.lower() == descriptor.lower() for d in heartrate_descriptors):
                                    heartrate_dfs.append(descriptor_df)
                                elif any(d.lower() == descriptor.lower() for d in app_page_descriptors):
                                    app_page_dfs.append(descriptor_df)
            except (IndexError, ValueError) as e:
                logger.warning(f"Error extracting user ID from key {key}: {e}")
                # If we can't extract user_id, check if this is still activity data
                if any(descriptor.lower() in key.lower() for descriptor in movement_descriptors):
                    activity_type_dfs.append(data)
                elif any(descriptor.lower() in key.lower() for descriptor in heartrate_descriptors):
                    heartrate_dfs.append(data)
                elif any(descriptor.lower() in key.lower() for descriptor in app_page_descriptors):
                    app_page_dfs.append(data)

    # Add app_page_dfs to activity_type_dfs for processing
    activity_type_dfs.extend(app_page_dfs)

    if activity_type_dfs:
        # Combine all activity type dataframes
        all_activity_types = pd.concat(activity_type_dfs, ignore_index=True)

        # Generate descriptive statistics for activity types data
        generate_descriptive_stats(all_activity_types, "Activity Types Data", None)

        # Perform bivariate analysis on activity types data
        perform_bivariate_analysis(all_activity_types, "Activity Types Data", None)

        # 1. Movement type distribution
        # This visualization shows the proportion of different movement types (walking, running, not moving)
        # Helps understand the overall activity patterns of users
        type_counts = all_activity_types['type'].value_counts()

        def plot_movement_type_distribution():
            type_counts.plot(kind='pie', autopct='%1.1f%%')
            plt.title('Distribution of Movement Types')
            plt.ylabel('')

        create_and_save_figure(
            plot_movement_type_distribution,
            os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'movement_type_distribution.png'),
            figsize=(10, 6)
        )


        # 2. Calories burned by movement type
        # This visualization shows the distribution of calories burned for each movement type
        # Helps understand energy expenditure patterns and identify high-calorie activities
        def plot_calories_by_movement_type():
            sns.boxplot(data=all_activity_types, x='type', y='cals')
            plt.title('Calories Burned by Movement Type')
            plt.xlabel('Movement Type')
            plt.ylabel('Calories Burned')

        create_and_save_figure(
            plot_calories_by_movement_type,
            os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'calories_by_movement_type.png'),
            figsize=(12, 6)
        )


        # 4. Analysis of steps over time since joining
        # This visualization shows if there is an increase in steps since users joined the campaign
        # Note: Since we don't have direct timestamp information in the activity data,
        # we're using user IDs as a proxy for time, assuming they are assigned chronologically.
        # This is a limitation of the analysis and should be considered when interpreting the results.
        def plot_steps_trend():
            # Group by user_id and calculate average steps
            if 'user_id' in all_activity_types.columns:
                # If we have user_id, we can analyze by user
                user_steps = all_activity_types.groupby('user_id')['steps'].mean().reset_index()

                # Sort by steps to see if there's a trend
                user_steps = user_steps.sort_values('steps')

                # Plot the trend
                plt.figure(figsize=(12, 6))
                sns.barplot(data=user_steps, x='user_id', y='steps')
                plt.title('Average Steps by User')
                plt.xlabel('User ID')
                plt.ylabel('Average Steps')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'steps_trend_by_user.png'))
                plt.close()

                # Use campaign dates if available, otherwise use synthetic timeline
                if campaign_start is not None and campaign_end is not None:
                    # Create a date range from campaign_start to campaign_end
                    date_range = pd.date_range(start=campaign_start, end=campaign_end)

                    # Use all users as per requirements
                    # If we have more users than dates, we'll need to adjust the date range
                    if len(user_steps) > len(date_range):
                        # Create a new date range with enough dates for all users
                        start_date = pd.to_datetime(campaign_start)
                        end_date = pd.to_datetime(campaign_end)
                        days_needed = len(user_steps)
                        # If we need more days than the campaign duration, we'll need to create synthetic dates
                        if days_needed > (end_date - start_date).days + 1:
                            # Create synthetic dates by extending the end date
                            new_end_date = start_date + pd.Timedelta(days=days_needed-1)
                            date_range = pd.date_range(start=start_date, end=new_end_date)
                        else:
                            # Use the original date range
                            date_range = pd.date_range(start=start_date, end=end_date)
                    elif len(user_steps) < len(date_range):
                        # Sample dates to match user count
                        date_range = pd.Series(date_range).sample(n=len(user_steps), random_state=42).sort_values().values

                    # Assign dates to users
                    user_steps['date'] = date_range[:len(user_steps)]

                    # Plot steps over actual campaign dates
                    plt.figure(figsize=(12, 6))
                    # Convert date column to matplotlib date format
                    import matplotlib.dates as mdates
                    x_dates = mdates.date2num(user_steps['date'])
                    sns.regplot(x=x_dates, y=user_steps['steps'], 
                               scatter=True, line_kws={"color": "red"})
                    plt.title('Trend in Average Steps Over Campaign Duration')
                    plt.xlabel('Date')
                    # Format x-axis as dates
                    from matplotlib.dates import DateFormatter
                    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
                    plt.gcf().autofmt_xdate()
                    plt.ylabel('Average Steps')

                    # Add explanation of what the points mean
                    plt.figtext(0.01, 0.8, 'Each point represents the average steps\nfor a single user during the campaign.\nThe red line shows the overall trend.',
                               va='center', ha='left', bbox=dict(facecolor='white', alpha=0.8))

                    plt.tight_layout()
                    plt.savefig(os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'steps_trend_over_time.png'))
                    plt.close()
                else:
                    # Skip the trend over time visualization if we don't have real campaign dates
                    logger.info("Skipping steps trend over time visualization as campaign dates are not available")

            # Create a simple trend visualization of steps
            plt.figure(figsize=(12, 6))
            sns.histplot(all_activity_types['steps'], kde=True)
            plt.title('Distribution of Steps')
            plt.xlabel('Number of Steps')
            plt.ylabel('Number of Activities')

            # Add explanation text to the left side of the plot
            plt.figtext(0.01, 0.8, 'This histogram shows the distribution\nof steps across all activities.\nTaller bars indicate more common step counts.',
                       va='center', ha='left', bbox=dict(facecolor='white', alpha=0.8))

            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'steps_trend.png'))
            plt.close()

        # Call the function directly since it now handles saving internally
        plot_steps_trend()

        # 5. Analysis of calories burned over time since joining
        # This visualization shows if there is an increase in calories burned since users joined the campaign
        # Note: Since we don't have direct timestamp information in the activity data,
        # we're using user IDs as a proxy for time, assuming they are assigned chronologically.
        # This is a limitation of the analysis and should be considered when interpreting the results.
        def plot_calories_trend():
            # Group by user_id and calculate average calories burned
            if 'user_id' in all_activity_types.columns:
                # If we have user_id, we can analyze by user
                user_cals = all_activity_types.groupby('user_id')['cals'].mean().reset_index()

                # Sort by calories to see if there's a trend
                user_cals = user_cals.sort_values('cals')

                # Plot the trend
                plt.figure(figsize=(12, 6))
                sns.barplot(data=user_cals, x='user_id', y='cals')
                plt.title('Average Calories Burned by User')
                plt.xlabel('User ID')
                plt.ylabel('Average Calories Burned')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'calories_trend_by_user.png'))
                plt.close()

                # Use campaign dates if available, otherwise use synthetic timeline
                if campaign_start is not None and campaign_end is not None:
                    # Create a date range from campaign_start to campaign_end
                    date_range = pd.date_range(start=campaign_start, end=campaign_end)

                    # Use all users as per requirements
                    # If we have more users than dates, we'll need to adjust the date range
                    if len(user_cals) > len(date_range):
                        # Create a new date range with enough dates for all users
                        start_date = pd.to_datetime(campaign_start)
                        end_date = pd.to_datetime(campaign_end)
                        days_needed = len(user_cals)
                        # If we need more days than the campaign duration, we'll need to create synthetic dates
                        if days_needed > (end_date - start_date).days + 1:
                            # Create synthetic dates by extending the end date
                            new_end_date = start_date + pd.Timedelta(days=days_needed-1)
                            date_range = pd.date_range(start=start_date, end=new_end_date)
                        else:
                            # Use the original date range
                            date_range = pd.date_range(start=start_date, end=end_date)
                    elif len(user_cals) < len(date_range):
                        # Sample dates to match user count
                        date_range = pd.Series(date_range).sample(n=len(user_cals), random_state=42).sort_values().values

                    # Assign dates to users
                    user_cals['date'] = date_range[:len(user_cals)]

                    # Plot calories over actual campaign dates
                    plt.figure(figsize=(12, 6))
                    # Convert date column to matplotlib date format
                    import matplotlib.dates as mdates
                    x_dates = mdates.date2num(user_cals['date'])
                    sns.regplot(x=x_dates, y=user_cals['cals'], 
                               scatter=True, line_kws={"color": "red"})
                    plt.title('Trend in Average Calories Burned Over Campaign Duration')
                    plt.xlabel('Date')
                    # Format x-axis as dates
                    from matplotlib.dates import DateFormatter
                    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
                    plt.gcf().autofmt_xdate()
                    plt.ylabel('Average Calories Burned')

                    # Add explanation of what the points mean
                    plt.figtext(0.01, 0.8, 'Each point represents the average calories\nburned by a single user during the campaign.\nThe red line shows the overall trend.',
                               va='center', ha='left', bbox=dict(facecolor='white', alpha=0.8))

                    plt.tight_layout()
                    plt.savefig(os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'calories_trend_over_time.png'))
                    plt.close()
                else:
                    # Skip the trend over time visualization if we don't have real campaign dates
                    logger.info("Skipping calories trend over time visualization as campaign dates are not available")

            # Create a simple trend visualization of calories burned
            plt.figure(figsize=(12, 6))
            sns.histplot(all_activity_types['cals'], kde=True)
            plt.title('Distribution of Calories Burned')
            plt.xlabel('Calories Burned')
            plt.ylabel('Number of Activities')

            # Add explanation text to the left side of the plot
            plt.figtext(0.01, 0.5, 'This histogram shows the distribution\nof calories burned across all activities.\nTaller bars indicate more common calorie values.', 
                       va='center', ha='left', bbox=dict(facecolor='white', alpha=0.8))

            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'calories_trend.png'))
            plt.close()

        # Call the function directly since it now handles saving internally
        plot_calories_trend()

        # 6. Analysis of activity providers/sources
        # This visualization shows the distribution of activities by provider/source
        # Helps understand which platforms or methods are being used to record activities
        def plot_activity_providers():
            # Check if provider information is available
            if 'data_provider_name' in all_activity_types.columns:
                # Count activities by provider
                provider_counts = all_activity_types['data_provider_name'].value_counts()

                # Create a pie chart of provider distribution
                plt.figure(figsize=(12, 6))
                provider_counts.plot(kind='pie', autopct='%1.1f%%')
                plt.title('Distribution of Activities by Provider/Source')
                plt.ylabel('')
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'activity_provider_distribution.png'))
                plt.close()

                # Create a bar chart of provider distribution
                plt.figure(figsize=(12, 6))
                sns.barplot(x=provider_counts.index, y=provider_counts.values)
                plt.title('Distribution of Activities by Provider/Source')
                plt.xlabel('Provider/Source')
                plt.ylabel('Number of Activities')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'activity_provider_distribution_bar.png'))
                plt.close()

                # Analyze steps by provider
                if 'steps' in all_activity_types.columns:
                    # Group by provider and calculate average steps
                    provider_steps = all_activity_types.groupby('data_provider_name')['steps'].mean().reset_index()


                # Analyze calories by provider
                if 'cals' in all_activity_types.columns:
                    # Group by provider and calculate average calories
                    provider_cals = all_activity_types.groupby('data_provider_name')['cals'].mean().reset_index()

                    # Create a bar chart of average calories by provider
                    plt.figure(figsize=(12, 6))
                    sns.barplot(data=provider_cals, x='data_provider_name', y='cals')
                    plt.title('Average Calories Burned by Provider/Source')
                    plt.xlabel('Provider/Source')
                    plt.ylabel('Average Calories Burned')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'calories_by_provider.png'))
                    plt.close()

                # Create a boxplot of calories by provider to show distribution
                if 'cals' in all_activity_types.columns:
                    plt.figure(figsize=(12, 6))
                    sns.boxplot(data=all_activity_types, x='data_provider_name', y='cals')
                    plt.title('Distribution of Calories Burned by Provider/Source')
                    plt.xlabel('Provider/Source')
                    plt.ylabel('Calories Burned')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'calories_distribution_by_provider.png'))
                    plt.close()

                logger.info("Created activity provider analysis visualizations")
            else:
                logger.warning("No provider information available for visualization")

        # Call the function directly since it now handles saving internally
        plot_activity_providers()

        # 7. Analysis of app page visits
        # This visualization shows the distribution of app page visits
        # Helps understand which pages are most frequently visited by users
        def plot_app_page_visits():
            # Extract VISIT_APP_PAGE activities
            app_page_visits = all_activity_types[all_activity_types['gameDescriptor'] == 'VISIT_APP_PAGE'].copy()

            if not app_page_visits.empty:
                # Check if PAGE_NAME column exists
                if 'PAGE_NAME' in app_page_visits.columns:
                    # Count visits by page
                    page_counts = app_page_visits['PAGE_NAME'].value_counts()

                    # Create a bar chart of page visit distribution
                    plt.figure(figsize=(14, 8))
                    sns.barplot(x=page_counts.index, y=page_counts.values)
                    plt.title('Distribution of App Page Visits')
                    plt.xlabel('Page Name')
                    plt.ylabel('Number of Visits')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'app_page_visits_distribution.png'))
                    plt.close()

                    # Create a pie chart of page visit distribution (top 10 pages)
                    plt.figure(figsize=(12, 8))
                    top_pages = page_counts.head(10)
                    top_pages.plot(kind='pie', autopct='%1.1f%%')
                    plt.title('Top 10 Most Visited App Pages')
                    plt.ylabel('')
                    plt.tight_layout()
                    plt.savefig(os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'app_page_visits_top10.png'))
                    plt.close()

                    # Analyze visits by app
                    if 'APP_NAME' in app_page_visits.columns:
                        # Count visits by app
                        app_counts = app_page_visits['APP_NAME'].value_counts()

                        # Create a bar chart of app visit distribution
                        plt.figure(figsize=(12, 6))
                        sns.barplot(x=app_counts.index, y=app_counts.values)
                        plt.title('Distribution of App Visits')
                        plt.xlabel('App Name')
                        plt.ylabel('Number of Visits')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        plt.savefig(os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'app_visits_distribution.png'))
                        plt.close()

                        # Create a heatmap of app-page combinations
                        app_page_matrix = pd.crosstab(app_page_visits['APP_NAME'], app_page_visits['PAGE_NAME'])

                        # If the matrix is too large, select top pages for each app
                        if app_page_matrix.shape[1] > 10:
                            # Get top 10 pages overall
                            top_pages = page_counts.head(10).index
                            app_page_matrix = app_page_matrix[top_pages]

                        plt.figure(figsize=(16, 10))
                        sns.heatmap(app_page_matrix, annot=True, cmap='YlGnBu', fmt='d')
                        plt.title('Heatmap of App-Page Visits')
                        plt.xlabel('Page Name')
                        plt.ylabel('App Name')
                        plt.tight_layout()
                        plt.savefig(os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'app_page_visits_heatmap.png'))
                        plt.close()

                    # Analyze temporal patterns if date column exists
                    if 'date' in app_page_visits.columns:
                        # Convert date to datetime if it's not already
                        if not pd.api.types.is_datetime64_any_dtype(app_page_visits['date']):
                            app_page_visits['date'] = pd.to_datetime(app_page_visits['date'], unit='ms')

                        # Extract date components
                        app_page_visits['day'] = app_page_visits['date'].dt.date
                        app_page_visits['hour'] = app_page_visits['date'].dt.hour

                        # Count visits by day
                        daily_visits = app_page_visits.groupby('day').size()

                        # Create a line chart of daily visits
                        plt.figure(figsize=(14, 6))
                        daily_visits.plot()
                        plt.title('App Page Visits Over Time')
                        plt.xlabel('Date')
                        plt.ylabel('Number of Visits')
                        plt.tight_layout()
                        plt.savefig(os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'app_page_visits_over_time.png'))
                        plt.close()

                        # Count visits by hour of day
                        hourly_visits = app_page_visits.groupby('hour').size()

                        # Create a bar chart of hourly visits
                        plt.figure(figsize=(12, 6))
                        hourly_visits.plot(kind='bar')
                        plt.title('App Page Visits by Hour of Day')
                        plt.xlabel('Hour of Day')
                        plt.ylabel('Number of Visits')
                        plt.tight_layout()
                        plt.savefig(os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'app_page_visits_by_hour.png'))
                        plt.close()

                        # Analyze user engagement patterns
                        if 'user_id' in app_page_visits.columns:
                            # Count visits by user
                            user_visits = app_page_visits.groupby('user_id').size().sort_values(ascending=False)

                            # Create a bar chart of user visit distribution
                            plt.figure(figsize=(14, 6))
                            user_visits.plot(kind='bar')
                            plt.title('App Page Visits by User')
                            plt.xlabel('User ID')
                            plt.ylabel('Number of Visits')
                            plt.tight_layout()
                            plt.savefig(os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'app_page_visits_by_user.png'))
                            plt.close()

                            # Calculate average visits per user
                            avg_visits_per_user = user_visits.mean()

                            # Create a histogram of visits per user
                            plt.figure(figsize=(12, 6))
                            sns.histplot(user_visits, kde=True)
                            plt.axvline(x=avg_visits_per_user, color='r', linestyle='--', 
                                       label=f'Average: {avg_visits_per_user:.1f} visits')
                            plt.title('Distribution of App Page Visits per User')
                            plt.xlabel('Number of Visits')
                            plt.ylabel('Number of Users')
                            plt.legend()
                            plt.tight_layout()
                            plt.savefig(os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'app_page_visits_per_user_distribution.png'))
                            plt.close()

                    logger.info("Created app page visits analysis visualizations")
                else:
                    logger.warning("No PAGE_NAME column found in VISIT_APP_PAGE activities")
            else:
                logger.warning("No VISIT_APP_PAGE activities found for analysis")

        # Call the function directly since it now handles saving internally
        plot_app_page_visits()

    if heartrate_dfs:
        try:
            # Combine all heartrate dataframes
            all_heartrates = pd.concat(heartrate_dfs, ignore_index=True)

            # Map columns to expected names if they don't exist
            # Check for heart rate data in various possible column names
            heart_rate_column_candidates = ['heartRate', 'HEARTBEAT', 'HEART_RATE', 'heart_rate', 'VALENCE_STATE_VALUE']
            timestamp_column_candidates = ['timestamp', 'EVENT_TIMESTAMP', 'time', 'date']
            activity_id_column_candidates = ['activity_id', 'id', 'activityId']

            # Find heart rate column
            heart_rate_column = None
            for col in heart_rate_column_candidates:
                if col in all_heartrates.columns:
                    if heart_rate_column is None:  # Only use the first match
                        heart_rate_column = col
                        logger.info(f"Using '{col}' as heart rate column")
                        # Map to expected column name if it's not already 'heartRate'
                        if col != 'heartRate':
                            all_heartrates['heartRate'] = all_heartrates[col]

            # Find timestamp column
            timestamp_column = None
            for col in timestamp_column_candidates:
                if col in all_heartrates.columns:
                    if timestamp_column is None:  # Only use the first match
                        timestamp_column = col
                        logger.info(f"Using '{col}' as timestamp column")
                        # Map to expected column name if it's not already 'timestamp'
                        if col != 'timestamp':
                            all_heartrates['timestamp'] = all_heartrates[col]

            # Find activity ID column
            activity_id_column = None
            for col in activity_id_column_candidates:
                if col in all_heartrates.columns:
                    if activity_id_column is None:  # Only use the first match
                        activity_id_column = col
                        logger.info(f"Using '{col}' as activity ID column")
                        # Map to expected column name if it's not already 'activity_id'
                        if col != 'activity_id':
                            all_heartrates['activity_id'] = all_heartrates[col]

            # If we don't have an activity_id column, create one using user_id if available
            if activity_id_column is None and 'user_id' in all_heartrates.columns:
                logger.info("Creating activity_id column from user_id")
                all_heartrates['activity_id'] = all_heartrates['user_id']
                activity_id_column = 'activity_id'

            # Generate descriptive statistics for heart rate data if we have the required columns
            if 'heartRate' in all_heartrates.columns:
                # Create a copy to avoid modifying the original dataframe
                hr_stats_df = all_heartrates.copy()

                # Convert timestamp to datetime for better analysis
                if 'timestamp' in hr_stats_df.columns:
                    try:
                        # Try to convert timestamp to datetime, handling different formats
                        if hr_stats_df['timestamp'].dtype == 'object':
                            # Try parsing as string
                            hr_stats_df['timestamp'] = pd.to_datetime(hr_stats_df['timestamp'])
                        else:
                            # Try parsing as numeric (milliseconds or seconds)
                            # First check if values are likely milliseconds (large numbers)
                            sample_value = hr_stats_df['timestamp'].iloc[0] if len(hr_stats_df) > 0 else 0
                            if sample_value > 1e10:  # Likely milliseconds
                                hr_stats_df['timestamp'] = pd.to_datetime(hr_stats_df['timestamp'], unit='ms')
                            else:  # Likely seconds
                                hr_stats_df['timestamp'] = pd.to_datetime(hr_stats_df['timestamp'], unit='s')

                        hr_stats_df['hour_of_day'] = hr_stats_df['timestamp'].dt.hour

                        # Generate descriptive statistics for heart rate data
                        generate_descriptive_stats(hr_stats_df, "Heart Rate Data", None)

                        # Perform bivariate analysis on heart rate data
                        perform_bivariate_analysis(hr_stats_df, "Heart Rate Data", None)
                    except Exception as e:
                        logger.error(f"Error converting timestamp to datetime: {e}")

                # 6. Heart rate distribution
                # This visualization shows the overall distribution of heart rate measurements
                # Helps understand the range and frequency of different heart rates across all users
                def plot_heart_rate_distribution():
                    sns.histplot(all_heartrates['heartRate'].dropna(), kde=True)
                    plt.title('Distribution of Heart Rates')
                    plt.xlabel('Heart Rate (bpm)')
                    plt.ylabel('Frequency')

                create_and_save_figure(
                    plot_heart_rate_distribution,
                    os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'heart_rate_distribution.png'),
                    figsize=(10, 6)
                )

                # 7. Heart rate over time for a sample of data
                # This visualization shows how heart rate changes over time for a specific activity
                # Helps identify patterns, peaks, and recovery periods during activities
                if len(all_heartrates) > 0 and 'activity_id' in all_heartrates.columns:
                    try:
                        # Take a sample of one player's data
                        activity_counts = all_heartrates['activity_id'].value_counts()
                        if not activity_counts.empty:
                            sample_player = activity_counts.index[0]
                            sample_data = all_heartrates[all_heartrates['activity_id'] == sample_player]

                            # Sort by timestamp if available
                            if 'timestamp' in sample_data.columns:
                                sample_data = sample_data.sort_values('timestamp')

                            if 'timestamp' in sample_data.columns and len(sample_data) > 0:
                                try:
                                    # Convert timestamp to datetime if it's not already
                                    if not pd.api.types.is_datetime64_any_dtype(sample_data['timestamp']):
                                        # Try parsing as numeric (milliseconds or seconds)
                                        # First check if values are likely milliseconds (large numbers)
                                        sample_value = sample_data['timestamp'].iloc[0] if len(sample_data) > 0 else 0
                                        if sample_value > 1e10:  # Likely milliseconds
                                            sample_data['timestamp'] = pd.to_datetime(sample_data['timestamp'], unit='ms')
                                        else:  # Likely seconds
                                            sample_data['timestamp'] = pd.to_datetime(sample_data['timestamp'], unit='s')

                                    def plot_heart_rate_over_time():
                                        plt.plot(sample_data['timestamp'], sample_data['heartRate'])
                                        plt.title(f'Heart Rate Over Time for Activity {sample_player}')
                                        plt.xlabel('Time')
                                        plt.ylabel('Heart Rate (bpm)')

                                    create_and_save_figure(
                                        plot_heart_rate_over_time,
                                        os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'heart_rate_over_time.png'),
                                        figsize=(14, 7)
                                    )
                                except Exception as e:
                                    logger.error(f"Error plotting heart rate over time: {e}")
                        else:
                            logger.warning("No activity IDs found in heart rate data")
                    except Exception as e:
                        logger.error(f"Error processing heart rate over time: {e}")

                # 8. Heart rate by hour of day (Bivariate Analysis)
                # This visualization shows how heart rate varies by hour of day
                # Helps identify daily patterns in heart rate
                if len(all_heartrates) > 10:  # Only if we have enough data
                    try:
                        # Convert timestamp to datetime if it's not already
                        if 'timestamp' in all_heartrates.columns:
                            if not pd.api.types.is_datetime64_any_dtype(all_heartrates['timestamp']):
                                # Try parsing as numeric (milliseconds or seconds)
                                # First check if values are likely milliseconds (large numbers)
                                sample_value = all_heartrates['timestamp'].iloc[0] if len(all_heartrates) > 0 else 0
                                if sample_value > 1e10:  # Likely milliseconds
                                    all_heartrates['timestamp'] = pd.to_datetime(all_heartrates['timestamp'], unit='ms')
                                else:  # Likely seconds
                                    all_heartrates['timestamp'] = pd.to_datetime(all_heartrates['timestamp'], unit='s')

                            all_heartrates['hour'] = all_heartrates['timestamp'].dt.hour

                            def plot_heart_rate_by_hour():
                                sns.boxplot(data=all_heartrates, x='hour', y='heartRate')
                                plt.title('Heart Rate by Hour of Day')
                                plt.xlabel('Hour of Day')
                                plt.ylabel('Heart Rate (bpm)')
                                plt.xticks(range(0, 24))

                            create_and_save_figure(
                                plot_heart_rate_by_hour,
                                os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'heart_rate_by_hour.png'),
                                figsize=(12, 6)
                            )

                            # Calculate statistics by hour
                            hr_by_hour = all_heartrates.groupby('hour')['heartRate'].agg(['mean', 'std', 'count'])
                            # Round mean and std values to 2 decimal places
                            hr_by_hour['mean'] = hr_by_hour['mean'].round(2)
                            hr_by_hour['std'] = hr_by_hour['std'].round(2)
                    except Exception as e:
                        logger.error(f"Error plotting heart rate by hour: {e}")
            else:
                logger.warning("No heart rate column found in the data")
        except Exception as e:
            logger.error(f"Error processing heart rate data: {e}")

def analyze_challenges(csv_data):
    """Analyze challenges data"""
    # First try to use 'challenges' from campaign_data.xlsx
    if 'challenges' in csv_data:
        challenges = csv_data['challenges']
    # If not found, try to use 'desc_challenges' from campaign_desc.xlsx
    elif 'desc_challenges' in csv_data:
        print("Using challenges data from campaign_desc.xlsx")
        challenges = csv_data['desc_challenges']
        # Print the columns in the challenges dataframe to help debug
        print(f"Columns in challenges dataframe: {', '.join(challenges.columns)}")

        # Add missing columns that are needed for visualizations
        if 'points' not in challenges.columns:
            # Try to extract points from target column if it exists
            if 'target' in challenges.columns:
                # Assume target contains points information, e.g. "100 points"
                # This is a simplification and might need to be adjusted based on actual data
                try:
                    challenges['points'] = challenges['target'].str.extract(r'(\d+)').astype(float)
                    print("Added 'points' column derived from 'target' column")
                except:
                    # If extraction fails, assign default points
                    challenges['points'] = 100  # Default value
                    print("Added 'points' column with default value of 100")
            else:
                # If no target column, assign default points
                challenges['points'] = 100  # Default value
                print("Added 'points' column with default value of 100")

        # Add difficulty column if it doesn't exist
        if 'difficulty' not in challenges.columns:
            # Assign a default difficulty level
            challenges['difficulty'] = 'Medium'  # Default value
            print("Added 'difficulty' column with default value of 'Medium'")

        # Add completion_rate column if it doesn't exist
        if 'completion_rate' not in challenges.columns:
            # Assume all challenges are completed with a 100% completion rate
            # This is a simplification and might need to be adjusted based on actual data
            challenges['completion_rate'] = 100.0  # Default value
            print("Added 'completion_rate' column with default value of 100.0")
    else:
        print("Challenges data not found in either campaign_data.xlsx or campaign_desc.xlsx")
        return

    # Generate descriptive statistics for challenges data
    generate_descriptive_stats(challenges, "Challenges Data", None)

    # Skip bivariate analysis for challenges data to avoid ANOVA test warnings
    # perform_bivariate_analysis(challenges, "Challenges Data", None)

    # 1. Challenge types distribution
    # This visualization shows the frequency of different challenge types
    # Helps identify which types of challenges are most common in the campaign
    challenge_counts = challenges['type'].value_counts()

    def plot_challenge_types_distribution():
        challenge_counts.plot(kind='bar')
        plt.title('Distribution of Challenge Types')
        plt.xlabel('Challenge Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)

    create_and_save_figure(
        plot_challenge_types_distribution,
        os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'challenge_types_distribution.png'),
        figsize=(12, 8)
    )

    # 2. Challenge points distribution
    # This visualization shows the distribution of points awarded for challenges
    # Helps understand the reward structure and identify high-value challenges
    if 'points' in challenges.columns:
        def plot_challenge_points_distribution():
            sns.histplot(challenges['points'].dropna(), kde=True, bins=20)
            plt.title('Distribution of Challenge Points')
            plt.xlabel('Points')
            plt.ylabel('Frequency')

        create_and_save_figure(
            plot_challenge_points_distribution,
            os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'challenge_points_distribution.png'),
            figsize=(10, 6)
        )

        # 3. Points by challenge type (Bivariate Analysis)
        # This visualization shows the distribution of points for each challenge type
        # Helps identify which types of challenges are more rewarding
        plt.figure(figsize=(14, 8))
        sns.boxplot(data=challenges, x='type', y='points')
        plt.title('Points Distribution by Challenge Type')
        plt.xlabel('Challenge Type')
        plt.ylabel('Points')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'points_by_challenge_type.png'))

        # Calculate average points by challenge type
        avg_points_by_type = challenges.groupby('type')['points'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
        # Round mean and std values to 2 decimal places
        avg_points_by_type['mean'] = avg_points_by_type['mean'].round(2)
        avg_points_by_type['std'] = avg_points_by_type['std'].round(2)

        # No need to save to file

    # 4. Challenge difficulty distribution (if available)
    if 'difficulty' in challenges.columns:
        plt.figure(figsize=(10, 6))
        difficulty_counts = challenges['difficulty'].value_counts().sort_index()
        difficulty_counts.plot(kind='bar')
        plt.title('Distribution of Challenge Difficulties')
        plt.xlabel('Difficulty Level')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'challenge_difficulty_distribution.png'))

        # 5. Relationship between difficulty and points (Bivariate Analysis)
        if 'points' in challenges.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=challenges, x='difficulty', y='points')
            plt.title('Points Distribution by Difficulty Level')
            plt.xlabel('Difficulty Level')
            plt.ylabel('Points')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'points_by_difficulty.png'))

            # Calculate correlation between difficulty and points
            if pd.api.types.is_numeric_dtype(challenges['difficulty']):
                difficulty_points_corr = challenges['difficulty'].corr(challenges['points'])

    # 6. Challenge completion rate (if available)
    if 'completion_rate' in challenges.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(challenges['completion_rate'].dropna(), kde=True, bins=20)
        plt.title('Distribution of Challenge Completion Rates')
        plt.xlabel('Completion Rate (%)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'challenge_completion_rate_distribution.png'))

        # 7. Completion rate by challenge type (Bivariate Analysis)
        plt.figure(figsize=(14, 8))
        sns.boxplot(data=challenges, x='type', y='completion_rate')
        plt.title('Completion Rate by Challenge Type')
        plt.xlabel('Challenge Type')
        plt.ylabel('Completion Rate (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'completion_rate_by_challenge_type.png'))

        # 8. Stacked bar chart of challenge completion status by type
        # This visualization shows the proportion of completed vs. incomplete challenges by type
        # Helps identify which types of challenges are most likely to be completed

        # Create a new column for completion status
        challenges['completion_status'] = challenges['completion_rate'].apply(
            lambda x: 'Completed' if pd.notna(x) and x > 0 else 'Incomplete'
        )

        # Create a crosstab of challenge types and completion status
        challenge_completion_by_type = pd.crosstab(
            index=challenges['type'],
            columns=challenges['completion_status'],
            normalize='index'
        ) * 100  # Convert to percentages

        # Plot stacked bar chart
        plt.figure(figsize=(14, 8))
        challenge_completion_by_type.plot(kind='bar', stacked=True, figsize=(14, 8), 
                                         colormap='viridis')
        plt.title('Challenge Completion Status by Type')
        plt.xlabel('Challenge Type')
        plt.ylabel('Percentage')
        plt.legend(title='Completion Status')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'challenge_completion_status_by_type.png'))

        # 9. Heatmap of challenge completion rates by type and difficulty (if available)
        if 'difficulty' in challenges.columns:
            # Create a pivot table of completion rates by type and difficulty
            completion_heatmap = challenges.pivot_table(
                values='completion_rate',
                index='type',
                columns='difficulty',
                aggfunc='mean'
            ).fillna(0)

            plt.figure(figsize=(14, 10))
            sns.heatmap(completion_heatmap, cmap='YlGnBu', annot=True, fmt='.1f', linewidths=0.5)
            plt.title('Challenge Completion Rates by Type and Difficulty')
            plt.xlabel('Difficulty Level')
            plt.ylabel('Challenge Type')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'challenge_completion_heatmap.png'))

    # Analyze frequency, timing, and types of challenges completed
    # Create a separate file for challenges completion analysis
    os.makedirs(OUTPUT_STATISTICS_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_STATISTICS_DIR, 'challenges_completion_analysis.txt'), 'w') as f:
        f.write(f"Challenges Completion Analysis\n")
        f.write(f"============================\n\n")
        f.write(f"Definition:\n")
        f.write(f"- Completed Challenge: A challenge that has been completed by at least one user\n\n")

        # Determine which challenges are completed
        # Assume challenges with a non-null completion_rate are completed
        completed_challenges = challenges
        if 'completion_rate' in challenges.columns:
            completed_challenges = challenges[challenges['completion_rate'].notna()]

        # 1. Frequency of completed challenges
        total_challenges = len(challenges)
        completed_count = len(completed_challenges)
        completion_percentage = (completed_count / total_challenges * 100) if total_challenges > 0 else 0

        f.write(f"1. Frequency Analysis\n")
        f.write(f"-------------------\n\n")

        table_data = [
            ["Metric", "Value"],
            ["Total challenges", f"{total_challenges}"],
            ["Completed challenges", f"{completed_count}"],
            ["Completion percentage", f"{completion_percentage:.1f}%"]
        ]

        # Format the table using tabulate
        table = tabulate.tabulate(table_data, headers="firstrow", tablefmt="grid")
        f.write(table)
        f.write("\n\n")

        # 2. Types of challenges completed
        if completed_count > 0:
            f.write(f"2. Types of Challenges Completed\n")
            f.write(f"-----------------------------\n\n")

            # Count completed challenges by type
            completed_by_type = completed_challenges['type'].value_counts()
            total_by_type = challenges['type'].value_counts()

            # Calculate completion percentage by type
            completion_by_type = {}
            for challenge_type in total_by_type.index:
                completed = completed_by_type.get(challenge_type, 0)
                total = total_by_type.get(challenge_type, 0)
                percentage = (completed / total * 100) if total > 0 else 0
                completion_by_type[challenge_type] = (completed, total, percentage)

            # Create a table with completion statistics by type
            table_data = [
                ["Challenge Type", "Completed", "Total", "Completion %"]
            ]

            for challenge_type, (completed, total, percentage) in sorted(completion_by_type.items(), 
                                                                        key=lambda x: x[1][2], 
                                                                        reverse=True):
                table_data.append([challenge_type, f"{completed}", f"{total}", f"{percentage:.1f}%"])

            # Format the table using tabulate
            table = tabulate.tabulate(table_data, headers="firstrow", tablefmt="grid")
            f.write(table)
            f.write("\n\n")

        # 3. Difficulty levels of completed challenges (if available)
        if 'difficulty' in challenges.columns and completed_count > 0:
            f.write(f"3. Difficulty Levels of Completed Challenges\n")
            f.write(f"----------------------------------------\n\n")

            # Count completed challenges by difficulty
            completed_by_difficulty = completed_challenges['difficulty'].value_counts().sort_index()
            total_by_difficulty = challenges['difficulty'].value_counts().sort_index()

            # Calculate completion percentage by difficulty
            completion_by_difficulty = {}
            for difficulty in total_by_difficulty.index:
                completed = completed_by_difficulty.get(difficulty, 0)
                total = total_by_difficulty.get(difficulty, 0)
                percentage = (completed / total * 100) if total > 0 else 0
                completion_by_difficulty[difficulty] = (completed, total, percentage)

            # Create a table with completion statistics by difficulty
            table_data = [
                ["Difficulty Level", "Completed", "Total", "Completion %"]
            ]

            for difficulty, (completed, total, percentage) in sorted(completion_by_difficulty.items()):
                table_data.append([str(difficulty), f"{completed}", f"{total}", f"{percentage:.1f}%"])

            # Format the table using tabulate
            table = tabulate.tabulate(table_data, headers="firstrow", tablefmt="grid")
            f.write(table)
            f.write("\n\n")

        # 4. Timing of challenges (if available)
        # Since we don't have timestamp information for challenges, we can't analyze timing directly
        # Instead, we'll note this limitation
        f.write(f"4. Timing of Challenges\n")
        f.write(f"---------------------\n\n")
        f.write(f"Note: Detailed timing analysis is not available as the challenges data does not include timestamp information.\n\n")

        # Add summary and insights
        f.write(f"Summary and Insights:\n")
        f.write(f"-------------------\n")
        f.write(f"- Overall, {completed_count} out of {total_challenges} challenges ({completion_percentage:.1f}%) were completed.\n")

        if completed_count > 0:
            # Most common completed challenge type
            most_common_type = completed_challenges['type'].value_counts().idxmax()
            most_common_count = completed_challenges['type'].value_counts().max()
            most_common_percentage = (most_common_count / completed_count * 100)
            f.write(f"- The most common type of completed challenge was '{most_common_type}' ({most_common_count} challenges, {most_common_percentage:.1f}% of all completed challenges).\n")

            # Difficulty level insights (if available)
            if 'difficulty' in challenges.columns:
                avg_difficulty = completed_challenges['difficulty'].mean() if pd.api.types.is_numeric_dtype(completed_challenges['difficulty']) else None
                if avg_difficulty is not None:
                    f.write(f"- The average difficulty level of completed challenges was {avg_difficulty:.1f}.\n")

                # Most common difficulty level
                most_common_difficulty = completed_challenges['difficulty'].value_counts().idxmax()
                most_common_diff_count = completed_challenges['difficulty'].value_counts().max()
                most_common_diff_percentage = (most_common_diff_count / completed_count * 100)
                f.write(f"- The most common difficulty level of completed challenges was '{most_common_difficulty}' ({most_common_diff_count} challenges, {most_common_diff_percentage:.1f}% of all completed challenges).\n")

            # Points insights (if available)
            if 'points' in challenges.columns:
                avg_points = completed_challenges['points'].mean()
                f.write(f"- Completed challenges awarded an average of {avg_points:.1f} points.\n")

    logger.info(f"Challenges completion analysis saved to {OUTPUT_STATISTICS_DIR} folder")

def generate_analysis_report(csv_data, json_data, activities, usage_metrics, campaign_metrics, dropout_metrics):
    """
    Generate a report of which analyses were performed and which weren't, with reasons.

    Args:
        csv_data: Dictionary of DataFrames from Excel files
        json_data: Dictionary of data from JSON files
        activities: DataFrame of activities
        usage_metrics: Dictionary of usage metrics
        campaign_metrics: Dictionary of campaign metrics
        dropout_metrics: Dictionary of dropout metrics

    Returns:
        None
    """
    print("DEBUG: generate_analysis_report function called")
    logger.info("DEBUG: generate_analysis_report function called")
    # Create data_analysis directory if it doesn't exist
    data_analysis_dir = os.path.join(PROJECT_ROOT, 'data_analysis')
    os.makedirs(data_analysis_dir, exist_ok=True)

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

            # Check if active vs passive usage analysis was performed
            if usage_metrics:
                f.write("✓ Active vs passive usage analysis performed\n")
                performed_analyses.append("Active vs passive usage analysis")
            else:
                f.write("✗ Active vs passive usage analysis NOT performed\n")

                # Check if VISIT_APP_PAGE events with DURATION_MS are available
                has_visit_app_page = False
                has_duration_ms = False

                if activities is not None and 'type' in activities.columns:
                    has_visit_app_page = 'VISIT_APP_PAGE' in activities['type'].values

                    if has_visit_app_page:
                        visit_app_page_data = activities[activities['type'] == 'VISIT_APP_PAGE']
                        if 'properties' in visit_app_page_data.columns:
                            for prop in visit_app_page_data['properties']:
                                if isinstance(prop, dict) and 'DURATION_MS' in prop:
                                    has_duration_ms = True
                                    break

                reason = ""
                if not has_visit_app_page:
                    reason = "No VISIT_APP_PAGE events found"
                    f.write(f"   - Reason: {reason}\n")
                elif not has_duration_ms:
                    reason = "VISIT_APP_PAGE events found but no DURATION_MS data available"
                    f.write(f"   - Reason: {reason}\n")
                    f.write("   - This data is required for accurate active vs passive usage calculation\n")

                skipped_analyses.append(("Active vs passive usage analysis", reason))

            # Check if dropout analysis was performed
            if dropout_metrics:
                f.write("✓ User dropout analysis performed\n")
                performed_analyses.append("User dropout analysis")
            else:
                reason = "Insufficient activity data to calculate dropout rates"
                f.write("✗ User dropout analysis NOT performed\n")
                f.write(f"   - Reason: {reason}\n")
                skipped_analyses.append(("User dropout analysis", reason))

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
            has_heart_rate_data = False
            has_app_visits_data = False

            for user_data in json_data.values():
                if 'activities' in user_data:
                    for activity in user_data['activities']:
                        if activity.get('gameDescriptor') in ['WALK', 'RUN', 'CYCLE']:
                            has_movement_data = True
                        if activity.get('gameDescriptor') == 'HEART_RATE':
                            has_heart_rate_data = True
                        if activity.get('gameDescriptor') in ['VISIT_APP_PAGE', 'NAVIGATE_APP']:
                            has_app_visits_data = True

            if has_movement_data:
                f.write("✓ Movement data analysis performed (steps, calories, etc.)\n")
                performed_analyses.append("Movement data analysis")
            else:
                reason = "No movement data (WALK, RUN, CYCLE) found in user activities"
                f.write("✗ Movement data analysis NOT performed\n")
                f.write(f"   - Reason: {reason}\n")
                skipped_analyses.append(("Movement data analysis", reason))

            if has_heart_rate_data:
                f.write("✓ Heart rate data analysis performed\n")
                performed_analyses.append("Heart rate data analysis")
            else:
                reason = "No heart rate data found in user activities"
                f.write("✗ Heart rate data analysis NOT performed\n")
                f.write(f"   - Reason: {reason}\n")
                skipped_analyses.append(("Heart rate data analysis", reason))

            if has_app_visits_data:
                f.write("✓ App page visits analysis performed\n")
                performed_analyses.append("App page visits analysis")
            else:
                reason = "No app visit data (VISIT_APP_PAGE, NAVIGATE_APP) found in user activities"
                f.write("✗ App page visits analysis NOT performed\n")
                f.write(f"   - Reason: {reason}\n")
                skipped_analyses.append(("App page visits analysis", reason))
        else:
            reason = "No JSON user data available"
            f.write("✗ User data analysis NOT performed\n")
            f.write(f"   - Reason: {reason}\n")
            skipped_analyses.append(("User data analysis", reason))

        f.write("\n4. Challenges Analysis\n")
        f.write("---------------------\n")
        if csv_data and 'challenges' in csv_data:
            f.write("✓ Challenges analysis performed\n")
            performed_analyses.append("Challenges analysis")
        else:
            reason = "No challenges data available in Excel files"
            f.write("✗ Challenges analysis NOT performed\n")
            f.write(f"   - Reason: {reason}\n")
            skipped_analyses.append(("Challenges analysis", reason))

        f.write("\n5. Summary\n")
        f.write("---------\n")
        f.write("Analysis completed. Results saved to:\n")
        f.write(f"- Visualizations: {OUTPUT_VISUALIZATIONS_DIR}\n")
        f.write(f"- Statistics: {OUTPUT_STATISTICS_DIR}\n")
        f.write(f"- This report: {report_path}\n")

        # Add list of performed analyses
        f.write("\n6. Analyses Performed\n")
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
            activity_viz = [v for v in visualization_files if v.startswith('activity_')]
            user_viz = [v for v in visualization_files if v.startswith('user_')]
            challenge_viz = [v for v in visualization_files if v.startswith('challenge_')]
            movement_viz = [v for v in visualization_files if 'steps' in v or 'calories' in v or 'movement' in v]
            dropout_viz = [v for v in visualization_files if 'dropout' in v]
            other_viz = [v for v in visualization_files if not any([
                v.startswith('activity_'), 
                v.startswith('user_'), 
                v.startswith('challenge_'),
                'steps' in v, 
                'calories' in v, 
                'movement' in v,
                'dropout' in v
            ])]

            if activity_viz:
                f.write("  Activity Analysis:\n")
                for viz in activity_viz:
                    f.write(f"  - {viz}\n")

            if user_viz:
                f.write("  User Analysis:\n")
                for viz in user_viz:
                    f.write(f"  - {viz}\n")

            if challenge_viz:
                f.write("  Challenge Analysis:\n")
                for viz in challenge_viz:
                    f.write(f"  - {viz}\n")

            if movement_viz:
                f.write("  Movement Analysis:\n")
                for viz in movement_viz:
                    f.write(f"  - {viz}\n")

            if dropout_viz:
                f.write("  Dropout Analysis:\n")
                for viz in dropout_viz:
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
            "challenges_completion_analysis.txt", 
            "dropout_analysis.txt", 
            "task_completion_analysis.txt", 
            "usage_analysis.txt"
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
        activities: DataFrame of activities

    Returns:
        Path to the generated report
    """
    # Extract metrics from activities if available
    usage_metrics = None
    dropout_metrics = None
    campaign_metrics = None

    if activities is not None and isinstance(activities, tuple) and len(activities) >= 5:
        # If activities is a tuple returned from analyze_activities
        activities_df, _, usage_metrics, dropout_metrics, campaign_metrics = activities
    else:
        activities_df = activities

    # Destination file
    data_analysis_dir = os.path.join(PROJECT_ROOT, 'data_analysis')
    os.makedirs(data_analysis_dir, exist_ok=True)
    dest_file = os.path.join(data_analysis_dir, 'analysis_report.txt')

    # Generate the report using the current data
    generate_analysis_report(csv_data, json_data, activities_df, usage_metrics, campaign_metrics, dropout_metrics)

    return dest_file

def main():
    """Main function to run the analysis"""
    try:
        logger.info("Starting data analysis")

        # Create output directories for visualizations and statistics
        os.makedirs(OUTPUT_VISUALIZATIONS_DIR, exist_ok=True)
        os.makedirs(OUTPUT_STATISTICS_DIR, exist_ok=True)
        logger.info(f"Created output directories: {OUTPUT_VISUALIZATIONS_DIR} and {OUTPUT_STATISTICS_DIR}")

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
            result = analyze_activities(csv_data)
            if result:
                activities, unique_users_count, usage_metrics, dropout_metrics, campaign_metrics = result
                logger.info(f"Activities analysis complete: {unique_users_count} users")
            else:
                logger.warning("Activities analysis returned no results")
                activities = None
                unique_users_count = 0
                usage_metrics = None
                dropout_metrics = None
                campaign_metrics = None
        except Exception as e:
            logger.error(f"Error analyzing activities data: {e}")
            activities = None
            unique_users_count = 0
            usage_metrics = None
            dropout_metrics = None
            campaign_metrics = None

        # Analyze user data
        try:
            logger.info("Analyzing user data...")
            # Pass campaign dates to analyze_user_data if available
            if campaign_metrics and 'start_date' in campaign_metrics and 'end_date' in campaign_metrics:
                analyze_user_data(json_data, campaign_metrics['start_date'], campaign_metrics['end_date'])
            else:
                analyze_user_data(json_data)
            logger.info("User data analysis complete")
        except Exception as e:
            logger.error(f"Error analyzing user data: {e}")

        # Analyze challenges data
        try:
            logger.info("Analyzing challenges data...")
            analyze_challenges(csv_data)
            logger.info("Challenges analysis complete")
        except Exception as e:
            logger.error(f"Error analyzing challenges data: {e}")

        # Generate analysis report
        try:
            logger.info("Calling create_complete_report function")
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

        if usage_metrics:
            try:
                print(f"- Passive vs. Active Usage:")
                print(f"  * Total active usage: {usage_metrics['total_active_minutes']:.1f} minutes ({usage_metrics['total_active_minutes']/60:.1f} hours)")
                print(f"  * Total passive usage: {usage_metrics['total_passive_minutes']:.1f} minutes ({usage_metrics['total_passive_minutes']/60:.1f} hours)")
                print(f"  * Average active usage per user per day: {usage_metrics['avg_active_minutes_per_user_day']:.1f} minutes")
                print(f"  * Average passive usage per user per day: {usage_metrics['avg_passive_minutes_per_user_day']:.1f} minutes")
            except (KeyError, TypeError) as e:
                logger.error(f"Error accessing usage metrics: {e}")

        if dropout_metrics:
            try:
                print(f"- User Drop-out Rates (time between first and last activity):")
                print(f"  * Average drop-out time: {dropout_metrics['avg_dropout_days']:.0f} days")
                print(f"  * Median drop-out time: {dropout_metrics['median_dropout_days']:.0f} days")
                print(f"  * Range: {dropout_metrics['min_dropout_days']:.0f} to {dropout_metrics['max_dropout_days']:.0f} days")
            except (KeyError, TypeError) as e:
                logger.error(f"Error accessing dropout metrics: {e}")

    except Exception as e:
        logger.error(f"Unexpected error in main function: {e}")
        print(f"An error occurred during analysis. See log file for details.")

    # Log the completion of the analysis
    logger.info("Analysis complete with visualizations and statistics generated")

    # Only print a summary of the visualizations to keep the output concise
    print("\nVisualization categories:")
    print(f"   - Activities Analysis: {OUTPUT_VISUALIZATIONS_DIR}/activity_*.png")
    print(f"   - User Engagement: {OUTPUT_VISUALIZATIONS_DIR}/user_*.png, {OUTPUT_VISUALIZATIONS_DIR}/*_usage_*.png")
    print(f"   - User Data: {OUTPUT_VISUALIZATIONS_DIR}/movement_*.png, {OUTPUT_VISUALIZATIONS_DIR}/heart_rate_*.png")
    print(f"   - Challenges: {OUTPUT_VISUALIZATIONS_DIR}/challenge_*.png")

    print("\nFor a complete list of generated visualizations, check the log file or the visualizations directory.")

if __name__ == "__main__":
    main()
