"""
Main pipeline runner for the GameBus-HealthBehaviorMining project.
"""
import argparse
import pandas as pd
import logging
import concurrent.futures
from typing import List, Dict, Any

from config.credentials import AUTHCODE
from config.paths import USERS_FILE_PATH
from src.extraction.gamebus_client import GameBusClient
from src.extraction.data_collectors import (
    LocationDataCollector, 
    MoodDataCollector,
    ActivityTypeDataCollector,
    HeartRateDataCollector,
    AccelerometerDataCollector,
    NotificationDataCollector
)
from src.utils.logging import setup_logging

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='GameBus Health Behavior Mining Pipeline')

    parser.add_argument('--user-id', type=int, 
                        help='Specific user ID to process')
    parser.add_argument('--data-types', nargs='+', default=['all'],
                        choices=['all', 'location', 'mood', 'activity', 'heartrate', 'accelerometer', 'notification'],
                        help='Data types to collect')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    parser.add_argument('--users-file', type=str,
                        help='Path to the XLSX file containing user credentials')

    return parser.parse_args()

def load_users(users_file: str) -> pd.DataFrame:
    """
    Load users from an XLSX file.

    Args:
        users_file: Path to users XLSX file

    Returns:
        DataFrame of users
    """
    try:
        df = pd.read_excel(users_file)
        # Ensure the required columns are present
        if 'email' in df.columns and 'password' in df.columns:
            # Include UserID column if it exists, otherwise just use email and password
            columns_to_keep = ['email', 'password']
            if 'UserID' in df.columns:
                columns_to_keep.append('UserID')
            return df[columns_to_keep]
        else:
            logging.error("Users file must contain 'email' and 'password' columns")
            raise ValueError("Users file must contain 'email' and 'password' columns")
    except Exception as e:
        logging.error(f"Failed to load users file: {e}")
        raise

def run_extraction(user_row: pd.Series, data_types: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run the extraction step for a single user.

    Args:
        user_row: User data row from DataFrame
        data_types: Types of data to collect

    Returns:
        Dictionary of collected data by type
    """
    username = user_row['email']
    password = user_row['password']

    logger = logging.getLogger(__name__)
    logger.info(f"Starting extraction for user: {username}")

    # Initialize GameBus client
    logger.info(f"Initializing GameBus client for user: {username}")
    client = GameBusClient(AUTHCODE)

    # Get player token and ID
    logger.info(f"Getting player token for user: {username}")
    token = client.get_player_token(username, password)
    if not token:
        logger.error(f"Failed to get token for user {username}")
        return {}

    logger.info(f"Getting player ID for user: {username}")
    player_id = client.get_player_id(token)
    if not player_id:
        logger.error(f"Failed to get player ID for user {username}")
        return {}

    logger.info(f"Successfully authenticated user {username} with player ID {player_id}")

    # Collect data based on requested types
    results = {}

    collectors = {
        'location': LocationDataCollector(client, token, player_id),
        'mood': MoodDataCollector(client, token, player_id),
        'activity': ActivityTypeDataCollector(client, token, player_id),
        'heartrate': HeartRateDataCollector(client, token, player_id),
        'accelerometer': AccelerometerDataCollector(client, token, player_id),
        'notification': NotificationDataCollector(client, token, player_id)
    }

    types_to_collect = list(collectors.keys()) if 'all' in data_types else data_types
    logger.info(f"Will collect these data types for user {username}: {types_to_collect}")

    for data_type in types_to_collect:
        if data_type in collectors:
            logger.info(f"Starting collection of {data_type} data for user {username}")
            try:
                # Set a timeout for each collector to prevent hanging
                import threading
                import time

                # Flag to indicate if collection is complete
                collection_complete = False
                collection_result = [None, None]  # [data, file_path]
                collection_error = [None]  # Error message

                def collect_data():
                    try:
                        data, file_path = collectors[data_type].collect()
                        collection_result[0] = data
                        collection_result[1] = file_path
                        nonlocal collection_complete
                        collection_complete = True
                    except Exception as e:
                        collection_error[0] = str(e)
                        collection_complete = True

                # Start collection in a separate thread
                collection_thread = threading.Thread(target=collect_data)
                collection_thread.daemon = True
                collection_thread.start()

                # Wait for collection to complete or timeout
                start_time = time.time()
                timeout = 120  # 2 minutes timeout per collector

                while not collection_complete and (time.time() - start_time) < timeout:
                    logger.info(f"Waiting for {data_type} data collection to complete for user {username}...")
                    time.sleep(10)  # Check every 10 seconds

                if not collection_complete:
                    logger.error(f"Timeout while collecting {data_type} data for user {username}")
                    continue

                if collection_error[0]:
                    logger.error(f"Failed to collect {data_type} data for user {username}: {collection_error[0]}")
                    continue

                data, file_path = collection_result

                if data:
                    results[data_type] = data
                    if file_path:
                        logger.info(f"Collected {len(data)} {data_type} data points, saved to {file_path}")
                    else:
                        logger.warning(f"Collected {len(data)} {data_type} data points, but no file was created")
                else:
                    logger.warning(f"No {data_type} data collected for user {username}")
            except Exception as e:
                logger.error(f"Failed to collect {data_type} data for user {username}: {e}")

    logger.info(f"Completed extraction for user: {username}")
    return results

def main():
    """Main function to run the pipeline."""
    args = parse_args()

    # Set up logging
    logger = setup_logging(log_level=args.log_level)
    logger.info("Starting GameBus Health Behavior Mining Pipeline")

    # Load users
    users_file = args.users_file if args.users_file else USERS_FILE_PATH
    users_df = load_users(users_file)
    logger.info(f"Loaded {len(users_df)} users from {users_file}")

    # Process specific user or all users
    if args.user_id:
        user_row = users_df[users_df['UserID'] == args.user_id].iloc[0]
        results = run_extraction(user_row, args.data_types)
    else:
        all_results = {}

        # Use a thread pool to process users in parallel
        max_workers = min(10, len(users_df))  # Limit the number of concurrent workers
        logger.info(f"Using {max_workers} workers for parallel processing")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_user = {
                executor.submit(run_extraction, user_row, args.data_types): user_row['email']
                for _, user_row in users_df.iterrows()
            }

            # Process results as they complete
            completed = 0
            total = len(future_to_user)

            for future in concurrent.futures.as_completed(future_to_user):
                user_email = future_to_user[future]
                completed += 1

                try:
                    user_results = future.result()
                    all_results[user_email] = user_results
                    logger.info(f"Completed user {completed}/{total}: {user_email}")
                except Exception as e:
                    logger.error(f"Error processing user {user_email}: {e}")

                # Report progress
                progress_pct = (completed / total) * 100
                logger.info(f"Progress: {progress_pct:.1f}% ({completed}/{total} users)")

    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    main() 
