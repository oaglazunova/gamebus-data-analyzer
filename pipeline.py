"""
Main pipeline runner for the GameBus-HealthBehaviorMining project.
"""
import argparse
import pandas as pd
import logging
import concurrent.futures
from typing import List, Dict, Any

from config.credentials import require_authcode
from src.utils.logging import setup_logging
from src.analysis.data_analysis import main as run_analysis
from src.scripts.create_user_email_mapping import main as build_user_email_mapping

from config.paths import USERS_FILE_PATH

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='GameBus Health Behavior Mining Pipeline')

    # Add mutually exclusive group for extract and analyze
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--extract', action='store_true',
                      help='Only extract all data for all users')
    group.add_argument('--analyze', action='store_true',
                      help='Only analyze extracted data')

    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')

    args = parser.parse_args()


    return args

def load_users(users_file: str) -> pd.DataFrame:
    """
    Load users from an XLSX file.

    Args:
        users_file: Path to users XLSX file

    Returns:
        DataFrame of users
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading users from {users_file}")

    try:
        df = pd.read_excel(users_file)
        logger.info(f"Loaded Excel file with columns: {', '.join(df.columns)}")

        # Ensure the required columns are present
        if 'email' in df.columns and 'password' in df.columns:
            columns_to_keep = ['email', 'password']
            logger.info(f"Using columns: {', '.join(columns_to_keep)}")
            return df[columns_to_keep]
        else:
            logger.error(f"Users file must contain 'email' and 'password' columns. Found columns: {', '.join(df.columns)}")
            raise ValueError("Users file must contain 'email' and 'password' columns")
    except Exception as e:
        logger.error(f"Failed to load users file: {e}")
        raise

def run_extraction(user_row: pd.Series) -> Dict[str, List[Dict[str, Any]]]:
    from src.extraction.gamebus_client import GameBusClient
    from src.extraction.data_collectors import AllDataCollector
    import time

    username = user_row['email']
    password = user_row['password']

    logger = logging.getLogger(__name__)
    logger.info(f"Starting extraction for user: {username}")

    client = GameBusClient(require_authcode())

    token = client.get_user_token(username, password)
    if not token:
        logger.error(f"Failed to get token for user {username}")
        return {}

    user_id_result = client.get_user_id(token)
    if not user_id_result:
        logger.error(f"Failed to get player ID for user {username}")
        return {}

    user_id, user_email = user_id_result
    logger.info(f"Successfully authenticated user {username} with player ID {user_id}")

    results = {}
    all_collector = AllDataCollector(client, token, user_id, user_email)

    try:
        start = time.time()
        logger.info(f"Starting collection of ALL data types for user {username}")
        data_dict, file_paths = all_collector.collect()
        elapsed = time.time() - start
        logger.info(f"Finished collection for user {username} in {elapsed:.1f}s")
    except Exception as e:
        logger.error(f"Failed to collect ALL data for user {username}: {e}")
        logger.exception(e)
        return results

    if data_dict:
        results = data_dict
        if file_paths:
            logger.info(f"Collected data for {len(data_dict)} data types, saved to {len(file_paths)} files")
            for file_path in file_paths:
                logger.info(f"  - {file_path}")
        else:
            logger.warning(f"Collected data for {len(data_dict)} data types, but no files were created")
        logger.info(f"Successful extraction for user: {username}")
    else:
        logger.warning(f"No data collected for user {username}")

    logger.info(f"Completed extraction for user: {username}")
    return results


def main():
    """Main function to run the pipeline."""
    args = parse_args()

    # Set up logging for extraction
    logger = setup_logging(log_level=args.log_level, log_type="extraction")
    logger.info("Starting GameBus Health Behavior Mining Pipeline")
    logger.info(f"Parsed arguments: extract={args.extract}, analyze={args.analyze}, log_level={args.log_level}")

    # Determine which steps to run based on command-line arguments
    # If neither --extract nor --analyze is specified, run both
    # If --extract is specified, only run extraction
    # If --analyze is specified, only run analysis
    if not args.extract and not args.analyze:
        # No specific flag provided, run both extraction and analysis
        should_run_extraction = True
        should_run_analysis = True
    else:
        # Specific flag provided, follow the flag
        should_run_extraction = args.extract
        should_run_analysis = args.analyze

    logger.info(f"Run extraction: {should_run_extraction}, Run analysis: {should_run_analysis}")

    # If only analysis is being run (--analyze flag is specified), optionally check for data_raw JSON but do not block analysis
    if should_run_analysis and not should_run_extraction:
        import os
        import glob
        from config.paths import RAW_DATA_DIR

        # Check if data_raw folder exists
        if not os.path.exists(RAW_DATA_DIR):
            logger.warning(f"Data directory {RAW_DATA_DIR} does not exist. Proceeding with analysis using Excel data only.")
        else:
            # Check if data_raw folder contains any data files
            data_files = glob.glob(f'{RAW_DATA_DIR}/*.json')
            if not data_files:
                logger.warning(f"No JSON data files found in {RAW_DATA_DIR}. Proceeding with analysis using Excel data only.")

    # Run extraction if needed
    if should_run_extraction:
        # Load users
        users_df = load_users(USERS_FILE_PATH)
        logger.info(f"Loaded {len(users_df)} users from {USERS_FILE_PATH}")

        if users_df.empty:
            logger.warning("No users found in the users file. Skipping extraction.")
            if not should_run_analysis:
                return
        else:
            max_workers = min(1, len(users_df))

        # Process all users
        all_results = {}

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_user = {
                    executor.submit(run_extraction, user_row): user_row['email']
                    for _, user_row in users_df.iterrows()
                }

                # Process results as they complete
                completed = 0
                total = len(future_to_user)

                try:
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
                except KeyboardInterrupt:
                    logger.warning("KeyboardInterrupt received. Cancelling pending tasks...")
                    # Cancel all pending futures
                    for future in future_to_user:
                        if not future.done():
                            future.cancel()
                    # Wait for the remaining tasks to complete
                    logger.warning("Waiting for running tasks to complete...")
                    # We don't wait for futures here to allow immediate interruption
                    raise  # Re-raise to be caught by the outer try-except

        except KeyboardInterrupt:
            logger.warning("Data extraction interrupted by user. Partial results may have been saved.")
            # Continue with analysis if requested, using partial results
            if should_run_analysis:
                logger.info("Continuing with analysis using partial results...")
            else:
                return

        # After successful extraction phase, generate the user-email mapping file
        try:
            logger.info("Generating user-email mapping file after extraction...")
            build_user_email_mapping()
            logger.info("User-email mapping file generated successfully")
        except Exception as e:
            logger.error(f"Failed to generate user-email mapping file: {e}")

        # Console summary for extraction success
        try:
            total_users = len(users_df)
            processed_users = len(all_results)
            successful_users = sum(1 for _email, res in all_results.items() if bool(res))
            logger.info(
                f"Extraction summary: {successful_users}/{total_users} users with data collected "
                f"({processed_users} processed)."
            )
            logger.info("Data extraction phase completed successfully")
        except Exception as e:
            logger.warning(f"Could not compute extraction summary: {e}")

    # Run data analysis if needed
    if should_run_analysis:
        logger.info("Starting data analysis...")
        setup_logging(log_level=args.log_level, log_type="analysis")
        run_analysis()
        logger.info("Data analysis completed")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("Program interrupted by user. Exiting gracefully...")
