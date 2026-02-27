"""
Main pipeline runner for the GameBus-HealthBehaviorMining project.
"""
import argparse
import pandas as pd
import logging
import concurrent.futures
from typing import List, Dict, Any

from config.credentials import require_authcode
from src.analysis.data_analysis import main as run_analysis
from src.scripts.create_user_email_mapping import main as build_user_email_mapping
from src.utils.logging import setup_logging, console_info, console_error

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

    username = user_row["email"]
    password = user_row["password"]

    logger = logging.getLogger(__name__)
    console_info(logger, f"[EXTRACT] {username}: starting")

    client = GameBusClient(require_authcode())

    token = client.get_user_token(username, password)
    if not token:
        logger.warning(f"Failed to get token for user {username}")
        console_info(logger, f"[EXTRACT] {username}: failed (authentication)")
        return {}

    user_id_result = client.get_user_id(token)
    if not user_id_result:
        logger.warning(f"Failed to get player ID for user {username}")
        console_info(logger, f"[EXTRACT] {username}: failed (player id lookup)")
        return {}

    user_id, user_email = user_id_result
    logger.info(f"Successfully authenticated user {username} with player ID {user_id}")
    console_info(logger, f"[EXTRACT] {username}: authenticated")

    all_collector = AllDataCollector(client, token, user_id, user_email)

    try:
        start = time.time()
        data_dict, file_paths = all_collector.collect()
        elapsed = time.time() - start
        logger.info(f"Finished collection for user {username} in {elapsed:.1f}s")
    except Exception:
        logger.warning(f"Failed to collect ALL data for user {username}")
        logger.info(f"Traceback for collection failure: {username}", exc_info=True)
        console_info(logger, f"[EXTRACT] {username}: failed during collection")
        return {}

    if data_dict:
        logger.info(
            f"Collected data for {len(data_dict)} data types, saved to {len(file_paths)} files"
        )
        for file_path in file_paths:
            logger.info(f"  - {file_path}")

        console_info(
            logger,
            f"[EXTRACT] {username}: done | {len(data_dict)} data types | {len(file_paths)} files | {elapsed:.1f}s",
        )
        return data_dict

    logger.warning(f"No data collected for user {username}")
    console_info(logger, f"[EXTRACT] {username}: no data returned | {elapsed:.1f}s")
    return {}


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
    console_info(
        logger,
        f"[PIPELINE] Start | extraction={should_run_extraction} | analysis={should_run_analysis}",
    )

    # If only analysis is being run, optionally check for raw JSON but do not block analysis
    if should_run_analysis and not should_run_extraction:
        import os
        import glob
        from config.paths import RAW_DATA_DIR

        if not os.path.exists(RAW_DATA_DIR):
            logger.warning(
                f"Data directory {RAW_DATA_DIR} does not exist. Proceeding with analysis using Excel data only."
            )
            console_info(logger, "[ANALYZE] No raw JSON directory found; continuing with Excel-only analysis")
        else:
            data_files = glob.glob(f"{RAW_DATA_DIR}/*.json")
            if not data_files:
                logger.warning(
                    f"No JSON data files found in {RAW_DATA_DIR}. Proceeding with analysis using Excel data only."
                )
                console_info(logger, "[ANALYZE] No raw JSON files found; continuing with Excel-only analysis")

    # Run extraction if needed
    if should_run_extraction:
        users_df = load_users(USERS_FILE_PATH)
        logger.info(f"Loaded {len(users_df)} users from {USERS_FILE_PATH}")
        console_info(logger, f"[EXTRACT] Loaded {len(users_df)} user(s)")

        all_results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        total_users = len(users_df)

        if users_df.empty:
            logger.warning("No users found in the users file. Skipping extraction.")
            console_info(logger, "[EXTRACT] No users found; skipping extraction")
        else:
            max_workers = min(4, total_users)  # keep 1 if you intentionally want sequential extraction
            console_info(logger, f"[EXTRACT] Starting extraction for {total_users} user(s)")

            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_user = {
                        executor.submit(run_extraction, user_row): user_row["email"]
                        for _, user_row in users_df.iterrows()
                    }

                    completed = 0
                    total = len(future_to_user)

                    try:
                        for future in concurrent.futures.as_completed(future_to_user):
                            user_email = future_to_user[future]
                            completed += 1

                            try:
                                user_results = future.result()
                            except Exception:
                                logger.warning(f"Error processing user {user_email}")
                                logger.info(f"Traceback for future failure: {user_email}", exc_info=True)
                                user_results = {}

                            all_results[user_email] = user_results
                            successful_so_far = sum(1 for _email, res in all_results.items() if bool(res))

                            logger.info(f"Completed user {completed}/{total}: {user_email}")
                            console_info(
                                logger,
                                f"[EXTRACT] Progress {completed}/{total} | successful={successful_so_far}",
                            )

                    except KeyboardInterrupt:
                        logger.warning("KeyboardInterrupt received. Cancelling pending tasks...")
                        for future in future_to_user:
                            if not future.done():
                                future.cancel()
                        raise

            except KeyboardInterrupt:
                logger.warning("Data extraction interrupted by user. Partial results may have been saved.")
                console_info(logger, "[EXTRACT] Interrupted; partial results may have been saved")
                if not should_run_analysis:
                    return

        if total_users > 0:
            try:
                logger.info("Generating user-email mapping file after extraction...")
                build_user_email_mapping()
                logger.info("User-email mapping file generated successfully")
                console_info(logger, "[EXTRACT] User-email mapping file generated")
            except Exception:
                logger.warning("Failed to generate user-email mapping file")
                logger.info("Traceback for user-email mapping failure", exc_info=True)

            try:
                processed_users = len(all_results)
                successful_users = sum(1 for _email, res in all_results.items() if bool(res))

                logger.info(
                    f"Extraction summary: {successful_users}/{total_users} users with data collected "
                    f"({processed_users} processed)."
                )
                console_info(
                    logger,
                    f"[EXTRACT] Summary: {successful_users}/{total_users} users successful ({processed_users} processed)",
                )
            except Exception:
                logger.warning("Could not compute extraction summary")
                logger.info("Traceback for extraction summary failure", exc_info=True)

    # Run data analysis if needed
    if should_run_analysis:
        analysis_logger = setup_logging(log_level=args.log_level, log_type="analysis")
        console_info(analysis_logger, "[ANALYZE] Starting analysis")
        try:
            run_analysis()
            console_info(analysis_logger, "[ANALYZE] Finished")
        except Exception:
            analysis_logger.error("Unhandled error during analysis", exc_info=True)
            console_error(analysis_logger, "[ANALYZE] Failed; see log file")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger = logging.getLogger(__name__)
        console_info(logger, "[PIPELINE] Interrupted by user")
    except Exception:
        logger = logging.getLogger(__name__)
        logger.error("Unhandled pipeline error", exc_info=True)
        console_error(logger, "[PIPELINE] Failed; see log file")
