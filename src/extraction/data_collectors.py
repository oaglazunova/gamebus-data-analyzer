"""
Data collectors for extracting user data from the GameBus API.
"""
import json
import os
import logging
import datetime
from typing import Dict, List, Tuple, Any, Optional

from config.paths import RAW_DATA_DIR
from config.settings import VALID_GAME_DESCRIPTORS

# Set up logging
logger = logging.getLogger(__name__)

class AllDataCollector:
    """
    Collector for all types of data.
    """
    def __init__(self, client, token: str, user_id: int):
        """
        Initialize the all data collector.

        Args:
            client: GameBusClient instance
            token: User access token
            user_id: User ID
        """
        self.client = client
        self.token = token
        self.user_id = user_id

        # Define the data type configurations
        self.data_configs = {}

        # Add all valid descriptors from the settings
        for descriptor in VALID_GAME_DESCRIPTORS:
            data_type = descriptor.lower()
            # Add configuration if it doesn't exist yet
            if data_type not in self.data_configs:
                self.data_configs[data_type] = {
                    'game_descriptor': descriptor,
                    'property_key': None
                }

    def collect(self) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
        """
        Collect all types of data from the GameBus API.

        Returns:
            Tuple containing:
            - Dictionary of data points by type
            - List of paths to the saved JSON files
        """
        logger.info(f"Collecting all data for user {self.user_id}")

        results = {}
        file_paths = []
        all_raw_responses = []  # Store all raw responses here to avoid duplicate API calls
        all_metadata = {}  # Store all metadata with X_ prefix

        # Create directory if it doesn't exist
        os.makedirs(RAW_DATA_DIR, exist_ok=True)

        # Try to collect data for each data type
        for data_type, config in self.data_configs.items():
            logger.info(f"Collecting {data_type} data for user {self.user_id}")
            try:
                # Get the game descriptor and property key for this data type
                game_descriptor = config['game_descriptor']
                property_key = config['property_key']

                # Get data from the API
                logger.info(f"Collecting {data_type} data for user {self.user_id}")
                raw_data, actual_game_descriptor, raw_responses = self.client.get_user_data(self.token, self.user_id, game_descriptor, try_all_descriptors=True)

                # Add raw responses to the collection for later use
                all_raw_responses.extend(raw_responses)

                if not raw_data:
                    logger.warning(f"No {data_type} data found for user {self.user_id}")
                    continue

                # Log the actual game descriptor used
                if actual_game_descriptor != game_descriptor:
                    logger.info(f"Using actual game descriptor '{actual_game_descriptor}' instead of requested '{game_descriptor}'")

                # Parse the data
                parsed_data = self.parse_data(raw_data, actual_game_descriptor, property_key)

                if not parsed_data:
                    logger.warning(f"No {data_type} data parsed for user {self.user_id}")
                    continue

                # Save the data to a file using the actual game descriptor
                file_path = self.save_data(parsed_data, actual_game_descriptor.lower())

                if parsed_data:
                    # Store results using the actual game descriptor to ensure correct parameters
                    results[actual_game_descriptor.lower()] = parsed_data

                    # Extract metadata with X_ prefix for separate files
                    metadata = self.extract_metadata(parsed_data, actual_game_descriptor.lower())
                    if metadata:
                        all_metadata[actual_game_descriptor.lower()] = metadata

                    if file_path:
                        file_paths.append(file_path)
                        logger.info(f"Collected {len(parsed_data)} {data_type} data points (game descriptor: {actual_game_descriptor}), saved to {file_path}")
                    else:
                        logger.warning(f"Collected {len(parsed_data)} {data_type} data points (game descriptor: {actual_game_descriptor}), but no file was created")
                else:
                    logger.warning(f"No {data_type} data collected for user {self.user_id}")
            except Exception as e:
                logger.error(f"Failed to collect {data_type} data for user {self.user_id}: {e}")
                logger.exception(e)  # Log the full exception traceback

        # No longer saving all data to a single file as per requirements
        if results:
            logger.info("Skipping creation of all_data.json file as per requirements")
        else:
            logger.warning(f"No data collected for user {self.user_id}")

        # Metadata extraction is still performed, but files are not saved as per requirements
        if all_metadata:
            logger.info(f"Extracted metadata for {len(all_metadata)} data types, but not saving to files as per requirements")

        # Save all raw responses to a single file
        if all_raw_responses:
            all_raw_file_name = f"user_{self.user_id}_all_raw.json"
            all_raw_file_path = os.path.join(RAW_DATA_DIR, all_raw_file_name)

            try:
                # Parse all raw responses into a list of JSON objects
                parsed_responses = []
                for raw_response in all_raw_responses:
                    try:
                        json_data = json.loads(raw_response)
                        parsed_responses.append(json_data)
                    except Exception as e:
                        logger.error(f"Failed to parse raw JSON response: {e}")
                        continue

                # Only save if there are valid parsed responses
                if parsed_responses:
                    # Write all parsed responses to a single file with proper formatting
                    with open(all_raw_file_path, 'w') as json_file:
                        json.dump(parsed_responses, json_file, indent=4)

                    file_paths.append(all_raw_file_path)
                    logger.info(f"Saved {len(parsed_responses)} raw JSON responses to {all_raw_file_path}")
            except Exception as e:
                logger.error(f"Failed to save all raw JSON responses to file: {e}")
                logger.exception(e)  # Log the full exception traceback
        else:
            logger.warning(f"No raw responses collected for user {self.user_id}")

        return results, file_paths

    def parse_data(self, raw_data: List[Dict[str, Any]], game_descriptor: str, property_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Parse raw data from the GameBus API.

        Args:
            raw_data: Raw data from the API
            game_descriptor: Game descriptor for the data
            property_key: Property key for the data (optional)

        Returns:
            Parsed data
        """
        data_list = []

        for data_point in raw_data:
            # Get the game descriptor of this data point
            data_point_gd = data_point.get("gameDescriptor", {}).get("translationKey")

            # Log the game descriptor for debugging
            logger.debug(f"Processing data point with game descriptor: {data_point_gd}")

            # Only process data points that match the requested game descriptor
            if data_point_gd != game_descriptor:
                logger.debug(f"Skipping data point with game descriptor: {data_point_gd} (expected: {game_descriptor})")
                continue

            # For all data types
            data = {}

            # Extract metadata with X_ prefix
            # Include the date field from the raw data if available
            if "date" in data_point:
                timestamp_ms = data_point.get("date")
                # Convert timestamp from milliseconds to a readable date format
                if timestamp_ms:
                    try:
                        # Convert milliseconds to seconds
                        timestamp_sec = timestamp_ms / 1000
                        # Convert to datetime
                        date_time = datetime.datetime.fromtimestamp(timestamp_sec)
                        # Format as string
                        formatted_date = date_time.strftime("%Y-%m-%d %H:%M:%S")
                        data["X_DATE"] = formatted_date
                        logger.debug(f"Added formatted date field: {formatted_date} (from timestamp: {timestamp_ms})")
                    except Exception as e:
                        # If conversion fails, use the original timestamp
                        data["X_DATE"] = timestamp_ms
                        logger.warning(f"Failed to convert timestamp {timestamp_ms} to readable format: {e}")
                else:
                    data["X_DATE"] = timestamp_ms
                    logger.debug(f"Added date field: {timestamp_ms}")

            # Extract additional metadata with X_ prefix
            if "id" in data_point:
                data["X_ACTIVITY_ID"] = data_point.get("id")
                logger.debug(f"Added activity ID field: {data_point.get('id')}")

            # Extract game descriptor information
            if "gameDescriptor" in data_point and isinstance(data_point.get("gameDescriptor"), dict):
                game_descriptor_data = data_point.get("gameDescriptor", {})
                if "translationKey" in game_descriptor_data:
                    data["X_GAME_DESCRIPTOR"] = game_descriptor_data.get("translationKey")
                    logger.debug(f"Added game descriptor field: {game_descriptor_data.get('translationKey')}")
                if "id" in game_descriptor_data:
                    data["X_GAME_DESCRIPTOR_ID"] = game_descriptor_data.get("id")
                    logger.debug(f"Added game descriptor ID field: {game_descriptor_data.get('id')}")

            # Extract player information
            if "player" in data_point and isinstance(data_point.get("player"), dict):
                player_data = data_point.get("player", {})
                if "name" in player_data:
                    data["X_PLAYER_NAME"] = player_data.get("name")
                    logger.debug(f"Added player name field: {player_data.get('name')}")
                if "id" in player_data:
                    data["X_PLAYER_ID"] = player_data.get("id")
                    logger.debug(f"Added player ID field: {player_data.get('id')}")

            # Extract timestamps if available
            if "createdAt" in data_point:
                timestamp_ms = data_point.get("createdAt")
                if timestamp_ms:
                    try:
                        timestamp_sec = timestamp_ms / 1000
                        date_time = datetime.datetime.fromtimestamp(timestamp_sec)
                        formatted_date = date_time.strftime("%Y-%m-%d %H:%M:%S")
                        data["X_CREATED_AT"] = formatted_date
                        logger.debug(f"Added created at field: {formatted_date}")
                    except Exception as e:
                        data["X_CREATED_AT"] = timestamp_ms
                        logger.warning(f"Failed to convert created at timestamp {timestamp_ms} to readable format: {e}")

            if "updatedAt" in data_point:
                timestamp_ms = data_point.get("updatedAt")
                if timestamp_ms:
                    try:
                        timestamp_sec = timestamp_ms / 1000
                        date_time = datetime.datetime.fromtimestamp(timestamp_sec)
                        formatted_date = date_time.strftime("%Y-%m-%d %H:%M:%S")
                        data["X_UPDATED_AT"] = formatted_date
                        logger.debug(f"Added updated at field: {formatted_date}")
                    except Exception as e:
                        data["X_UPDATED_AT"] = timestamp_ms
                        logger.warning(f"Failed to convert updated at timestamp {timestamp_ms} to readable format: {e}")

            # Extract location data if available
            if "latitude" in data_point and "longitude" in data_point:
                data["X_LATITUDE"] = data_point.get("latitude")
                data["X_LONGITUDE"] = data_point.get("longitude")
                logger.debug(f"Added location fields: lat={data_point.get('latitude')}, lon={data_point.get('longitude')}")

            # Extract property keys
            if "propertyInstances" in data_point and isinstance(data_point.get("propertyInstances"), list):
                property_keys = []
                for prop_instance in data_point.get("propertyInstances", []):
                    if "property" in prop_instance and isinstance(prop_instance.get("property"), dict):
                        prop_data = prop_instance.get("property", {})
                        if "translationKey" in prop_data:
                            property_keys.append(prop_data.get("translationKey"))
                if property_keys:
                    data["X_PROPERTY_KEYS"] = property_keys
                    logger.debug(f"Added property keys field: {property_keys}")

            for property_instance in data_point.get("propertyInstances", []):
                prop_key = property_instance.get("property", {}).get("translationKey")
                property_value = property_instance.get("value")
                data[prop_key] = property_value

            if data:
                data_list.append(data)
                logger.debug(f"Added data point with keys: {list(data.keys())}")

        return data_list

    def save_raw_responses(self, raw_responses: List[str], data_type: str) -> str:
        """
        Save raw JSON responses to a single file.

        Args:
            raw_responses: List of raw JSON responses
            data_type: Type of data being saved

        Returns:
            Path to the saved file
        """
        # Don't create files if there is no data
        if not raw_responses:
            logger.info(f"No raw responses to save for {data_type}")
            return ""

        # Create file path
        file_name = f"user_{self.user_id}_{data_type}_raw.json"
        file_path = os.path.join(RAW_DATA_DIR, file_name)

        try:
            # Parse all raw responses into a list of JSON objects
            parsed_responses = []
            for raw_response in raw_responses:
                try:
                    json_data = json.loads(raw_response)
                    parsed_responses.append(json_data)
                except Exception as e:
                    logger.error(f"Failed to parse raw JSON response for {data_type}: {e}")
                    continue

            # Only save if there are valid parsed responses
            if parsed_responses:
                # Write all parsed responses to a single file with proper formatting
                with open(file_path, 'w') as json_file:
                    json.dump(parsed_responses, json_file, indent=4)

                logger.info(f"Saved {len(parsed_responses)} raw JSON responses for {data_type} to {file_path}")
                return file_path
            else:
                logger.warning(f"No valid raw JSON responses to save for {data_type}")
                return ""
        except Exception as e:
            logger.error(f"Failed to save raw JSON responses for {data_type} to file: {e}")
            return ""

    def extract_metadata(self, data: List[Dict[str, Any]], data_type: str) -> List[Dict[str, Any]]:
        """
        Extract metadata with X_ prefix from parsed data.

        Args:
            data: Parsed data
            data_type: Type of data

        Returns:
            List of metadata dictionaries
        """
        metadata_list = []

        for item in data:
            # Extract all fields with X_ prefix
            metadata = {k: v for k, v in item.items() if k.startswith("X_")}

            if metadata:
                metadata_list.append(metadata)
                logger.debug(f"Extracted metadata with keys: {list(metadata.keys())}")

        logger.info(f"Extracted {len(metadata_list)} metadata items for {data_type}")
        return metadata_list

    def save_data(self, data: List[Dict[str, Any]], data_type: str) -> str:
        """
        Save data to a JSON file.

        Args:
            data: Data to save
            data_type: Type of data being saved

        Returns:
            Path to the saved JSON file
        """
        # Don't create files if there is no data
        if not data:
            logger.info(f"No data to save for {data_type}")
            return ""

        # Create file path
        json_file_name = f"user_{self.user_id}_{data_type}.json"
        json_file_path = os.path.join(RAW_DATA_DIR, json_file_name)

        # Fields to exclude from JSON files
        excluded_fields = ["X_GAME_DESCRIPTOR", "X_GAME_DESCRIPTOR_ID", "X_PLAYER_ID", "X_PROPERTY_KEYS"]

        # Filter out excluded fields from each data point
        filtered_data = []
        for item in data:
            filtered_item = {k: v for k, v in item.items() if k not in excluded_fields}
            filtered_data.append(filtered_item)

        # Save filtered data to JSON file
        try:
            with open(json_file_path, 'w') as json_file:
                json.dump(filtered_data, json_file, indent=4)

            logger.info(f"Saved {len(filtered_data)} {data_type} data points to {json_file_path}")
            return json_file_path
        except Exception as e:
            logger.error(f"Failed to save data to file: {e}")
            return ""
