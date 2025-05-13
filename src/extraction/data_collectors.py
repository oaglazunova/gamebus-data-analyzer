"""
Data collectors for different types of GameBus data.
"""
import ast
import json
import pandas as pd
import logging
import os
from typing import Dict, List, Any, Tuple

from config.settings import VALID_PROPERTY_KEYS
from config.paths import RAW_DATA_DIR
from src.extraction.gamebus_client import GameBusClient

# Set up logging
logger = logging.getLogger(__name__)

class DataCollector:
    """
    Base class for collecting data from GameBus.
    """

    # Class-level cache for raw data
    _raw_data_cache = {}

    def __init__(self, client: GameBusClient, token: str, user_id: int):
        """
        Initialize the data collector.

        Args:
            client: GameBus client
            token: Access token
            user_id: User ID
        """
        self.client = client
        self.token = token
        self.user_id = user_id

    def _save_raw_data(self, data: List[Dict[str, Any]], filename: str) -> str:
        """
        Save raw data to a JSON file.

        Args:
            data: Data to save
            filename: Name of the file

        Returns:
            Path to the saved file
        """
        if not data:
            logger.warning(f"No data to save for {filename}. File will not be created.")
            return ""

        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        file_path = os.path.join(RAW_DATA_DIR, filename)

        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

        logger.info(f"Data saved to {file_path} with {len(data)} records")
        return file_path

    def _save_csv_data(self, data: List[Dict[str, Any]], filename: str) -> str:
        """
        Save data to a CSV file.

        Args:
            data: Data to save
            filename: Name of the file

        Returns:
            Path to the saved file
        """
        if not data:
            logger.warning(f"No data to save for {filename}. CSV file will not be created.")
            return ""

        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        file_path = os.path.join(RAW_DATA_DIR, filename)

        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)

        logger.info(f"Data saved to {file_path} with {len(data)} records")
        return file_path

    def _parse_general_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parse general GameBus data.

        Args:
            raw_data: Raw data from GameBus

        Returns:
            Parsed data
        """
        if not raw_data:
            logger.warning("No raw data to parse in _parse_general_data")
            return []

        # Pre-allocate the list with an estimated size to avoid resizing
        data_list = []
        data_list_append = data_list.append  # Local reference for faster append

        # Cache common operations
        get = dict.get

        for data_point in raw_data:
            try:
                # Initialize data dictionary with estimated capacity
                data = {}

                # Extract basic metadata
                activity_id = get(data_point, "id")
                activity_date = get(data_point, "date")
                game_descriptor_dict = get(data_point, "gameDescriptor", {})
                game_descriptor = get(game_descriptor_dict, "translationKey", "UNKNOWN")

                # Extract user information if available
                user = get(data_point, "player", {})
                if user:
                    data["user_id"] = get(user, "id")
                    first_name = get(user, 'firstName', '')
                    last_name = get(user, 'lastName', '')
                    data["user_name"] = f"{first_name} {last_name}".strip()

                # Extract property instances
                for property_instance in get(data_point, "propertyInstances", []):
                    try:
                        property_dict = get(property_instance, "property", {})
                        property_key = get(property_dict, "translationKey")
                        property_value = get(property_instance, "value")

                        if property_key:
                            # Try to parse JSON or list values
                            if isinstance(property_value, str) and (
                                property_value.startswith('{') or 
                                property_value.startswith('[')
                            ):
                                try:
                                    data[property_key] = ast.literal_eval(property_value)
                                except (ValueError, SyntaxError):
                                    data[property_key] = property_value
                            else:
                                data[property_key] = property_value
                    except Exception as e:
                        logger.warning(f"Error processing property instance: {e}")
                        # Continue with next property instance

                # Add activity metadata
                data["activity_id"] = activity_id
                data["date"] = activity_date
                data["gameDescriptor"] = game_descriptor

                # Add points information if available
                if "points" in data_point:
                    data["points"] = get(data_point, "points")

                # Add data provider information if available
                data_provider = get(data_point, "dataProvider", {})
                if data_provider:
                    data["data_provider_id"] = get(data_provider, "id")
                    data["data_provider_name"] = get(data_provider, "name")

                data_list_append(data)

            except Exception as e:
                logger.error(f"Error parsing data point: {e}")
                # Try to save minimal information
                try:
                    data_list_append({
                        "activity_id": get(data_point, "id"),
                        "date": get(data_point, "date"),
                        "gameDescriptor": get(get(data_point, "gameDescriptor", {}), "translationKey", "UNKNOWN"),
                        "parse_error": str(e)
                    })
                except Exception as inner_e:
                    logger.error(f"Failed to save error information: {inner_e}")

        logger.info(f"Parsed {len(data_list)} data points")
        return data_list

    def _parse_tizen_data(self, raw_data: List[Dict[str, Any]], property_key: str) -> List[Dict[str, Any]]:
        """
        Parse Tizen-specific data.

        Args:
            raw_data: Raw data from GameBus
            property_key: Property key to extract

        Returns:
            Parsed data
        """
        if property_key not in VALID_PROPERTY_KEYS:
            logger.warning(f"Property key '{property_key}' not in VALID_PROPERTY_KEYS. Attempting to use it anyway.")

        if not raw_data:
            logger.warning(f"No raw data to parse in _parse_tizen_data for property_key: {property_key}")
            return []

        # Pre-allocate the list with an estimated size to avoid resizing
        data_list = []
        data_list_append = data_list.append  # Local reference for faster append

        # Cache common operations
        get = dict.get

        # Create a set of property keys to match for faster lookup
        property_keys_to_match = {property_key}

        # Add substring matches for more flexible matching
        for valid_key in VALID_PROPERTY_KEYS:
            if property_key in valid_key:
                property_keys_to_match.add(valid_key)

        for data_point in raw_data:
            activity_id = get(data_point, "id")
            activity_date = get(data_point, "date")
            game_descriptor_dict = get(data_point, "gameDescriptor", {})
            game_descriptor = get(game_descriptor_dict, "translationKey", "UNKNOWN")

            for property_instance in get(data_point, "propertyInstances", []):
                property_dict = get(property_instance, "property", {})
                prop_key = get(property_dict, "translationKey")
                property_value = get(property_instance, "value")

                # Skip if this is not the property we're looking for
                # Try exact match first, then check if it's in our set of keys to match
                if prop_key != property_key and prop_key not in property_keys_to_match:
                    continue

                # Try to parse the property value
                try:
                    # For list-like properties (logs) or any property that might contain structured data
                    if prop_key in ["ACTIVITY_TYPE", "HRM_LOG", "ACCELEROMETER_LOG"] or (
                        isinstance(property_value, str) and (
                            property_value.startswith('{') or 
                            property_value.startswith('[')
                        )
                    ):
                        try:
                            # Try to parse as a list or dict
                            if isinstance(property_value, str):
                                parsed_value = ast.literal_eval(property_value)
                            else:
                                parsed_value = property_value

                            # Handle both list and dict formats
                            if isinstance(parsed_value, dict):
                                # Add activity metadata
                                parsed_value["activity_id"] = activity_id
                                parsed_value["activity_date"] = activity_date
                                parsed_value["game_descriptor"] = game_descriptor
                                data_list_append(parsed_value)
                            elif isinstance(parsed_value, list):
                                # Process list items in batches to reduce memory usage
                                batch_size = 100
                                for i in range(0, len(parsed_value), batch_size):
                                    batch = parsed_value[i:i+batch_size]
                                    for item in batch:
                                        if isinstance(item, dict):
                                            # Add activity metadata
                                            item["activity_id"] = activity_id
                                            item["activity_date"] = activity_date
                                            item["game_descriptor"] = game_descriptor
                                            data_list_append(item)
                                        else:
                                            # Handle primitive values in lists
                                            data_list_append({
                                                "value": item,
                                                "activity_id": activity_id,
                                                "activity_date": activity_date,
                                                "game_descriptor": game_descriptor
                                            })
                            else:
                                # Handle unexpected types
                                data_list_append({
                                    "value": parsed_value,
                                    "activity_id": activity_id,
                                    "activity_date": activity_date,
                                    "game_descriptor": game_descriptor
                                })
                        except (ValueError, SyntaxError) as e:
                            # Try to save the raw value
                            data_list_append({
                                "raw_value": str(property_value)[:1000] if isinstance(property_value, str) else str(property_value),
                                "activity_id": activity_id,
                                "activity_date": activity_date,
                                "game_descriptor": game_descriptor,
                                "parse_error": str(e)
                            })
                    else:
                        # For simple properties
                        data_list_append({
                            "value": property_value,
                            "activity_id": activity_id,
                            "activity_date": activity_date,
                            "game_descriptor": game_descriptor,
                            "property_key": prop_key
                        })
                except Exception as e:
                    logger.error(f"Unexpected error parsing property {prop_key}: {e}")
                    # Try to save minimal information
                    data_list_append({
                        "activity_id": activity_id,
                        "activity_date": activity_date,
                        "game_descriptor": game_descriptor,
                        "property_key": prop_key,
                        "parse_error": str(e)
                    })

        logger.info(f"Parsed {len(data_list)} data points for property key {property_key}")
        return data_list


class LocationDataCollector(DataCollector):
    """
    Collector for GPS location data.
    """

    def collect(self) -> Tuple[List[Dict[str, Any]], str]:
        """
        Collect GPS location data.

        Returns:
            Tuple of (parsed data, file path)
        """
        # Use class-level cache for raw data
        cache_key = f"{self.user_id}_GEOFENCE"
        if cache_key in self._raw_data_cache:
            logger.info(f"Using cached GEOFENCE data for user {self.user_id}")
            raw_data = self._raw_data_cache[cache_key]
        else:
            logger.info(f"Fetching GEOFENCE data for user {self.user_id}")
            raw_data = self.client.get_user_data(self.token, self.user_id, "GEOFENCE", try_all_descriptors=True)
            self._raw_data_cache[cache_key] = raw_data

        parsed_data = self._parse_general_data(raw_data)
        file_path = self._save_raw_data(parsed_data, f"user_{self.user_id}_location.json")
        return parsed_data, file_path


class MoodDataCollector(DataCollector):
    """
    Collector for mood logging data.
    """

    def collect(self) -> Tuple[List[Dict[str, Any]], str]:
        """
        Collect mood logging data.

        Returns:
            Tuple of (parsed data, file path)
        """
        # Use class-level cache for raw data
        cache_key = f"{self.user_id}_LOG_MOOD"
        if cache_key in self._raw_data_cache:
            logger.info(f"Using cached LOG_MOOD data for user {self.user_id}")
            raw_data = self._raw_data_cache[cache_key]
        else:
            logger.info(f"Fetching LOG_MOOD data for user {self.user_id}")
            raw_data = self.client.get_user_data(self.token, self.user_id, "LOG_MOOD", try_all_descriptors=True)
            self._raw_data_cache[cache_key] = raw_data

        parsed_data = self._parse_general_data(raw_data)
        file_path = self._save_raw_data(parsed_data, f"user_{self.user_id}_mood.json")
        return parsed_data, file_path


class ActivityTypeDataCollector(DataCollector):
    """
    Collector for activity type data.
    """

    def collect(self) -> Tuple[List[Dict[str, Any]], str]:
        """
        Collect activity type data.

        Returns:
            Tuple of (parsed data, file path)
        """
        # Use class-level cache for raw data
        cache_key = f"{self.user_id}_TIZEN(DETAIL)"
        if cache_key in self._raw_data_cache:
            logger.info(f"Using cached TIZEN data for user {self.user_id}")
            raw_data = self._raw_data_cache[cache_key]
        else:
            logger.info(f"Fetching TIZEN data for user {self.user_id}")
            raw_data = self.client.get_user_data(self.token, self.user_id, "TIZEN(DETAIL)", try_all_descriptors=True)
            self._raw_data_cache[cache_key] = raw_data

        parsed_data = self._parse_tizen_data(raw_data, "ACTIVITY_TYPE")
        file_path = self._save_raw_data(parsed_data, f"user_{self.user_id}_activity_type.json")
        return parsed_data, file_path


class HeartRateDataCollector(DataCollector):
    """
    Collector for heart rate data.
    """

    def collect(self) -> Tuple[List[Dict[str, Any]], str]:
        """
        Collect heart rate data.

        Returns:
            Tuple of (parsed data, file path)
        """
        # Use class-level cache for raw data
        cache_key = f"{self.user_id}_TIZEN(DETAIL)"
        if cache_key in self._raw_data_cache:
            logger.info(f"Using cached TIZEN data for user {self.user_id}")
            raw_data = self._raw_data_cache[cache_key]
        else:
            logger.info(f"Fetching TIZEN data for user {self.user_id}")
            raw_data = self.client.get_user_data(self.token, self.user_id, "TIZEN(DETAIL)", try_all_descriptors=True)
            self._raw_data_cache[cache_key] = raw_data

        parsed_data = self._parse_tizen_data(raw_data, "HRM_LOG")
        file_path = self._save_raw_data(parsed_data, f"user_{self.user_id}_heartrate.json")
        return parsed_data, file_path


class AccelerometerDataCollector(DataCollector):
    """
    Collector for accelerometer data.
    """

    def collect(self) -> Tuple[List[Dict[str, Any]], str]:
        """
        Collect accelerometer data.

        Returns:
            Tuple of (parsed data, file path)
        """
        # Use class-level cache for raw data
        cache_key = f"{self.user_id}_TIZEN(DETAIL)"
        if cache_key in self._raw_data_cache:
            logger.info(f"Using cached TIZEN data for user {self.user_id}")
            raw_data = self._raw_data_cache[cache_key]
        else:
            logger.info(f"Fetching TIZEN data for user {self.user_id}")
            raw_data = self.client.get_user_data(self.token, self.user_id, "TIZEN(DETAIL)", try_all_descriptors=True)
            self._raw_data_cache[cache_key] = raw_data

        parsed_data = self._parse_tizen_data(raw_data, "ACCELEROMETER_LOG")
        file_path = self._save_raw_data(parsed_data, f"user_{self.user_id}_accelerometer.json")
        return parsed_data, file_path


class NotificationDataCollector(DataCollector):
    """
    Collector for notification data.
    """

    def collect(self) -> Tuple[List[Dict[str, Any]], str]:
        """
        Collect notification data.

        Returns:
            Tuple of (parsed data, file path)
        """
        # Use class-level cache for raw data
        cache_key = f"{self.user_id}_NOTIFICATION(DETAIL)"
        if cache_key in self._raw_data_cache:
            logger.info(f"Using cached NOTIFICATION data for user {self.user_id}")
            raw_data = self._raw_data_cache[cache_key]
        else:
            logger.info(f"Fetching NOTIFICATION data for user {self.user_id}")
            raw_data = self.client.get_user_data(self.token, self.user_id, "NOTIFICATION(DETAIL)", try_all_descriptors=True)
            self._raw_data_cache[cache_key] = raw_data

        parsed_data = self._parse_general_data(raw_data)
        file_path = self._save_raw_data(parsed_data, f"user_{self.user_id}_notifications.json")
        return parsed_data, file_path
