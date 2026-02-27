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
    Collector for all configured GameBus data descriptors.
    """

    def __init__(
        self,
        client,
        token: str,
        user_id: int,
        user_email: Optional[str] = None,
        account_user_id: Optional[str] = None,
    ):
        """
        Initialize the collector.

        Args:
            client: GameBusClient instance
            token: User access token
            user_id: Player ID (GameBus player.id)
            user_email: User email (optional)
            account_user_id: External/User ID from the users file (optional)
        """
        self.client = client
        self.token = token
        self.user_id = user_id
        self.user_email = user_email
        self.account_user_id = account_user_id

        # Keep a simple ordered list of descriptors instead of an over-engineered config dict
        self.descriptors = list(VALID_GAME_DESCRIPTORS)

    def collect(self) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
        """
        Collect all configured data types from the GameBus API.

        Returns:
            Tuple containing:
            - Dictionary of parsed data by descriptor (key = descriptor.lower())
            - List of paths to saved JSON files
        """
        results: Dict[str, List[Dict[str, Any]]] = {}
        file_paths: List[str] = []
        all_raw_responses: List[str] = []

        os.makedirs(RAW_DATA_DIR, exist_ok=True)

        for descriptor in self.descriptors:
            data_type = descriptor.lower()

            try:
                raw_data, actual_game_descriptor, raw_responses = self.client.get_user_data(
                    self.token,
                    self.user_id,
                    descriptor,
                    try_all_descriptors=False,
                )
            except Exception as e:
                logger.error(
                    f"Failed to fetch {data_type} data for user {self.user_id}: {e}"
                )
                logger.exception(e)
                continue

            if raw_responses:
                all_raw_responses.extend(raw_responses)

            if not raw_data:
                logger.warning(f"No {data_type} data found for user {self.user_id}")
                continue

            try:
                parsed_data = self.parse_data(raw_data, actual_game_descriptor)
            except Exception as e:
                logger.error(
                    f"Failed to parse {data_type} data for user {self.user_id}: {e}"
                )
                logger.exception(e)
                continue

            if not parsed_data:
                logger.warning(
                    f"No parsed {data_type} data remained after filtering for user {self.user_id}"
                )
                continue

            try:
                actual_data_type = actual_game_descriptor.lower()
                file_path = self.save_data(parsed_data, actual_data_type)

                results[actual_data_type] = parsed_data

                if file_path:
                    file_paths.append(file_path)
            except Exception as e:
                logger.error(
                    f"Failed to save {data_type} data for user {self.user_id}: {e}"
                )
                logger.exception(e)

        raw_file_path = self._save_all_raw_responses(all_raw_responses)
        if raw_file_path:
            file_paths.append(raw_file_path)
        elif not all_raw_responses:
            logger.warning(f"No raw responses collected for user {self.user_id}")

        return results, file_paths

    def parse_data(
        self,
        raw_data: List[Dict[str, Any]],
        game_descriptor: str,
    ) -> List[Dict[str, Any]]:
        """
        Parse raw data from the GameBus API.

        Args:
            raw_data: Raw data from the API
            game_descriptor: Descriptor that the returned items should match

        Returns:
            Parsed data
        """
        data_list: List[Dict[str, Any]] = []

        for data_point in raw_data:
            data_point_gd = (
                data_point.get("gameDescriptor", {}) or {}
            ).get("translationKey")

            # Keep only items that match the descriptor we are currently saving
            if data_point_gd != game_descriptor:
                continue

            data: Dict[str, Any] = {}

            # Common metadata
            self._add_timestamp_field(data, data_point, "date", "X_DATE")
            self._add_timestamp_field(data, data_point, "createdAt", "X_CREATED_AT")
            self._add_timestamp_field(data, data_point, "updatedAt", "X_UPDATED_AT")

            if "id" in data_point:
                data["X_ACTIVITY_ID"] = data_point.get("id")

            game_descriptor_data = data_point.get("gameDescriptor")
            if isinstance(game_descriptor_data, dict):
                if "translationKey" in game_descriptor_data:
                    data["X_GAME_DESCRIPTOR"] = game_descriptor_data.get("translationKey")
                if "id" in game_descriptor_data:
                    data["X_GAME_DESCRIPTOR_ID"] = game_descriptor_data.get("id")

            player_data = data_point.get("player")
            if isinstance(player_data, dict):
                if "name" in player_data:
                    data["X_PLAYER_NAME"] = player_data.get("name")
                if "id" in player_data:
                    data["X_PLAYER_ID"] = player_data.get("id")

            if "latitude" in data_point and "longitude" in data_point:
                data["X_LATITUDE"] = data_point.get("latitude")
                data["X_LONGITUDE"] = data_point.get("longitude")

            property_instances = data_point.get("propertyInstances")
            if not isinstance(property_instances, list):
                property_instances = []

            property_keys: List[str] = []
            for property_instance in property_instances:
                prop_data = property_instance.get("property")
                if isinstance(prop_data, dict):
                    prop_key = prop_data.get("translationKey")
                    if prop_key:
                        property_keys.append(prop_key)

            if property_keys:
                data["X_PROPERTY_KEYS"] = property_keys

            # Extract actual property values
            for property_instance in property_instances:
                prop_data = property_instance.get("property")
                if not isinstance(prop_data, dict):
                    continue

                prop_key = prop_data.get("translationKey")
                if not prop_key:
                    continue

                property_value = property_instance.get("value")

                # If the same property appears multiple times, keep all values
                if prop_key in data and not prop_key.startswith("X_"):
                    existing = data[prop_key]
                    if isinstance(existing, list):
                        existing.append(property_value)
                    else:
                        data[prop_key] = [existing, property_value]
                else:
                    data[prop_key] = property_value

            if data:
                data_list.append(data)

        return data_list

    def _save_all_raw_responses(self, raw_responses: List[str]) -> str:
        """
        Save all raw JSON responses for the user into a single file.

        Returns:
            Path to the saved file, or "" if nothing was saved.
        """
        if not raw_responses:
            return ""

        file_name = f"player_{self.user_id}_all_raw.json"
        file_path = os.path.join(RAW_DATA_DIR, file_name)

        try:
            parsed_responses = []
            for raw_response in raw_responses:
                try:
                    parsed_responses.append(json.loads(raw_response))
                except Exception as e:
                    logger.error(f"Failed to parse raw JSON response: {e}")

            if not parsed_responses:
                logger.warning(
                    f"No valid raw JSON responses to save for user {self.user_id}"
                )
                return ""

            with open(file_path, "w", encoding="utf-8") as json_file:
                json.dump(parsed_responses, json_file, indent=4)

            logger.info(
                f"Saved {len(parsed_responses)} raw response payloads to {file_path}"
            )
            return file_path

        except Exception as e:
            logger.error(
                f"Failed to save all raw JSON responses for user {self.user_id}: {e}"
            )
            logger.exception(e)
            return ""

    def _add_timestamp_field(
        self,
        target: Dict[str, Any],
        source: Dict[str, Any],
        source_key: str,
        target_key: str,
    ) -> None:
        """
        Copy a timestamp field from the raw payload into the parsed item,
        converting epoch milliseconds to a readable datetime string when possible.
        """
        if source_key not in source:
            return

        timestamp_value = source.get(source_key)
        if timestamp_value is None:
            target[target_key] = timestamp_value
            return

        try:
            target[target_key] = self._format_epoch_value(timestamp_value)
        except Exception as e:
            target[target_key] = timestamp_value
            logger.warning(
                f"Failed to convert timestamp {timestamp_value} for {target_key}: {e}"
            )

    def _format_epoch_value(self, value: Any) -> Any:
        """
        Convert an epoch timestamp (ms or s) to 'YYYY-MM-DD HH:MM:SS' when possible.
        Otherwise return the original value unchanged.
        """
        if isinstance(value, (int, float)):
            # Heuristic: GameBus timestamps are usually milliseconds
            if value >= 1e12:
                dt = datetime.datetime.fromtimestamp(value / 1000)
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            if value >= 1e9:
                dt = datetime.datetime.fromtimestamp(value)
                return dt.strftime("%Y-%m-%d %H:%M:%S")

        return value

    def _format_dates_in_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert date-like fields in an item to a readable string format.

        Rules:
        - Keys containing DATE, TIME, TIMESTAMP, or ending with _AT are treated as date-like.
        - Numeric values are interpreted as epoch ms if >= 1e12, else epoch seconds if >= 1e9.
        - Digit-only strings of length >= 13 or length 10 are treated similarly.
        - ISO 8601 strings are normalized to 'YYYY-MM-DD HH:MM:SS' when key looks date-like.
        """
        def is_date_like(key: str) -> bool:
            k = (key or "").upper()
            return ("DATE" in k) or ("TIME" in k) or ("TIMESTAMP" in k) or k.endswith("_AT")

        def to_readable(dt: datetime.datetime) -> str:
            try:
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                return dt.isoformat(sep=" ")

        def convert_value(key: str, value: Any) -> Any:
            if not is_date_like(key):
                return value

            try:
                if isinstance(value, (int, float)):
                    if value >= 1e12:
                        return to_readable(datetime.datetime.fromtimestamp(value / 1000))
                    if value >= 1e9:
                        return to_readable(datetime.datetime.fromtimestamp(value))
                    return value

                if isinstance(value, str):
                    v = value.strip()

                    if v.isdigit():
                        if len(v) >= 13:
                            return to_readable(datetime.datetime.fromtimestamp(int(v[:13]) / 1000))
                        if len(v) == 10:
                            return to_readable(datetime.datetime.fromtimestamp(int(v)))

                    try:
                        iso_v = v.replace("Z", "+00:00")
                        dt = datetime.datetime.fromisoformat(iso_v)
                        return to_readable(dt)
                    except Exception:
                        return value

            except Exception:
                return value

            return value

        return {k: convert_value(k, v) for k, v in item.items()}

    def save_data(self, data: List[Dict[str, Any]], data_type: str) -> str:
        """
        Save data to a JSON file.

        Args:
            data: Data to save
            data_type: Type of data being saved

        Returns:
            Path to the saved JSON file
        """
        if not data:
            logger.info(f"No data to save for {data_type}")
            return ""

        json_file_name = f"player_{self.user_id}_{data_type}.json"
        json_file_path = os.path.join(RAW_DATA_DIR, json_file_name)

        excluded_fields = [
            "X_GAME_DESCRIPTOR",
            "X_GAME_DESCRIPTOR_ID",
            "X_PLAYER_ID",
            "X_PROPERTY_KEYS",
        ]

        filtered_data = []
        for item in data:
            filtered_item = {k: v for k, v in item.items() if k not in excluded_fields}
            formatted_item = self._format_dates_in_item(filtered_item)
            filtered_data.append(formatted_item)

        try:
            with open(json_file_path, "w", encoding="utf-8") as json_file:
                json.dump(filtered_data, json_file, indent=4)

            logger.info(
                f"Saved {len(filtered_data)} data points for {data_type} to {json_file_path}"
            )
            return json_file_path

        except Exception as e:
            logger.error(f"Failed to save data to file: {e}")
            logger.exception(e)
            return ""