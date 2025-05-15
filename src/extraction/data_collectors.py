"""
Data collectors for extracting user data from the GameBus API.
"""
import json
import os
import logging
import ast
from typing import Dict, List, Tuple, Any, Optional

from config.paths import RAW_DATA_DIR
from config.settings import VALID_GAME_DESCRIPTORS, VALID_PROPERTY_KEYS

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

        # Define aggregation configurations
        self.aggregation_configs = {
            'DAILY_WALK': {
                'activity_scheme': ['WALK'],
                'transformations': [
                    {'operation': 'SUM', 'source': 'DISTANCE', 'target': 'DISTANCE_SUM'},
                    {'operation': 'AVERAGE', 'source': 'DISTANCE', 'target': 'DISTANCE_AVG'},
                    {'operation': 'MAXIMUM', 'source': 'DISTANCE', 'target': 'DISTANCE_MAX'},
                    {'operation': 'SUM', 'source': 'STEPS', 'target': 'STEPS_SUM'},
                    {'operation': 'AVERAGE', 'source': 'STEPS', 'target': 'STEPS_AVG'},
                    {'operation': 'MAXIMUM', 'source': 'STEPS', 'target': 'STEPS_MAX'}
                ]
            },
            'DAILY_RUN': {
                'activity_scheme': ['RUN'],
                'transformations': [
                    {'operation': 'SUM', 'source': 'DISTANCE', 'target': 'DISTANCE_SUM'},
                    {'operation': 'AVERAGE', 'source': 'DISTANCE', 'target': 'DISTANCE_AVG'},
                    {'operation': 'MAXIMUM', 'source': 'DISTANCE', 'target': 'DISTANCE_MAX'},
                    {'operation': 'SUM', 'source': 'STEPS', 'target': 'STEPS_SUM'},
                    {'operation': 'AVERAGE', 'source': 'STEPS', 'target': 'STEPS_AVG'},
                    {'operation': 'MAXIMUM', 'source': 'STEPS', 'target': 'STEPS_MAX'}
                ]
            },
            'DAILY_WALK_RUN': {
                'activity_scheme': ['WALK', 'RUN'],
                'transformations': [
                    {'operation': 'SUM', 'source': 'DISTANCE', 'target': 'DISTANCE_SUM'},
                    {'operation': 'AVERAGE', 'source': 'DISTANCE', 'target': 'DISTANCE_AVG'},
                    {'operation': 'MAXIMUM', 'source': 'DISTANCE', 'target': 'DISTANCE_MAX'},
                    {'operation': 'SUM', 'source': 'STEPS', 'target': 'STEPS_SUM'},
                    {'operation': 'AVERAGE', 'source': 'STEPS', 'target': 'STEPS_AVG'},
                    {'operation': 'MAXIMUM', 'source': 'STEPS', 'target': 'STEPS_MAX'}
                ]
            },
            'DAILY_BIKE': {
                'activity_scheme': ['BIKE'],
                'transformations': [
                    {'operation': 'SUM', 'source': 'DISTANCE', 'target': 'DISTANCE_SUM'},
                    {'operation': 'AVERAGE', 'source': 'DISTANCE', 'target': 'DISTANCE_AVG'},
                    {'operation': 'MAXIMUM', 'source': 'DISTANCE', 'target': 'DISTANCE_MAX'}
                ]
            },
            'WEEKLY_WALK': {
                'activity_scheme': ['WALK'],
                'transformations': [
                    {'operation': 'SUM', 'source': 'DISTANCE', 'target': 'DISTANCE_SUM'},
                    {'operation': 'AVERAGE', 'source': 'DISTANCE', 'target': 'DISTANCE_AVG'},
                    {'operation': 'MAXIMUM', 'source': 'DISTANCE', 'target': 'DISTANCE_MAX'},
                    {'operation': 'SUM', 'source': 'STEPS', 'target': 'STEPS_SUM'},
                    {'operation': 'AVERAGE', 'source': 'STEPS', 'target': 'STEPS_AVG'},
                    {'operation': 'MAXIMUM', 'source': 'STEPS', 'target': 'STEPS_MAX'}
                ]
            },
            'WEEKLY_RUN': {
                'activity_scheme': ['RUN'],
                'transformations': [
                    {'operation': 'SUM', 'source': 'DISTANCE', 'target': 'DISTANCE_SUM'},
                    {'operation': 'AVERAGE', 'source': 'DISTANCE', 'target': 'DISTANCE_AVG'},
                    {'operation': 'MAXIMUM', 'source': 'DISTANCE', 'target': 'DISTANCE_MAX'},
                    {'operation': 'SUM', 'source': 'STEPS', 'target': 'STEPS_SUM'},
                    {'operation': 'AVERAGE', 'source': 'STEPS', 'target': 'STEPS_AVG'},
                    {'operation': 'MAXIMUM', 'source': 'STEPS', 'target': 'STEPS_MAX'}
                ]
            },
            'WEEKLY_WALK_RUN': {
                'activity_scheme': ['WALK', 'RUN'],
                'transformations': [
                    {'operation': 'SUM', 'source': 'DISTANCE', 'target': 'DISTANCE_SUM'},
                    {'operation': 'AVERAGE', 'source': 'DISTANCE', 'target': 'DISTANCE_AVG'},
                    {'operation': 'MAXIMUM', 'source': 'DISTANCE', 'target': 'DISTANCE_MAX'},
                    {'operation': 'SUM', 'source': 'STEPS', 'target': 'STEPS_SUM'},
                    {'operation': 'AVERAGE', 'source': 'STEPS', 'target': 'STEPS_AVG'},
                    {'operation': 'MAXIMUM', 'source': 'STEPS', 'target': 'STEPS_MAX'}
                ]
            },
            'WEEKLY_BIKE': {
                'activity_scheme': ['BIKE'],
                'transformations': [
                    {'operation': 'SUM', 'source': 'DISTANCE', 'target': 'DISTANCE_SUM'},
                    {'operation': 'AVERAGE', 'source': 'DISTANCE', 'target': 'DISTANCE_AVG'},
                    {'operation': 'MAXIMUM', 'source': 'DISTANCE', 'target': 'DISTANCE_MAX'}
                ]
            }
        }

        # Special configurations for TIZEN(DETAIL) with different property keys
        self.data_configs['activity'] = {
            'game_descriptor': "TIZEN(DETAIL)",
            'property_key': "ACTIVITY_TYPE"
        }
        self.data_configs['heartrate'] = {
            'game_descriptor': "TIZEN(DETAIL)",
            'property_key': "HRM_LOG"
        }
        self.data_configs['accelerometer'] = {
            'game_descriptor': "TIZEN(DETAIL)",
            'property_key': "ACCELEROMETER_LOG"
        }
        self.data_configs['ppg'] = {
            'game_descriptor': "TIZEN(DETAIL)",
            'property_key': "PPG_LOG"
        }

        # Add configurations for specific game descriptors based on the table in the issue description
        # GameBus Watch App data types
        game_descriptors = [
            "GENERAL_ACTIVITY",
            "WALK",
            "RUN",
            "BIKE",
            "TRANSPORT",
            "LOCATION",
            "SOCIAL_SELFIE",
            "PHYSICAL_ACTIVITY",
            "SCORE_GAMEBUS_POINTS",
            "DAY_AGGREGATE",
            "DAY_AGGREGATE_WALK",
            "DAY_AGGREGATE_RUN",
            "NUTRITION_DIARY",
            "DRINKING_DIARY",
            "CONSENT",
            "PLAY_LOTTERY",
            "NAVIGATE_APP",
            "H5P_GENERAL",
            "GENERAL_SURVEY",
            "BODY_MASS_INDEX",
            "MIPIP",
            "WALK_PERFORMANCE",
            "BIKE_PERFORMANCE",
            "SPORTS_PERFORMANCE",
            "LOG_MOOD",
            "TIZEN(DETAIL)",  # Already handled specially above, but included for completeness
            "NOTIFICATION(DETAIL)",
            "TIZEN_LOG(DETAIL)",
            "HOW_ACTIVE_DO_YOU_FEEL_AT_THE_MOMENT",
            "DO_LATER",
            "BODY_MOVEMENT",
            "EMOTION_SCALE_FIVE",
            "RECENTLY_EATEN",
            "RECENTLY_COLD_BEVERAGE",
            "ELDERLY_ACTIVITY",
            "VALENCE",
            "AROUSAL",
            "LOG_AWARENESS",
            "SUBJECTIVE_MOOD",
            "SUBJECTIVE_EMOTION",
            "LOCATION_FAMILIARITY",
            "PEOPLE_COMPANY",
            "GEOFENCE"
        ]

        # Nutrida data types
        nutrida_descriptors = [
            "VISIT_APP_PAGE",
            "PLAN_MEAL",
            "NUTRITION_SUMMARY",
            "NUTRITION_DIARY_VID"
        ]

        # Garmin data types (from the table, they provide the same data as DAY_AGGREGATE)
        garmin_descriptors = [
            "DAY_AGGREGATE"  # Already included in game_descriptors, but mentioned here for clarity
        ]

        # Add all valid descriptors from the table
        for descriptor in game_descriptors + nutrida_descriptors:
            data_type = descriptor.lower()
            # Skip TIZEN(DETAIL) as it's handled specially above
            if descriptor == "TIZEN(DETAIL)":
                continue
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
        aggregated_results = {}  # Store aggregated results

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

                # Don't save raw responses as per requirements
                # Just collect them for processing
                pass

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
                    # Store results using only the data_type to avoid duplication
                    results[data_type] = parsed_data

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

        # Perform aggregations
        if results:
            logger.info("Performing aggregations")
            aggregated_results = self.perform_aggregations(results)

            # Save aggregated results to files
            for aggregation_type, aggregated_data in aggregated_results.items():
                if aggregated_data:
                    aggregation_file_path = self.save_data(aggregated_data, aggregation_type.lower())
                    if aggregation_file_path:
                        file_paths.append(aggregation_file_path)
                        logger.info(f"Saved {len(aggregated_data)} {aggregation_type} aggregated data points to {aggregation_file_path}")
                    else:
                        logger.warning(f"Failed to save {aggregation_type} aggregated data")
                else:
                    logger.warning(f"No {aggregation_type} aggregated data to save")

        # Save all data to a single file without timestamp
        if results:
            # Create file path without timestamp as per requirements
            all_data_file_name = f"user_{self.user_id}_all_data.json"
            all_data_file_path = os.path.join(RAW_DATA_DIR, all_data_file_name)

            try:
                # Save parsed data
                with open(all_data_file_path, 'w') as json_file:
                    json.dump(results, json_file, indent=4)

                file_paths.append(all_data_file_path)
                logger.info(f"Saved all parsed data to {all_data_file_path}")

                # Don't save raw responses as per requirements
                logger.info("Not saving raw responses as per requirements")
            except Exception as e:
                logger.error(f"Failed to save all data to file: {e}")
                logger.exception(e)  # Log the full exception traceback
        else:
            logger.warning(f"No data collected for user {self.user_id}")

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

            # For TIZEN(DETAIL) data with specific property key
            if game_descriptor == "TIZEN(DETAIL)" and property_key:
                for property_instance in data_point.get("propertyInstances", []):
                    prop_key = property_instance.get("property", {}).get("translationKey")
                    property_value = property_instance.get("value")

                    # Log the property key for debugging
                    logger.debug(f"Processing property with key: {prop_key}")

                    if prop_key == property_key:
                        try:
                            # Try to parse the property value as a Python literal
                            property_value_dict = ast.literal_eval(property_value)
                            logger.debug(f"Successfully parsed property value as: {type(property_value_dict)}")

                            # Handle different property keys differently
                            if property_key == "ACTIVITY_TYPE":
                                # For ACTIVITY_TYPE, append the parsed value to the data list
                                data_list.append(property_value_dict)
                                logger.debug("Added 1 item to data list")
                            elif property_key in ["HRM_LOG", "ACCELEROMETER_LOG", "PPG_LOG"]:
                                # For HRM_LOG, ACCELEROMETER_LOG, and PPG_LOG, extend the data list with the items in the parsed value
                                if isinstance(property_value_dict, list):
                                    data_list.extend(property_value_dict)
                                    logger.debug(f"Added {len(property_value_dict)} items to data list")
                                else:
                                    data_list.append(property_value_dict)
                                    logger.debug("Added 1 item to data list")
                            else:
                                # For other property keys, append the parsed value to the data list
                                data_list.append(property_value_dict)
                                logger.debug("Added 1 item to data list")
                        except (ValueError, SyntaxError) as e:
                            logger.error(f"Failed to parse property value: {e}")
                            continue
            else:
                # For other data types (GEOFENCE, LOG_MOOD, SELFREPORT, NOTIFICATION(DETAIL))
                data = {}
                for property_instance in data_point.get("propertyInstances", []):
                    prop_key = property_instance.get("property", {}).get("translationKey")
                    property_value = property_instance.get("value")
                    data[prop_key] = property_value

                # Special handling for GENERAL_ACTIVITY to ensure it has all required parameters
                if game_descriptor == "GENERAL_ACTIVITY":
                    # List of required parameters for GENERAL_ACTIVITY
                    required_params = [
                        "ACTIVITY_TYPE",
                        "DESCRIPTION",
                        "DURATION",
                        "SECRET",
                        "STEPS",
                        "STEPS_SUM",
                        "VIDEO",
                        "EMOTION_AFTER_ACTIVITY"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"GENERAL_ACTIVITY data after adding defaults: {data}")

                # Special handling for LOCATION to ensure it has all required parameters
                elif game_descriptor == "LOCATION":
                    # List of required parameters for LOCATION
                    required_params = [
                        "COORDINATES",
                        "DESCRIPTION",
                        "DURATION",
                        "JSON.GEO",
                        "LIKERT_SCALE_SIX",
                        "SECRET",
                        "VIDEO"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"LOCATION data after adding defaults: {data}")

                # Special handling for PHYSICAL_ACTIVITY to ensure it has all required parameters
                elif game_descriptor == "PHYSICAL_ACTIVITY":
                    # List of required parameters for PHYSICAL_ACTIVITY
                    required_params = [
                        "DESCRIPTION",
                        "DURATION",
                        "PA_INTENSITY",
                        "SECRET",
                        "SPORTS",
                        "VIDEO"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"PHYSICAL_ACTIVITY data after adding defaults: {data}")

                # Special handling for DAY_AGGREGATE to ensure it has all required parameters
                elif game_descriptor == "DAY_AGGREGATE":
                    # List of required parameters for DAY_AGGREGATE
                    required_params = [
                        "CONFIGURATION_NAME",
                        "START_DATE",
                        "END_DATE",
                        "STEPS_SUM",
                        "STEPS_AVG",
                        "STEPS_MAX",
                        "DISTANCE_SUM",
                        "DISTANCE_AVG",
                        "DISTANCE_MAX",
                        "DEVICE_TYPE"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"DAY_AGGREGATE data after adding defaults: {data}")

                # Special handling for DAY_AGGREGATE_WALK to ensure it has all required parameters
                elif game_descriptor == "DAY_AGGREGATE_WALK":
                    # List of required parameters for DAY_AGGREGATE_WALK
                    required_params = [
                        "CONFIGURATION_NAME",
                        "DESCRIPTION",
                        "DISTANCE_AVG",
                        "DISTANCE_MAX",
                        "DISTANCE_SUM",
                        "END_DATE",
                        "SECRET",
                        "START_DATE",
                        "STEPS_AVG",
                        "STEPS_MAX",
                        "STEPS_SUM",
                        "VIDEO"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"DAY_AGGREGATE_WALK data after adding defaults: {data}")

                # Special handling for DAY_AGGREGATE_RUN to ensure it has all required parameters
                elif game_descriptor == "DAY_AGGREGATE_RUN":
                    # List of required parameters for DAY_AGGREGATE_RUN
                    required_params = [
                        "CONFIGURATION_NAME",
                        "DESCRIPTION",
                        "DISTANCE_AVG",
                        "DISTANCE_MAX",
                        "DISTANCE_SUM",
                        "END_DATE",
                        "SECRET",
                        "START_DATE",
                        "STEPS_AVG",
                        "STEPS_MAX",
                        "STEPS_SUM",
                        "VIDEO"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"DAY_AGGREGATE_RUN data after adding defaults: {data}")

                # Special handling for LOG_MOOD to ensure it has all required parameters
                elif game_descriptor == "LOG_MOOD":
                    # List of required parameters for LOG_MOOD
                    required_params = [
                        "AROUSAL_STATE_VALUE",
                        "EVENT_AREA",
                        "EVENT_TIMESTAMP",
                        "INTOXICATION_LEVEL",
                        "PANAS_AFFECTIVE_EMOTIONAL_STATE_KEY",
                        "PANAS_AFFECTIVE_EMOTIONAL_STATE_VALUE",
                        "TRAFFIC_LIGHT_VALUE",
                        "VALENCE_STATE_VALUE"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"LOG_MOOD data after adding defaults: {data}")

                # Special handling for NOTIFICATION(DETAIL) to ensure it has all required parameters
                elif game_descriptor == "NOTIFICATION(DETAIL)":
                    # List of required parameters for NOTIFICATION(DETAIL)
                    required_params = [
                        "ACTION",
                        "EVENT_TIMESTAMP"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"NOTIFICATION(DETAIL) data after adding defaults: {data}")

                # Special handling for TIZEN_LOG(DETAIL) to ensure it has all required parameters
                elif game_descriptor == "TIZEN_LOG(DETAIL)":
                    # List of required parameters for TIZEN_LOG(DETAIL)
                    required_params = [
                        "TIZEN_LOG_PAYLOAD"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"TIZEN_LOG(DETAIL) data after adding defaults: {data}")

                # Special handling for HOW_ACTIVE_DO_YOU_FEEL_AT_THE_MOMENT to ensure it has all required parameters
                elif game_descriptor == "HOW_ACTIVE_DO_YOU_FEEL_AT_THE_MOMENT":
                    # List of required parameters for HOW_ACTIVE_DO_YOU_FEEL_AT_THE_MOMENT
                    required_params = [
                        "LIKERT_SCALE_FIVE"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"HOW_ACTIVE_DO_YOU_FEEL_AT_THE_MOMENT data after adding defaults: {data}")

                # Special handling for DO_LATER to ensure it has all required parameters
                elif game_descriptor == "DO_LATER":
                    # List of required parameters for DO_LATER
                    required_params = [
                        "LIKERT_SCALE_SEVEN"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"DO_LATER data after adding defaults: {data}")

                # Special handling for BODY_MOVEMENT to ensure it has all required parameters
                elif game_descriptor == "BODY_MOVEMENT":
                    # List of required parameters for BODY_MOVEMENT
                    required_params = [
                        "DUTCH_POLICE_MOVES",
                        "LIKERT_SCALE_THREE"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"BODY_MOVEMENT data after adding defaults: {data}")

                # Special handling for EMOTION_SCALE_FIVE to ensure it has all required parameters
                elif game_descriptor == "EMOTION_SCALE_FIVE":
                    # List of required parameters for EMOTION_SCALE_FIVE
                    required_params = [
                        "LIKERT_SCALE_FIVE"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"EMOTION_SCALE_FIVE data after adding defaults: {data}")

                # Special handling for RECENTLY_EATEN to ensure it has all required parameters
                elif game_descriptor == "RECENTLY_EATEN":
                    # List of required parameters for RECENTLY_EATEN
                    required_params = [
                        "LIKERT_SCALE_FOUR"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"RECENTLY_EATEN data after adding defaults: {data}")

                # Special handling for RECENTLY_COLD_BEVERAGE to ensure it has all required parameters
                elif game_descriptor == "RECENTLY_COLD_BEVERAGE":
                    # List of required parameters for RECENTLY_COLD_BEVERAGE
                    required_params = [
                        "LIKERT_SCALE_THREE"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"RECENTLY_COLD_BEVERAGE data after adding defaults: {data}")

                # Special handling for ELDERLY_ACTIVITY to ensure it has all required parameters
                elif game_descriptor == "ELDERLY_ACTIVITY":
                    # List of required parameters for ELDERLY_ACTIVITY
                    required_params = [
                        "CONTEXT",
                        "SPEED.AVG"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"ELDERLY_ACTIVITY data after adding defaults: {data}")

                # Special handling for VALENCE to ensure it has all required parameters
                elif game_descriptor == "VALENCE":
                    # List of required parameters for VALENCE
                    required_params = [
                        "LIKERT_SCALE_THREE"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"VALENCE data after adding defaults: {data}")

                # Special handling for AROUSAL to ensure it has all required parameters
                elif game_descriptor == "AROUSAL":
                    # List of required parameters for AROUSAL
                    required_params = [
                        "LIKERT_SCALE_THREE"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"AROUSAL data after adding defaults: {data}")

                # Special handling for LOG_AWARENESS to ensure it has all required parameters
                elif game_descriptor == "LOG_AWARENESS":
                    # List of required parameters for LOG_AWARENESS
                    required_params = [
                        "HEARTBEAT",
                        "LIGHT",
                        "MEETING",
                        "SMELL",
                        "SOUND",
                        "TASTE",
                        "TOUCH"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"LOG_AWARENESS data after adding defaults: {data}")

                # Special handling for SUBJECTIVE_MOOD to ensure it has all required parameters
                elif game_descriptor == "SUBJECTIVE_MOOD":
                    # List of required parameters for SUBJECTIVE_MOOD
                    required_params = [
                        "LIKERT_SCALE_TEN"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"SUBJECTIVE_MOOD data after adding defaults: {data}")

                # Special handling for SUBJECTIVE_EMOTION to ensure it has all required parameters
                elif game_descriptor == "SUBJECTIVE_EMOTION":
                    # List of required parameters for SUBJECTIVE_EMOTION
                    required_params = [
                        "LIKERT_SCALE_TEN"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"SUBJECTIVE_EMOTION data after adding defaults: {data}")

                # Special handling for LOCATION_FAMILIARITY to ensure it has all required parameters
                elif game_descriptor == "LOCATION_FAMILIARITY":
                    # List of required parameters for LOCATION_FAMILIARITY
                    required_params = [
                        "LIKERT_SCALE_FIVE"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"LOCATION_FAMILIARITY data after adding defaults: {data}")

                # Special handling for PEOPLE_COMPANY to ensure it has all required parameters
                elif game_descriptor == "PEOPLE_COMPANY":
                    # List of required parameters for PEOPLE_COMPANY
                    required_params = [
                        "LIKERT_SCALE_SIX"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"PEOPLE_COMPANY data after adding defaults: {data}")

                # Special handling for GEOFENCE to ensure it has all required parameters
                elif game_descriptor == "GEOFENCE":
                    # List of required parameters for GEOFENCE
                    required_params = [
                        "LATITUDE",
                        "LONGITUDE",
                        "ALTIDUDE",
                        "SPEED",
                        "ERROR",
                        "TIMESTAMP"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"GEOFENCE data after adding defaults: {data}")

                # Special handling for TIZEN(DETAIL) to ensure it has all required parameters
                elif game_descriptor == "TIZEN(DETAIL)":
                    # List of required parameters for TIZEN(DETAIL)
                    required_params = [
                        "ACCELEROMETER_LOG",
                        "ACTIVITY_TYPE",
                        "HRM_LOG",
                        "PPG_LOG"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"TIZEN(DETAIL) data after adding defaults: {data}")

                # Special handling for VISIT_APP_PAGE to ensure it has all required parameters
                elif game_descriptor == "VISIT_APP_PAGE":
                    # List of required parameters for VISIT_APP_PAGE
                    required_params = [
                        "APP_NAME",
                        "DURATION_MS",
                        "PAGE_NAME"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"VISIT_APP_PAGE data after adding defaults: {data}")

                # Special handling for PLAN_MEAL to ensure it has all required parameters
                elif game_descriptor == "PLAN_MEAL":
                    # List of required parameters for PLAN_MEAL
                    required_params = [
                        "MEAL_PLAN_JS",
                        "NR_DAYS",
                        "NR_MEALS"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"PLAN_MEAL data after adding defaults: {data}")

                # Special handling for NUTRITION_SUMMARY to ensure it has all required parameters
                elif game_descriptor == "NUTRITION_SUMMARY":
                    # List of required parameters for NUTRITION_SUMMARY
                    required_params = [
                        "AGGREGATION_LEVEL",
                        "CARBS_CONFORMANCE",
                        "FAT_CONFORMANCE",
                        "FIBER_CONFORMANCE",
                        "SUMMARY_SCORE"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"NUTRITION_SUMMARY data after adding defaults: {data}")

                # Special handling for NUTRITION_DIARY_VID to ensure it has all required parameters
                elif game_descriptor == "NUTRITION_DIARY_VID":
                    # List of required parameters for NUTRITION_DIARY_VID
                    required_params = [
                        "DESCRIPTION",
                        "VIDEO",
                        "IS_COMPLIANT",
                        "MEAL_CARB_WEIGHT_ESTIMATED_G",
                        "MEAL_TOTAL_WEIGHT_ESTIMATED_G",
                        "REF_TO_MEAL_PLAN",
                        "food_video"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"NUTRITION_DIARY_VID data after adding defaults: {data}")

                # Special handling for WALK to ensure it has all required parameters
                elif game_descriptor == "WALK":
                    # List of required parameters for WALK
                    required_params = [
                        "DESCRIPTION",
                        "DISTANCE",
                        "DURATION",
                        "KCALORIES",
                        "SECRET",
                        "SPEED.AVG",
                        "SPEED.MAX",
                        "STEPS",
                        "VIDEO",
                        "EMOTION_AFTER_ACTIVITY"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"WALK data after adding defaults: {data}")

                # Special handling for RUN to ensure it has all required parameters
                elif game_descriptor == "RUN":
                    # List of required parameters for RUN
                    required_params = [
                        "DESCRIPTION",
                        "DISTANCE",
                        "DURATION",
                        "KCALORIES",
                        "SECRET",
                        "SPEED.AVG",
                        "SPEED.MAX",
                        "STEPS",
                        "VIDEO",
                        "EMOTION_AFTER_ACTIVITY"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"RUN data after adding defaults: {data}")

                # Special handling for BIKE to ensure it has all required parameters
                elif game_descriptor == "BIKE":
                    # List of required parameters for BIKE
                    required_params = [
                        "DESCRIPTION",
                        "DISTANCE",
                        "DURATION",
                        "KCALORIES",
                        "SECRET",
                        "SPEED.AVG",
                        "VIDEO",
                        "EMOTION_AFTER_ACTIVITY"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"BIKE data after adding defaults: {data}")

                # Special handling for TRANSPORT to ensure it has all required parameters
                elif game_descriptor == "TRANSPORT":
                    # List of required parameters for TRANSPORT
                    required_params = [
                        "ALTITUDE.AVG",
                        "ALTITUDE.MAX",
                        "ALTITUDE.MIN",
                        "DESCRIPTION",
                        "DISTANCE",
                        "DURATION",
                        "JSONARRAY.GEO",
                        "SECRET",
                        "SPEED.AVG",
                        "SPEED.MAX",
                        "VIDEO"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"TRANSPORT data after adding defaults: {data}")

                # Special handling for SOCIAL_SELFIE to ensure it has all required parameters
                elif game_descriptor == "SOCIAL_SELFIE":
                    # List of required parameters for SOCIAL_SELFIE
                    required_params = [
                        "SECRET",
                        "DESCRIPTION",
                        "DURATION",
                        "GROUP_SIZE",
                        "COORDINATES",
                        "JSON.GEO",
                        "VIDEO"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"SOCIAL_SELFIE data after adding defaults: {data}")

                # Special handling for SCORE_GAMEBUS_POINTS to ensure it has all required parameters
                elif game_descriptor == "SCORE_GAMEBUS_POINTS":
                    # List of required parameters for SCORE_GAMEBUS_POINTS
                    required_params = [
                        "ACTIVITY",
                        "FOR_CHALLENGE",
                        "CHALLENGE_RULE",
                        "NUMBER_OF_POINTS"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"SCORE_GAMEBUS_POINTS data after adding defaults: {data}")

                # Special handling for NUTRITION_DIARY to ensure it has all required parameters
                elif game_descriptor == "NUTRITION_DIARY":
                    # List of required parameters for NUTRITION_DIARY
                    required_params = [
                        "SECRET",
                        "DESCRIPTION",
                        "MEAL_TYPE",
                        "FIBERS_WEIGHT",
                        "KCAL_CARB",
                        "KCAL_FATS",
                        "KCAL_PROTEINS",
                        "PERC_CARB",
                        "PERC_FATS",
                        "PERC_FIBERS",
                        "PERC_PROTEINS",
                        "VIDEO"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"NUTRITION_DIARY data after adding defaults: {data}")

                # Special handling for DRINKING_DIARY to ensure it has all required parameters
                elif game_descriptor == "DRINKING_DIARY":
                    # List of required parameters for DRINKING_DIARY
                    required_params = [
                        "SECRET",
                        "DESCRIPTION",
                        "RANGE_GLASS_ALC",
                        "RANGE_GLASS_SUGAR",
                        "VIDEO"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"DRINKING_DIARY data after adding defaults: {data}")

                # Special handling for CONSENT to ensure it has all required parameters
                elif game_descriptor == "CONSENT":
                    # List of required parameters for CONSENT
                    required_params = [
                        "DESCRIPTION",
                        "FOR_CAMPAIGN"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"CONSENT data after adding defaults: {data}")

                # Special handling for PLAY_LOTTERY to ensure it has all required parameters
                elif game_descriptor == "PLAY_LOTTERY":
                    # List of required parameters for PLAY_LOTTERY
                    required_params = [
                        "LOTTERY",
                        "FOR_CHALLENGE",
                        "WIN",
                        "REWARD"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"PLAY_LOTTERY data after adding defaults: {data}")

                # Special handling for NAVIGATE_APP to ensure it has all required parameters
                elif game_descriptor == "NAVIGATE_APP":
                    # List of required parameters for NAVIGATE_APP
                    required_params = [
                        "APP",
                        "FOR_CAMPAIGN",
                        "SESSION",
                        "UID",
                        "URI"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"NAVIGATE_APP data after adding defaults: {data}")

                # Special handling for H5P_GENERAL to ensure it has all required parameters
                elif game_descriptor == "H5P_GENERAL":
                    # List of required parameters for H5P_GENERAL
                    required_params = [
                        "SECRET",
                        "DESCRIPTION",
                        "DURATION",
                        "ACTIVITY_TYPE",
                        "MIN_SCORE",
                        "MAX_SCORE",
                        "SCORE",
                        "SCORE_RATIO"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"H5P_GENERAL data after adding defaults: {data}")

                # Special handling for GENERAL_SURVEY to ensure it has all required parameters
                elif game_descriptor == "GENERAL_SURVEY":
                    # List of required parameters for GENERAL_SURVEY
                    required_params = [
                        "SECRET",
                        "DESCRIPTION",
                        "GENERAL_SURVEY_ANSWER"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"GENERAL_SURVEY data after adding defaults: {data}")

                # Special handling for BODY_MASS_INDEX to ensure it has all required parameters
                elif game_descriptor == "BODY_MASS_INDEX":
                    # List of required parameters for BODY_MASS_INDEX
                    required_params = [
                        "SECRET",
                        "DURATION",
                        "AGE",
                        "BODY_MASS_INDEX",
                        "GENDER",
                        "LENGTH",
                        "WAIST_CIRCUMFERENCE",
                        "WEIGHT"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"BODY_MASS_INDEX data after adding defaults: {data}")

                # Special handling for MIPIP to ensure it has all required parameters
                elif game_descriptor == "MIPIP":
                    # List of required parameters for MIPIP
                    required_params = [
                        "SECRET",
                        "DURATION",
                        "MIPIP_AGREE_Q01",
                        "MIPIP_AGREE_Q02",
                        "MIPIP_AGREE_Q03",
                        "MIPIP_AGREE_Q04",
                        "MIPIP_AGREEABLENESS",
                        "MIPIP_CONSC_Q01",
                        "MIPIP_CONSC_Q02",
                        "MIPIP_CONSC_Q03",
                        "MIPIP_CONSC_Q04",
                        "MIPIP_CONSCIENTIOUSNESS",
                        "MIPIP_EXTRA_Q01",
                        "MIPIP_EXTRA_Q02",
                        "MIPIP_EXTRA_Q03",
                        "MIPIP_EXTRA_Q04",
                        "MIPIP_EXTRAVERSION",
                        "MIPIP_INTEL_Q01",
                        "MIPIP_INTEL_Q02",
                        "MIPIP_INTEL_Q03",
                        "MIPIP_INTEL_Q04",
                        "MIPIP_INTELLECT",
                        "MIPIP_NEURO_Q01",
                        "MIPIP_NEURO_Q02",
                        "MIPIP_NEURO_Q03",
                        "MIPIP_NEURO_Q04",
                        "MIPIP_NEUROTICISM"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"MIPIP data after adding defaults: {data}")

                # Special handling for WALK_PERFORMANCE to ensure it has all required parameters
                elif game_descriptor == "WALK_PERFORMANCE":
                    # List of required parameters for WALK_PERFORMANCE
                    required_params = [
                        "SECRET",
                        "DURATION",
                        "WALK_PERFORMANCE_BASELINE",
                        "WALK_PERFORMANCE_GOAL",
                        "WALK_PERFORMANCE_GROW"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"WALK_PERFORMANCE data after adding defaults: {data}")

                # Special handling for BIKE_PERFORMANCE to ensure it has all required parameters
                elif game_descriptor == "BIKE_PERFORMANCE":
                    # List of required parameters for BIKE_PERFORMANCE
                    required_params = [
                        "SECRET",
                        "DURATION",
                        "BIKE_PERFORMANCE_BASELINE",
                        "BIKE_PERFORMANCE_GOAL",
                        "BIKE_PERFORMANCE_GROW"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"BIKE_PERFORMANCE data after adding defaults: {data}")

                # Special handling for SPORTS_PERFORMANCE to ensure it has all required parameters
                elif game_descriptor == "SPORTS_PERFORMANCE":
                    # List of required parameters for SPORTS_PERFORMANCE
                    required_params = [
                        "SECRET",
                        "DURATION",
                        "SPORTS_PERFORMANCE_BASELINE",
                        "SPORTS_PERFORMANCE_GOAL",
                        "SPORTS_PERFORMANCE_GROW"
                    ]

                    # Add default values for any missing parameters
                    for param in required_params:
                        if param not in data:
                            logger.info(f"Adding default value for missing parameter: {param}")
                            data[param] = ""  # Default empty string for missing parameters
                        elif data[param] is None:
                            logger.info(f"Parameter {param} is None, setting to empty string")
                            data[param] = ""  # Replace None with empty string

                    # Log the final data for debugging
                    logger.info(f"SPORTS_PERFORMANCE data after adding defaults: {data}")

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

        # Create directory if it doesn't exist
        os.makedirs(RAW_DATA_DIR, exist_ok=True)

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

    def perform_aggregations(self, results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform aggregations on the collected data.

        Args:
            results: Dictionary of data points by type

        Returns:
            Dictionary of aggregated data by aggregation type
        """
        logger.info("Performing aggregations on collected data")
        aggregated_results = {}

        # Process each aggregation type
        for aggregation_type, config in self.aggregation_configs.items():
            logger.info(f"Processing {aggregation_type} aggregation")
            activity_scheme = config['activity_scheme']
            transformations = config['transformations']

            # Collect all data points for the activities in the scheme
            activity_data = []
            for activity in activity_scheme:
                activity_lower = activity.lower()
                if activity_lower in results:
                    activity_data.extend(results[activity_lower])
                    logger.info(f"Added {len(results[activity_lower])} data points from {activity}")
                else:
                    logger.warning(f"No data found for {activity} in results")

            if not activity_data:
                logger.warning(f"No data found for any activity in {aggregation_type} scheme: {activity_scheme}")
                continue

            # Determine if this is a daily or weekly aggregation
            is_daily = aggregation_type.startswith('DAILY_')
            period_type = 'daily' if is_daily else 'weekly'
            logger.info(f"Performing {period_type} aggregation for {aggregation_type}")

            # Group data by day or week
            grouped_data = self._group_data_by_period(activity_data, period_type)
            logger.info(f"Grouped data into {len(grouped_data)} {period_type} periods")

            # Apply transformations to each group
            aggregated_data = []
            for period, period_data in grouped_data.items():
                aggregated_point = {
                    'period': period,
                    'period_type': period_type,
                    'activity_scheme': ', '.join(activity_scheme)
                }

                # Apply each transformation
                for transform in transformations:
                    operation = transform['operation']
                    source = transform['source']
                    target = transform['target']

                    # Extract values for the source field
                    values = []
                    for data_point in period_data:
                        if source in data_point:
                            try:
                                value = float(data_point[source])
                                values.append(value)
                            except (ValueError, TypeError):
                                logger.warning(f"Could not convert {source} value '{data_point[source]}' to float, skipping")

                    # Apply the operation
                    if values:
                        if operation == 'SUM':
                            aggregated_point[target] = sum(values)
                        elif operation == 'AVERAGE':
                            aggregated_point[target] = sum(values) / len(values)
                        elif operation == 'MAXIMUM':
                            aggregated_point[target] = max(values)
                        else:
                            logger.warning(f"Unknown operation: {operation}, skipping")
                    else:
                        logger.warning(f"No valid values found for {source}, skipping {operation} operation")
                        aggregated_point[target] = 0  # Default value

                aggregated_data.append(aggregated_point)

            logger.info(f"Created {len(aggregated_data)} aggregated data points for {aggregation_type}")
            aggregated_results[aggregation_type] = aggregated_data

        return aggregated_results

    def _group_data_by_period(self, data: List[Dict[str, Any]], period_type: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group data by day or week.

        Args:
            data: List of data points
            period_type: 'daily' or 'weekly'

        Returns:
            Dictionary of data points grouped by period
        """
        import datetime

        grouped_data = {}

        for data_point in data:
            # Try to get the date from the data point
            date_str = data_point.get('date') or data_point.get('timestamp') or data_point.get('EVENT_TIMESTAMP')

            if not date_str:
                logger.warning(f"No date found in data point, skipping: {data_point}")
                continue

            try:
                # Try to parse the date string
                # First try ISO format
                try:
                    date = datetime.datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    # Then try timestamp format
                    try:
                        date = datetime.datetime.fromtimestamp(float(date_str) / 1000)
                    except (ValueError, TypeError):
                        # Finally try a generic format
                        date = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            except Exception as e:
                logger.warning(f"Failed to parse date '{date_str}': {e}, skipping data point")
                continue

            # Determine the period key
            if period_type == 'daily':
                period_key = date.strftime("%Y-%m-%d")
            else:  # weekly
                # Get the week number and year
                year, week, _ = date.isocalendar()
                period_key = f"{year}-W{week:02d}"

            # Add the data point to the appropriate period
            if period_key not in grouped_data:
                grouped_data[period_key] = []
            grouped_data[period_key].append(data_point)

        return grouped_data

    def save_data(self, data: List[Dict[str, Any]], data_type: str) -> str:
        """
        Save data to a JSON file.

        Args:
            data: Data to save
            data_type: Type of data being saved

        Returns:
            Path to the saved JSON file
        """
        # Create directory if it doesn't exist
        os.makedirs(RAW_DATA_DIR, exist_ok=True)

        # Create file path without timestamp as per requirements
        json_file_name = f"user_{self.user_id}_{data_type}.json"
        json_file_path = os.path.join(RAW_DATA_DIR, json_file_name)

        # Save data to JSON file
        try:
            with open(json_file_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)

            logger.info(f"Saved {len(data)} {data_type} data points to {json_file_path}")
            return json_file_path
        except Exception as e:
            logger.error(f"Failed to save data to file: {e}")
            return ""
