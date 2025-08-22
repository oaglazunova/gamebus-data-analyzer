import json
import os
import random
import datetime
import sys
from typing import Dict, List, Any, Tuple
import string
import pandas as pd

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the list of valid game descriptors and their parameters
from config.settings import VALID_GAME_DESCRIPTORS
from config.paths import RAW_DATA_DIR, USERS_FILE_PATH, CONFIG_DIR
from config.property_schemes import ACTIVITY_TO_PROPERTIES, PROPERTY_DETAILS

# Ensure the data directory exists
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Constants
DEFAULT_NUM_USERS = 10  # Default number of users if users.xlsx is not found
NUM_DATA_POINTS = 20    # Number of data points to generate for each game descriptor
# Time span for generated data (in days)
TIME_SPAN_DAYS = 30

def load_property_schemes() -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, Any]]]:
    """
    Load property schemes from the config file.

    Returns:
        Tuple containing:
        - Dictionary mapping activity schemes to their property schemes
        - Dictionary mapping property names to their details (type, unit, etc.)
    """
    print("Loading property schemes from config file")

    # The property schemes are already imported from the config file
    print(f"Loaded {len(ACTIVITY_TO_PROPERTIES)} activity schemes and {len(PROPERTY_DETAILS)} property schemes")
    return ACTIVITY_TO_PROPERTIES, PROPERTY_DETAILS

def generate_random_value(param_name: str, property_details: Dict[str, Dict[str, Any]] = None) -> Any:
    """
    Generate a random value for a parameter based on its name and property details.

    Args:
        param_name: Name of the parameter
        property_details: Dictionary mapping property names to their details

    Returns:
        Random value appropriate for the parameter
    """
    # Check if we have property details for this parameter
    if property_details and param_name in property_details:
        details = property_details[param_name]
        prop_type = details['type']
        prop_unit = details['unit']
        prop_values = details['values']
        prop_input_as = details['input_as']

        # Generate value based on property type
        if prop_type == 'BOOL':
            return random.choice([True, False])

        elif prop_type == 'STRING':
            if prop_values:
                # If there are predefined values, choose one
                try:
                    values_list = prop_values.split(',')
                    return random.choice(values_list).strip()
                except:
                    pass
            # Otherwise generate a random string
            return ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(10, 30)))

        elif prop_type == 'INT':
            # Try to extract range from unit if available
            if prop_unit and '[' in prop_unit and ']' in prop_unit:
                try:
                    range_str = prop_unit[prop_unit.find('[')+1:prop_unit.find(']')]
                    min_val, max_val = map(int, range_str.split('-'))
                    return random.randint(min_val, max_val)
                except:
                    pass
            # Default integer range
            return random.randint(1, 100)

        elif prop_type == 'DOUBLE':
            # Try to extract range from unit if available
            if prop_unit and '[' in prop_unit and ']' in prop_unit:
                try:
                    range_str = prop_unit[prop_unit.find('[')+1:prop_unit.find(']')]
                    min_val, max_val = map(float, range_str.split('-'))
                    return round(random.uniform(min_val, max_val), 2)
                except:
                    pass
            # Default double range
            return round(random.uniform(0.0, 100.0), 2)

        elif prop_type == 'COORDINATE':
            # Generate a random coordinate
            return round(random.uniform(-90.0, 90.0), 6)

        elif prop_type == 'ACTIVITY':
            # Return a random activity ID
            return random.randint(1, 1000)

        elif prop_type == 'DATETIME':
            # Generate a random date/time within the specified time span
            days_ago = random.randint(0, TIME_SPAN_DAYS)
            hours_ago = random.randint(0, 23)
            minutes_ago = random.randint(0, 59)
            seconds_ago = random.randint(0, 59)

            dt = datetime.datetime.now() - datetime.timedelta(
                days=days_ago, 
                hours=hours_ago,
                minutes=minutes_ago,
                seconds=seconds_ago
            )

            # Return as ISO format string
            return dt.isoformat()

    # If no property details or not found in property details, fall back to name-based generation
    # Handle specific parameter types based on name patterns
    if param_name in ["SECRET", "DESCRIPTION"]:
        # Generate a random string
        return ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(10, 30)))

    elif "TIMESTAMP" in param_name or "DATE" in param_name:
        # Generate a random date/time within the specified time span
        days_ago = random.randint(0, TIME_SPAN_DAYS)
        hours_ago = random.randint(0, 23)
        minutes_ago = random.randint(0, 59)
        seconds_ago = random.randint(0, 59)

        dt = datetime.datetime.now() - datetime.timedelta(
            days=days_ago, 
            hours=hours_ago,
            minutes=minutes_ago,
            seconds=seconds_ago
        )

        # Return as ISO format string
        return dt.isoformat()

    elif "DURATION" in param_name:
        # Duration in seconds
        return random.randint(60, 3600)

    elif "DISTANCE" in param_name:
        # Distance in meters
        return random.randint(100, 10000)

    elif "STEPS" in param_name:
        # Number of steps
        return random.randint(1000, 20000)

    elif "SPEED" in param_name:
        # Speed in km/h
        return round(random.uniform(1.0, 30.0), 2)

    elif "KCAL" in param_name or "CAL" in param_name:
        # Calories
        return random.randint(50, 1000)

    elif "WEIGHT" in param_name:
        # Weight in kg
        return round(random.uniform(50.0, 100.0), 1)

    elif "LENGTH" in param_name or "HEIGHT" in param_name:
        # Height in cm
        return random.randint(150, 200)

    elif "AGE" in param_name:
        # Age in years
        return random.randint(18, 80)

    elif "GENDER" in param_name:
        # Gender
        return random.choice(["MALE", "FEMALE", "OTHER"])

    elif "LIKERT_SCALE" in param_name:
        # Extract the scale from the parameter name (e.g., LIKERT_SCALE_FIVE -> 5)
        scale_map = {
            "THREE": 3,
            "FOUR": 4,
            "FIVE": 5,
            "SIX": 6,
            "SEVEN": 7,
            "TEN": 10
        }

        # Find which scale is mentioned in the parameter name
        for scale_name, scale_value in scale_map.items():
            if scale_name in param_name:
                return random.randint(1, scale_value)

        # Default to a 5-point scale if no specific scale is found
        return random.randint(1, 5)

    elif "COORDINATES" in param_name or "LATITUDE" in param_name:
        # Latitude
        return round(random.uniform(-90.0, 90.0), 6)

    elif "LONGITUDE" in param_name:
        # Longitude
        return round(random.uniform(-180.0, 180.0), 6)

    elif "ALTITUDE" in param_name:
        # Altitude in meters
        return random.randint(0, 5000)

    elif "GROUP_SIZE" in param_name:
        # Number of people in a group
        return random.randint(2, 20)

    elif "SCORE" in param_name or "POINTS" in param_name:
        # Score or points
        return random.randint(0, 100)

    elif "EMOTION" in param_name:
        # Emotion
        emotions = ["HAPPY", "SAD", "ANGRY", "EXCITED", "CALM", "TIRED", "STRESSED", "RELAXED"]
        return random.choice(emotions)

    elif "ACTIVITY_TYPE" in param_name:
        # Activity type
        activities = ["WALKING", "RUNNING", "CYCLING", "SWIMMING", "YOGA", "WEIGHT_TRAINING", "HIKING"]
        return random.choice(activities)

    elif "MEAL_TYPE" in param_name:
        # Meal type
        meal_types = ["BREAKFAST", "LUNCH", "DINNER", "SNACK"]
        return random.choice(meal_types)

    elif "PERC" in param_name:
        # Percentage
        return round(random.uniform(0.0, 100.0), 2)

    elif "JSON" in param_name or "JSONARRAY" in param_name:
        # JSON data
        return json.dumps({
            "key1": random.randint(1, 100),
            "key2": ''.join(random.choices(string.ascii_letters, k=10)),
            "key3": [random.randint(1, 10) for _ in range(5)]
        })

    elif "VIDEO" in param_name:
        # URL to a video
        return f"https://example.com/videos/{random.randint(1000, 9999)}.mp4"

    elif "FOR_CHALLENGE" in param_name or "FOR_CAMPAIGN" in param_name:
        # Boolean value
        return random.choice(["true", "false"])

    elif "WIN" in param_name:
        # Boolean value
        return random.choice(["true", "false"])

    elif "REWARD" in param_name:
        # Reward description
        rewards = ["POINTS", "BADGE", "TROPHY", "CERTIFICATE", "DISCOUNT"]
        return random.choice(rewards)

    elif "ACTION" in param_name:
        # Action type
        actions = ["RECEIVED", "READ", "CLICKED", "DISMISSED"]
        return random.choice(actions)

    elif "ERROR" in param_name:
        # Error margin (for geofence data)
        return round(random.uniform(0.1, 10.0), 2)

    # Default case: generate a random integer
    return random.randint(1, 100)

def generate_data_for_game_descriptor(game_descriptor: str, params_comment: str, property_details: Dict[str, Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Generate random data for a game descriptor.

    Args:
        game_descriptor: The game descriptor
        params_comment: Comment containing the parameters for this game descriptor
        property_details: Dictionary mapping property names to their details

    Returns:
        List of data points for this game descriptor
    """
    # Extract parameters from the comment
    # The comment format is like: "# PARAM1, PARAM2, PARAM3"
    params = [param.strip() for param in params_comment.split(',')]

    # Generate multiple data points for this game descriptor
    data_points = []

    # Calculate how many data points of each type to generate
    regular_points = int(NUM_DATA_POINTS * 0.7)  # 70% regular data
    missing_value_points = int(NUM_DATA_POINTS * 0.1)  # 10% with missing values
    extreme_value_points = int(NUM_DATA_POINTS * 0.1)  # 10% with extreme values
    special_char_points = int(NUM_DATA_POINTS * 0.1)  # 10% with special characters

    # Ensure we generate at least the requested number of data points
    total_points = regular_points + missing_value_points + extreme_value_points + special_char_points
    if total_points < NUM_DATA_POINTS:
        regular_points += (NUM_DATA_POINTS - total_points)

    # Generate regular data points
    for _ in range(regular_points):
        data_point = {}
        for param in params:
            data_point[param] = generate_random_value(param, property_details)
        data_points.append(data_point)

    # Generate data points with missing values
    for _ in range(missing_value_points):
        data_point = {}
        for param in params:
            # Randomly skip some parameters (20% chance)
            if random.random() > 0.2:
                data_point[param] = generate_random_value(param, property_details)
        data_points.append(data_point)

    # Generate data points with extreme values
    for _ in range(extreme_value_points):
        data_point = {}
        for param in params:
            # For numeric parameters, sometimes use extreme values
            if "DURATION" in param or "DISTANCE" in param or "STEPS" in param or "SPEED" in param or "KCAL" in param:
                if random.random() > 0.5:
                    # Use an extreme value (very high)
                    data_point[param] = random.randint(100000, 1000000)
                else:
                    data_point[param] = generate_random_value(param, property_details)
            else:
                data_point[param] = generate_random_value(param, property_details)
        data_points.append(data_point)

    # Generate data points with special characters
    for _ in range(special_char_points):
        data_point = {}
        for param in params:
            if "DESCRIPTION" in param or "SECRET" in param:
                if random.random() > 0.5:
                    # Use special characters
                    special_chars = "!@#$%^&*()_+-=[]{}|;':\",./<>?\\`~"
                    data_point[param] = ''.join(random.choices(string.ascii_letters + string.digits + special_chars, k=random.randint(10, 30)))
                else:
                    data_point[param] = generate_random_value(param, property_details)
            else:
                data_point[param] = generate_random_value(param, property_details)
        data_points.append(data_point)

    return data_points

def main():
    """
    Generate test data for multiple users with all game descriptors and their parameters.
    If users.xlsx exists in the config directory, it will be used to get user information (email, password, ID).
    The script will generate test data for each user in the Excel file.
    If the Excel file doesn't exist or doesn't contain the required columns, default user IDs will be used.
    For filenames, user IDs are extracted from campaign_data.xlsx, column pid.
    """
    # Define the path to campaign_data.xlsx
    campaign_data_path = os.path.join(CONFIG_DIR, 'campaign_data.xlsx')

    # Extract unique user IDs from campaign_data.xlsx, column pid
    campaign_user_ids = []
    if os.path.exists(campaign_data_path):
        try:
            # Try to read the 'activities' sheet from campaign_data.xlsx
            xl = pd.ExcelFile(campaign_data_path)
            if 'activities' in xl.sheet_names:
                activities_df = pd.read_excel(campaign_data_path, sheet_name='activities')
                if 'pid' in activities_df.columns:
                    # Extract unique user IDs from the 'pid' column
                    campaign_user_ids = activities_df['pid'].unique().tolist()
                    print(f"Extracted {len(campaign_user_ids)} unique user IDs from campaign_data.xlsx, column pid")
                else:
                    print(f"Column 'pid' not found in the 'activities' sheet of {campaign_data_path}")
            else:
                print(f"Sheet 'activities' not found in {campaign_data_path}")
        except Exception as e:
            print(f"Error reading {campaign_data_path}: {e}")
    else:
        print(f"{campaign_data_path} not found")

    # Check if users.xlsx exists
    users = []
    user_ids = []

    if os.path.exists(USERS_FILE_PATH):
        try:
            # Read user data from Excel file
            df = pd.read_excel(USERS_FILE_PATH)

            # Check if required columns exist
            required_columns = ['email', 'password']
            if all(col in df.columns for col in required_columns):
                # Create a list of user dictionaries with all available information
                for _, row in df.iterrows():
                    user = {
                        'email': row['email'],
                        'password': row['password']
                    }

                    # Add UserID if available (could be 'UserID', 'id', or 'user_id')
                    if 'UserID' in df.columns:
                        user['id'] = row['UserID']
                    elif 'id' in df.columns:
                        user['id'] = row['id']
                    elif 'user_id' in df.columns:
                        user['id'] = row['user_id']
                    else:
                        # Generate a default ID if none is provided
                        user['id'] = len(users) + 1

                    users.append(user)
                    user_ids.append(user['id'])

                print(f"Found {len(users)} users in {USERS_FILE_PATH}")
            else:
                missing_columns = [col for col in required_columns if col not in df.columns]
                print(f"Missing required columns in {USERS_FILE_PATH}: {missing_columns}")
                print(f"Using default user IDs")
                user_ids = list(range(1, DEFAULT_NUM_USERS + 1))
        except Exception as e:
            print(f"Error reading {USERS_FILE_PATH}: {e}")
            print(f"Using default user IDs")
            user_ids = list(range(1, DEFAULT_NUM_USERS + 1))
    else:
        print(f"{USERS_FILE_PATH} not found, using default user IDs")
        user_ids = list(range(1, DEFAULT_NUM_USERS + 1))

    print(f"Generating test data for {len(user_ids)} users...")

    # Load property schemes from the config file
    activity_to_properties, property_details = load_property_schemes()

    # Define the parameters for each game descriptor
    # First try to use the property schemes from the config file
    # If not available, fall back to the hardcoded parameters
    game_descriptor_params = {}

    # Define the hardcoded parameters as a fallback
    hardcoded_params = {
        "GENERAL_ACTIVITY": "ACTIVITY_TYPE, SECRET, DESCRIPTION, DURATION, STEPS, STEPS_SUM, VIDEO, EMOTION_AFTER_ACTIVITY",
        "WALK": "DESCRIPTION, DISTANCE, DURATION, KCALORIES, SECRET, SPEED.AVG, SPEED.MAX, STEPS, VIDEO, EMOTION_AFTER_ACTIVITY",
        "RUN": "DESCRIPTION, DISTANCE, DURATION, KCALORIES, SECRET, SPEED.AVG, SPEED.MAX, STEPS, VIDEO, EMOTION_AFTER_ACTIVITY",
        "BIKE": "DESCRIPTION, DISTANCE, DURATION, KCALORIES, SECRET, SPEED.AVG, VIDEO, EMOTION_AFTER_ACTIVITY",
        "TRANSPORT": "ALTITUDE.AVG, ALTITUDE.MAX, ALTITUDE.MIN, DESCRIPTION, DISTANCE, DURATION, JSONARRAY.GEO, SECRET, SPEED.AVG, SPEED.MAX, VIDEO",
        "LOCATION": "SECRET, DESCRIPTION, DURATION, COORDINATES, JSON.GEO, VIDEO, LIKERT_SCALE_SIX",
        "SOCIAL_SELFIE": "SECRET, DESCRIPTION, DURATION, GROUP_SIZE, COORDINATES, JSON.GEO, VIDEO",
        "PHYSICAL_ACTIVITY": "SECRET, DESCRIPTION, DURATION, PA_INTENSITY, SPORTS, VIDEO",
        "SCORE_GAMEBUS_POINTS": "ACTIVITY, FOR_CHALLENGE, CHALLENGE_RULE, NUMBER_OF_POINTS",
        "DAY_AGGREGATE": "CONFIGURATION_NAME, START_DATE, END_DATE, STEPS_SUM, STEPS_AVG, STEPS_MAX, DISTANCE_SUM, DISTANCE_AVG, DISTANCE_MAX, DEVICE_TYPE, PA_DISTANCE_SUM, PA_CAL_SUM",
        "DAY_AGGREGATE_WALK": "SECRET, DESCRIPTION, CONFIGURATION_NAME, START_DATE, END_DATE, STEPS_SUM, STEPS_AVG, STEPS_MAX, DISTANCE_SUM, DISTANCE_AVG, DISTANCE_MAX, VIDEO",
        "DAY_AGGREGATE_RUN": "SECRET, DESCRIPTION, CONFIGURATION_NAME, START_DATE, END_DATE, STEPS_SUM, STEPS_AVG, STEPS_MAX, DISTANCE_SUM, DISTANCE_AVG, DISTANCE_MAX, VIDEO",
        "NUTRITION_DIARY": "SECRET, DESCRIPTION, MEAL_TYPE, FIBERS_WEIGHT, KCAL_CARB, KCAL_FATS, KCAL_PROTEINS, PERC_CARB, PERC_FATS, PERC_FIBERS, PERC_PROTEINS, VIDEO",
        "DRINKING_DIARY": "SECRET, DESCRIPTION, RANGE_GLASS_ALC, RANGE_GLASS_SUGAR, VIDEO",
        "CONSENT": "DESCRIPTION, FOR_CAMPAIGN",
        "PLAY_LOTTERY": "LOTTERY, FOR_CHALLENGE, WIN, REWARD",
        "NAVIGATE_APP": "APP, FOR_CAMPAIGN, SESSION, UID, URI",
        "H5P_GENERAL": "SECRET, DESCRIPTION, DURATION, ACTIVITY_TYPE, MIN_SCORE, MAX_SCORE, SCORE, SCORE_RATIO",
        "GENERAL_SURVEY": "SECRET, DESCRIPTION, GENERAL_SURVEY_ANSWER",
        "BODY_MASS_INDEX": "SECRET, DURATION, AGE, BODY_MASS_INDEX, GENDER, LENGTH, WAIST_CIRCUMFERENCE, WEIGHT",
        "MIPIP": "SECRET, DURATION, MIPIP_AGREE_Q01, MIPIP_AGREE_Q02, MIPIP_AGREE_Q03, MIPIP_AGREE_Q04, MIPIP_AGREEABLENESS, MIPIP_CONSC_Q01, MIPIP_CONSC_Q02, MIPIP_CONSC_Q03, MIPIP_CONSC_Q04, MIPIP_CONSCIENTIOUSNESS, MIPIP_EXTRA_Q01, MIPIP_EXTRA_Q02, MIPIP_EXTRA_Q03, MIPIP_EXTRA_Q04, MIPIP_EXTRAVERSION, MIPIP_INTEL_Q01, MIPIP_INTEL_Q02, MIPIP_INTEL_Q03, MIPIP_INTEL_Q04, MIPIP_INTELLECT, MIPIP_NEURO_Q01, MIPIP_NEURO_Q02, MIPIP_NEURO_Q03, MIPIP_NEURO_Q04, MIPIP_NEUROTICISM",
        "WALK_PERFORMANCE": "SECRET, DURATION, WALK_PERFORMANCE_BASELINE, WALK_PERFORMANCE_GOAL, WALK_PERFORMANCE_GROW",
        "BIKE_PERFORMANCE": "SECRET, DURATION, BIKE_PERFORMANCE_BASELINE, BIKE_PERFORMANCE_GOAL, BIKE_PERFORMANCE_GROW",
        "SPORTS_PERFORMANCE": "SECRET, DURATION, SPORTS_PERFORMANCE_BASELINE, SPORTS_PERFORMANCE_GOAL, SPORTS_PERFORMANCE_GROW",
        "LOG_MOOD": "AROUSAL_STATE_VALUE, EVENT_AREA, EVENT_TIMESTAMP, INTOXICATION_LEVEL, PANAS_AFFECTIVE_EMOTIONAL_STATE_KEY, PANAS_AFFECTIVE_EMOTIONAL_STATE_VALUE, TRAFFIC_LIGHT_VALUE, VALENCE_STATE_VALUE",
        "NOTIFICATION(DETAIL)": "ACTION, EVENT_TIMESTAMP",
        "TIZEN_LOG(DETAIL)": "TIZEN_LOG_PAYLOAD",
        "HOW_ACTIVE_DO_YOU_FEEL_AT_THE_MOMENT": "LIKERT_SCALE_FIVE",
        "DO_LATER": "LIKERT_SCALE_SEVEN",
        "BODY_MOVEMENT": "DUTCH_POLICE_MOVES, LIKERT_SCALE_THREE",
        "EMOTION_SCALE_FIVE": "LIKERT_SCALE_FIVE",
        "RECENTLY_EATEN": "LIKERT_SCALE_FOUR",
        "RECENTLY_COLD_BEVERAGE": "LIKERT_SCALE_THREE",
        "ELDERLY_ACTIVITY": "CONTEXT, SPEED.AVG",
        "VALENCE": "LIKERT_SCALE_THREE",
        "AROUSAL": "LIKERT_SCALE_THREE",
        "LOG_AWARENESS": "HEARTBEAT, LIGHT, MEETING, SMELL, SOUND, TASTE, TOUCH",
        "SUBJECTIVE_MOOD": "LIKERT_SCALE_TEN",
        "SUBJECTIVE_EMOTION": "LIKERT_SCALE_TEN",
        "LOCATION_FAMILIARITY": "LIKERT_SCALE_FIVE",
        "PEOPLE_COMPANY": "LIKERT_SCALE_SIX",
        "VISIT_APP_PAGE": "APP_NAME, PAGE_NAME, DURATION_MS",
        "PLAN_MEAL": "NR_DAYS, NR_MEALS, MEAL_PLAN_JS",
        "NUTRITION_SUMMARY": "AGGREGATION_LEVEL, SUMMARY_SCORE, CARBS_CONFORMANCE, FAT_CONFORMANCE, FIBER_CONFORMANCE",
        "NUTRITION_DIARY_VID": "DESCRIPTION, IS_COMPLIANT, REF_TO_MEAL_PLAN, MEAL_TOTAL_WEIGHT_ESTIMATED_G, MEAL_CARB_WEIGHT_ESTIMATED_G, VIDEO",
        "GEOFENCE": "LATITUDE, LONGITUDE, ALTITUDE, SPEED, ERROR, TIMESTAMP"
    }

    # Use property schemes from Excel if available, otherwise use hardcoded params
    for game_descriptor in VALID_GAME_DESCRIPTORS:
        if activity_to_properties and game_descriptor in activity_to_properties:
            # Use property schemes from Excel
            properties = activity_to_properties[game_descriptor]
            game_descriptor_params[game_descriptor] = ", ".join(properties)
            print(f"Using Excel property schemes for {game_descriptor}: {properties}")
        elif game_descriptor in hardcoded_params:
            # Use hardcoded parameters
            game_descriptor_params[game_descriptor] = hardcoded_params[game_descriptor]
            print(f"Using hardcoded parameters for {game_descriptor}")
        else:
            # For game descriptors without defined parameters, use a default set of parameters
            default_params = "SECRET, DESCRIPTION, DURATION, TIMESTAMP"
            game_descriptor_params[game_descriptor] = default_params
            print(f"No parameters found for {game_descriptor}, using default parameters: {default_params}")

    print(f"Defined parameters for {len(game_descriptor_params)} game descriptors")

    # Use campaign user IDs for filenames if available
    filename_user_ids = campaign_user_ids if campaign_user_ids else user_ids

    # Ensure user ID 107 is included for the example geofence data
    if 107 not in filename_user_ids:
        filename_user_ids.append(107)
        print(f"Added user ID 107 for example geofence data")

    print(f"Using {len(filename_user_ids)} unique user IDs from campaign_data.xlsx for filenames")

    # Generate data for each filename user ID
    for filename_user_id in filename_user_ids:
        print(f"Generating data for user ID {filename_user_id}...")

        # Dictionary to store all data for this user
        user_data = {}

        # Generate data for each game descriptor
        for game_descriptor, params_comment in game_descriptor_params.items():
            print(f"  Generating data for game descriptor: {game_descriptor}")

            # Generate data for this game descriptor
            data_points = generate_data_for_game_descriptor(game_descriptor, params_comment, property_details)

            # Store the data points
            user_data[game_descriptor.lower()] = data_points

            # Save the data points to a separate file
            file_name = f"user_{filename_user_id}_{game_descriptor.lower()}.json"
            file_path = os.path.join(RAW_DATA_DIR, file_name)

            with open(file_path, 'w') as json_file:
                json.dump(data_points, json_file, indent=4)

            print(f"  Saved {len(data_points)} data points to {file_path}")

        # Create an all_data file that combines data from all game descriptors
        all_data_file_name = f"user_{filename_user_id}_all_data.json"
        all_data_file_path = os.path.join(RAW_DATA_DIR, all_data_file_name)

        with open(all_data_file_path, 'w') as json_file:
            json.dump(user_data, json_file, indent=4)

        print(f"  Saved combined data for all game descriptors to {all_data_file_path}")

    print("Data generation complete!")

if __name__ == "__main__":
    main()
