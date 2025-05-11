"""
General settings for the GameBus-HealthBehaviorMining project.
"""

# GameBus API configuration
DEFAULT_PAGE_SIZE = 50
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30  # seconds

# Data collection settings
VALID_GAME_DESCRIPTORS = [
    "GEOFENCE",           # Location data
    "LOG_MOOD",           # Mood logging data
    "TIZEN(DETAIL)",      # Watch/wearable data
    "NOTIFICATION(DETAIL)",  # Notification data
    "SELFREPORT",         # Self-reported data
    "STEPS",              # Step count data
    "SLEEP",              # Sleep data
    "WEIGHT",             # Weight data
    "BLOOD_PRESSURE",     # Blood pressure data
    "HEART_RATE",         # Heart rate data
    "CALORIES",           # Calories data
    "DISTANCE",           # Distance data
    "FLOORS",             # Floors data
    "ELEVATION",          # Elevation data
    "ACTIVE_MINUTES",     # Active minutes data
    "EXERCISE",           # Exercise data
    "NUTRITION",          # Nutrition data
    "WATER",              # Water intake data
    "MEDITATION",         # Meditation data
    "STRESS",             # Stress data
    "OXYGEN_SATURATION",  # Oxygen saturation data
    "BODY_TEMPERATURE",   # Body temperature data
    "BODY_MASS_INDEX",    # Body mass index data
    "BODY_FAT",           # Body fat data
    "MUSCLE_MASS",        # Muscle mass data
    "BONE_MASS",          # Bone mass data
    "HYDRATION",          # Hydration data
    "PULSE_WAVE_VELOCITY" # Pulse wave velocity data
]

# Property keys for different data types
VALID_PROPERTY_KEYS = [
    "UNKNOWN",
    "ACTIVITY_TYPE",      # Activity type data
    "HRM_LOG",            # Heart rate monitoring data
    "ACCELEROMETER_LOG",  # Accelerometer data
    "STEPS",              # Step count data
    "DISTANCE",           # Distance data
    "CALORIES",           # Calories data
    "FLOORS",             # Floors data
    "ELEVATION",          # Elevation data
    "ACTIVE_MINUTES",     # Active minutes data
    "SLEEP_DURATION",     # Sleep duration data
    "SLEEP_QUALITY",      # Sleep quality data
    "WEIGHT",             # Weight data
    "BMI",                # Body mass index data
    "BODY_FAT",           # Body fat data
    "MUSCLE_MASS",        # Muscle mass data
    "BONE_MASS",          # Bone mass data
    "HYDRATION",          # Hydration data
    "SYSTOLIC",           # Systolic blood pressure data
    "DIASTOLIC",          # Diastolic blood pressure data
    "HEART_RATE",         # Heart rate data
    "RESTING_HEART_RATE", # Resting heart rate data
    "MAX_HEART_RATE",     # Maximum heart rate data
    "OXYGEN_SATURATION",  # Oxygen saturation data
    "BODY_TEMPERATURE",   # Body temperature data
    "MOOD",               # Mood data
    "STRESS",             # Stress data
    "NUTRITION",          # Nutrition data
    "WATER",              # Water intake data
    "MEDITATION",         # Meditation data
    "LOCATION",           # Location data
    "ACTIVITY_DURATION",  # Activity duration data
    "ACTIVITY_INTENSITY", # Activity intensity data
    "ACTIVITY_NAME",      # Activity name data
    "ACTIVITY_DATE",      # Activity date data
    "ACTIVITY_TIME",      # Activity time data
    "ACTIVITY_CALORIES",  # Activity calories data
    "ACTIVITY_DISTANCE",  # Activity distance data
    "ACTIVITY_STEPS",     # Activity steps data
    "ACTIVITY_FLOORS",    # Activity floors data
    "ACTIVITY_ELEVATION", # Activity elevation data
    "ACTIVITY_HEART_RATE" # Activity heart rate data
]

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s" 
