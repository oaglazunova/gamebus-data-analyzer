"""
Path configurations for the GameBus-HealthBehaviorMining project.
"""
import os

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Data directories
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data_raw")


# Define default paths
USERS_FILE_PATH = os.path.join(PROJECT_ROOT, "config", "users.xlsx")
OUTPUT_PATH = RAW_DATA_DIR

# Ensure directories exist
for directory in [RAW_DATA_DIR]:
    os.makedirs(directory, exist_ok=True) 
