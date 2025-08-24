"""
Credentials configuration for GameBus API.
Contains authentication information needed to access the GameBus API.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
AUTHCODE = os.getenv('GAMEBUS_API_KEY')
if not AUTHCODE:
    raise ValueError("GAMEBUS_API_KEY environment variable is not set. Please add it to your .env file.")

# Default API endpoints
# BASE_URL = "https://api-new.gamebus.eu/v2"
BASE_URL = 'https://api.healthyw8.gamebus.eu/v2'
TOKEN_URL = f"{BASE_URL}/oauth/token"
# USER_ID_URL returns the current account info; expected JSON includes:
#   player.id (int): the GameBus Player ID used in all activities endpoints
#   email (str): the account email
USER_ID_URL = f"{BASE_URL}/users/current"
ACTIVITIES_URL = f"{BASE_URL}/players/{{}}/activities?sort=-date"
