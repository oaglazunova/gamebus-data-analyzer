"""
GameBus API client for interacting with the GameBus platform.
"""
import requests
import logging
import random
from typing import Dict, List, Optional, Any
import time

from config.credentials import BASE_URL, TOKEN_URL, PLAYER_ID_URL, ACTIVITIES_URL
from config.settings import MAX_RETRIES, REQUEST_TIMEOUT, VALID_GAME_DESCRIPTORS

# Set up logging
logger = logging.getLogger(__name__)

class GameBusClient:
    """
    Client for interacting with the GameBus API.
    """

    def __init__(self, authcode: str):
        """
        Initialize the GameBus client.

        Args:
            authcode: Authentication code for GameBus API
        """
        self.authcode = authcode
        self.base_url = BASE_URL
        self.token_url = TOKEN_URL
        self.player_id_url = PLAYER_ID_URL
        self.activities_url = ACTIVITIES_URL
        self._cache = {}  # Cache for API responses

    def get_player_token(self, username: str, password: str) -> Optional[str]:
        """
        Get an access token for the player.

        Args:
            username: Player's username/email
            password: Player's password

        Returns:
            Access token or None if authentication failed
        """
        payload = {
            "grant_type": "password",
            "username": username,
            "password": password
        }
        headers = {
            "Authorization": f"Basic {self.authcode}",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        try:
            response = requests.post(self.token_url, headers=headers, data=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            token = response.json().get("access_token")
            logger.info("Token fetched successfully")
            return token
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get player token: {e}")
            return None

    def _fetch_paginated_data(self, data_url: str, token: str, page_size: int = 50) -> List[Dict[str, Any]]:
        """
        Fetch paginated data from the GameBus API.

        Args:
            data_url: URL to fetch data from
            token: Access token
            page_size: Number of items per page

        Returns:
            List of data points
        """
        headers = {"Authorization": f"Bearer {token}"}
        logger.info(f"Fetching paginated data from URL: {data_url}")

        all_data = []
        page = 0
        empty_pages_count = 0
        max_empty_pages = 3  # Stop after 3 consecutive empty pages
        max_pages = 10  # Maximum number of pages to fetch to prevent infinite loops
        start_time = time.time()
        max_time = 60  # Maximum time in seconds for the entire pagination process

        while empty_pages_count < max_empty_pages and page < max_pages and (time.time() - start_time) < max_time:
            paginated_url = f"{data_url}&page={page}&size={page_size}"
            retry_count = 0
            success = False

            logger.info(f"Fetching page {page} from URL: {paginated_url}")

            while retry_count < MAX_RETRIES and not success and (time.time() - start_time) < max_time:
                try:
                    logger.info(f"Requesting URL: {paginated_url}, attempt {retry_count + 1}/{MAX_RETRIES}")
                    response = requests.get(paginated_url, headers=headers, timeout=REQUEST_TIMEOUT)

                    # Log response status and headers for debugging
                    logger.info(f"Response status: {response.status_code}")

                    response.raise_for_status()

                    # Try to parse JSON response
                    try:
                        data = response.json()
                        success = True
                    except ValueError as json_err:
                        logger.error(f"Failed to parse JSON response: {json_err}")
                        logger.debug(f"Response content: {response.text[:500]}...")  # Log first 500 chars
                        retry_count += 1
                        time.sleep(2 * retry_count)  # Exponential backoff
                        continue

                    logger.info(f"Data fetched successfully for page {page}")

                    if not data:
                        logger.info(f"Empty data returned for page {page}")
                        empty_pages_count += 1
                        page += 1
                        break

                    # Reset empty pages counter if we got data
                    empty_pages_count = 0
                    all_data.extend(data)
                    page += 1

                except requests.exceptions.RequestException as e:
                    retry_count += 1
                    if retry_count < MAX_RETRIES:
                        logger.warning(f"Failed to retrieve data for page {page}, attempt {retry_count}/{MAX_RETRIES}: {e}")

                        # Check for rate limiting (status code 429)
                        if hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
                            # Get retry-after header if available
                            retry_after = e.response.headers.get('Retry-After')
                            if retry_after:
                                try:
                                    # Try to parse as integer seconds
                                    sleep_time = int(retry_after)
                                except ValueError:
                                    # Default to exponential backoff
                                    sleep_time = min(2 ** retry_count + (0.1 * random.random()), 30)
                            else:
                                # Default to exponential backoff with longer wait for rate limiting
                                sleep_time = min(5 ** retry_count + (0.5 * random.random()), 60)

                            logger.warning(f"Rate limit exceeded. Waiting for {sleep_time:.2f} seconds before retrying.")
                        else:
                            # Standard exponential backoff with jitter for other errors
                            sleep_time = min(2 ** retry_count + (0.1 * random.random()), 15)  # Increased max sleep time
                            logger.info(f"Retrying in {sleep_time:.2f} seconds")

                        time.sleep(sleep_time)
                    else:
                        logger.error(f"Failed to retrieve data for page {page} after {MAX_RETRIES} attempts: {e}")
                        if hasattr(e, 'response') and e.response is not None:
                            logger.error(f"Response status code: {e.response.status_code}")
                            logger.error(f"Response content: {e.response.text[:500]}...")  # Log first 500 chars
                        return all_data  # Return what we have so far

            # If we couldn't get data after all retries, break the pagination loop
            if not success:
                logger.warning(f"Failed to get data for page {page} after all retries, stopping pagination")
                break

        # Log the reason for stopping pagination
        if empty_pages_count >= max_empty_pages:
            logger.info(f"Stopped pagination after {max_empty_pages} consecutive empty pages")
        elif page >= max_pages:
            logger.info(f"Stopped pagination after reaching maximum number of pages ({max_pages})")
        elif (time.time() - start_time) >= max_time:
            logger.info(f"Stopped pagination after reaching maximum time limit ({max_time} seconds)")

        logger.info(f"Total data points fetched: {len(all_data)}")
        return all_data

    def get_player_id(self, token: str) -> Optional[int]:
        """
        Get the player ID using the access token.

        Args:
            token: Access token

        Returns:
            Player ID or None if retrieval failed
        """
        headers = {
            "Authorization": f"Bearer {token}"
        }

        try:
            response = requests.get(self.player_id_url, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            player_data = response.json()
            player_id = player_data.get("player", {}).get("id")
            logger.info("Player ID fetched successfully")
            return player_id
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get player ID: {e}")
            return None

    def get_player_data(self, token: str, player_id: int, game_descriptor: str, 
                       page_size: int = 50, try_all_descriptors: bool = False) -> List[Dict[str, Any]]:
        """
        Get player data for a specific game descriptor.

        Args:
            token: Access token
            player_id: Player ID
            game_descriptor: Type of data to retrieve (e.g., "GEOFENCE", "LOG_MOOD")
            page_size: Number of items per page
            try_all_descriptors: If True, try all valid game descriptors if the specified one doesn't return data

        Returns:
            List of player data points
        """
        # Create a cache key based on player_id and game_descriptor
        cache_key = f"{player_id}_{game_descriptor}"

        # Check if data is already in cache
        if cache_key in self._cache:
            logger.info(f"Using cached data for player {player_id} with game descriptor '{game_descriptor}'")
            return self._cache[cache_key]

        logger.info(f"Getting player data for player {player_id} with game descriptor '{game_descriptor}'")

        if game_descriptor not in VALID_GAME_DESCRIPTORS:
            logger.warning(f"Game descriptor '{game_descriptor}' not in VALID_GAME_DESCRIPTORS. Attempting to use it anyway.")

        headers = {"Authorization": f"Bearer {token}"}

        # Construct URL based on game descriptor
        urls_to_try = []

        # Standard URL format with gds parameter
        urls_to_try.append((self.activities_url + "&gds={}").format(player_id, game_descriptor))

        # Special case for SELFREPORT
        if game_descriptor == "SELFREPORT":
            urls_to_try = [(self.activities_url + "&excludedGds=").format(player_id)] + urls_to_try

        # Try without any game descriptor filter
        urls_to_try.append(self.activities_url.format(player_id))

        # Use the first URL as the default
        data_url = urls_to_try[0]

        # If we're trying all descriptors, we'll only use the first URL for now
        # The other URLs will be tried later if needed
        if not try_all_descriptors:
            # Try each URL until we get data
            for url in urls_to_try:
                logger.info(f"Trying URL: {url}")
                # Check if this URL is already in cache
                url_cache_key = f"{player_id}_{url}"
                if url_cache_key in self._cache:
                    logger.info(f"Using cached data for URL: {url}")
                    temp_data = self._cache[url_cache_key]
                else:
                    temp_data = self._fetch_paginated_data(url, token, page_size)
                    # Cache the result
                    self._cache[url_cache_key] = temp_data

                if temp_data:
                    logger.info(f"Found data using URL: {url}")
                    # Cache the result for the original cache key
                    self._cache[cache_key] = temp_data
                    return temp_data

            # If we didn't find any data, use the default URL and continue with the normal flow
            logger.warning(f"No data found with any URL format. Using default URL: {data_url}")

        # Use the _fetch_paginated_data method to get the data
        all_player_data = self._fetch_paginated_data(data_url, token, page_size)

        logger.info(f"Total data points fetched: {len(all_player_data)}")

        # Cache the result
        self._cache[cache_key] = all_player_data

        # If no data was found and try_all_descriptors is True, try all valid game descriptors
        # But limit the number of descriptors to try to avoid hanging
        if not all_player_data and try_all_descriptors:
            logger.info(f"No data found for game descriptor '{game_descriptor}'. Trying a limited set of game descriptors.")

            # Only try a few common descriptors instead of all of them
            common_descriptors = ["GEOFENCE", "LOG_MOOD", "TIZEN(DETAIL)", "NOTIFICATION(DETAIL)", "SELFREPORT"]
            descriptors_to_try = [d for d in common_descriptors if d != game_descriptor]

            logger.info(f"Will try these descriptors: {descriptors_to_try}")

            for descriptor in descriptors_to_try:
                # Check if this descriptor is already in cache
                descriptor_cache_key = f"{player_id}_{descriptor}"
                if descriptor_cache_key in self._cache:
                    logger.info(f"Using cached data for game descriptor: {descriptor}")
                    descriptor_data = self._cache[descriptor_cache_key]
                    if descriptor_data:
                        logger.info(f"Found {len(descriptor_data)} data points with game descriptor '{descriptor}' (from cache)")
                        # Cache the result for the original cache key
                        self._cache[cache_key] = descriptor_data
                        return descriptor_data

                logger.info(f"Trying game descriptor: {descriptor}")
                try:
                    # Recursive call with the new descriptor, but don't try all descriptors again
                    descriptor_data = self.get_player_data(token, player_id, descriptor, page_size, try_all_descriptors=False)
                    if descriptor_data:
                        logger.info(f"Found {len(descriptor_data)} data points with game descriptor '{descriptor}'")
                        # Cache the result for the original cache key
                        self._cache[cache_key] = descriptor_data
                        return descriptor_data
                except Exception as e:
                    logger.warning(f"Error trying game descriptor '{descriptor}': {e}")
                    continue

            logger.warning("No data found with any of the common game descriptors")

        return all_player_data
