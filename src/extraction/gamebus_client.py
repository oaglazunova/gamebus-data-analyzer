"""
GameBus API client for interacting with the GameBus platform.
"""
import requests
import logging
import random
from typing import Dict, List, Optional, Any, Tuple
import time

from config.credentials import BASE_URL, TOKEN_URL, USER_ID_URL, ACTIVITIES_URL
from config.settings import MAX_RETRIES, REQUEST_TIMEOUT, VALID_GAME_DESCRIPTORS, DEFAULT_PAGE_SIZE

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
        self.user_id_url = USER_ID_URL
        self.activities_url = ACTIVITIES_URL
        self._cache = {}  # Cache for API responses

    def get_user_token(self, username: str, password: str) -> Optional[str]:
        """
        Get an access token for the user.

        Args:
            username: User's username/email
            password: User's password

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
            "Origin": "https://base.healthyw8.gamebus.eu",
            "Referer": "https://base.healthyw8.gamebus.eu/",
            "Accept": "application/json, text/plain, */*"
        }

        # Add retry logic with exponential backoff
        retry_count = 0
        max_retries = MAX_RETRIES
        start_time = time.time()
        max_time = 60  # Maximum time in seconds for the token retrieval process

        while retry_count < max_retries and (time.time() - start_time) < max_time:
            try:
                # Make OPTIONS request first to mimic browser behavior
                # For token endpoint, we need to use POST in the OPTIONS request
                try:
                    base_url = self.token_url.split('?')[0]
                    logger.info(f"Making OPTIONS request to token URL: {base_url}")

                    options_headers = {
                        "Access-Control-Request-Method": "POST",
                        "Access-Control-Request-Headers": "authorization,content-type",
                        "Origin": "https://base.healthyw8.gamebus.eu",
                        "Referer": "https://base.healthyw8.gamebus.eu/"
                    }

                    options_response = requests.options(base_url, headers=options_headers, timeout=REQUEST_TIMEOUT)
                except requests.exceptions.RequestException as e:
                    logger.warning(f"OPTIONS request for token URL failed, continuing anyway: {e}")

                response = requests.post(self.token_url, headers=headers, data=payload, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                token = response.json().get("access_token")
                return token
            except requests.exceptions.RequestException as e:
                retry_count += 1
                if retry_count < max_retries:
                    logger.warning(f"Failed to get user token, attempt {retry_count}/{max_retries}: {e}")

                    # Determine sleep time based on error type
                    if hasattr(e, 'response') and e.response is not None:
                        if e.response.status_code == 429:
                            # Rate limiting - use longer backoff
                            sleep_time = min(5 ** retry_count + (0.5 * random.random()), 60)
                            logger.warning(f"Rate limit exceeded. Waiting for {sleep_time:.2f} seconds before retrying.")
                        elif e.response.status_code >= 500:
                            # Server error (5xx) - use longer backoff times
                            sleep_time = min(10 ** retry_count + (1.0 * random.random()), 120)
                            logger.warning(f"Server error (status code: {e.response.status_code}). Waiting for {sleep_time:.2f} seconds before retrying.")
                            # Log more details about the error
                            logger.error(f"Server error details: {e}")
                            if hasattr(e.response, 'text'):
                                logger.error(f"Response content: {e.response.text[:1000]}...")
                        else:
                            # Standard exponential backoff for other errors
                            sleep_time = min(2 ** retry_count + (0.1 * random.random()), 15)
                            logger.info(f"HTTP error {e.response.status_code}. Retrying in {sleep_time:.2f} seconds")
                    else:
                        # Network error - standard backoff
                        sleep_time = min(2 ** retry_count + (0.1 * random.random()), 15)
                        logger.info(f"Network error (no status code). Retrying in {sleep_time:.2f} seconds")

                    time.sleep(sleep_time)
                else:
                    logger.error(f"Failed to get user token after {max_retries} attempts: {e}")
                    if hasattr(e, 'response') and e.response is not None:
                        logger.error(f"Response status code: {e.response.status_code}")
                        logger.error(f"Response content: {e.response.text[:500]}...")
                    return None

        # If we've exhausted all retries or exceeded max time
        logger.error(f"Failed to get user token after {retry_count} attempts or {max_time} seconds")
        return None

    def _make_options_request(self, url: str) -> bool:
        """
        Make an OPTIONS request to check CORS before making the actual request.
        This mimics browser behavior.

        Args:
            url: URL to make the OPTIONS request to

        Returns:
            True if the OPTIONS request was successful, False otherwise
        """
        try:
            # Extract the base URL without query parameters for OPTIONS request
            base_url = url.split('?')[0]

            # Headers for OPTIONS request
            options_headers = {
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "authorization",
                "Origin": "https://base.healthyw8.gamebus.eu",
                "Referer": "https://base.healthyw8.gamebus.eu/"
            }

            # Make the OPTIONS request
            options_response = requests.options(base_url, headers=options_headers, timeout=REQUEST_TIMEOUT)

            # Check if the OPTIONS request was successful
            if options_response.status_code == 200:
                return True
            else:
                logger.warning(f"OPTIONS request failed with status code: {options_response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to make OPTIONS request: {e}")
            return False

    def _fetch_paginated_data(self, data_url: str, token: str, page_size: int = DEFAULT_PAGE_SIZE) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Fetch paginated data from the GameBus API.

        Args:
            data_url: URL to fetch data from
            token: Access token
            page_size: Number of items per page

        Returns:
            Tuple containing:
            - List of data points
            - List of raw JSON responses
        """
        headers = {
            "Authorization": f"Bearer {token}",
            "Origin": "https://base.healthyw8.gamebus.eu",
            "Referer": "https://base.healthyw8.gamebus.eu/",
            "Accept": "application/json, text/plain, */*"
        }

        all_data = []
        raw_responses = []  # Store raw JSON responses
        page = 0
        empty_pages_count = 0
        max_empty_pages = 10  # Stop after 10 consecutive empty pages (increased from 5)
        max_pages = 50  # Maximum number of pages to fetch to prevent infinite loops (increased from 20)
        start_time = time.time()
        max_time = 300  # Maximum time in seconds for the entire pagination process (increased from 120)

        while empty_pages_count < max_empty_pages and page < max_pages and (time.time() - start_time) < max_time:
            paginated_url = f"{data_url}&page={page}&size={page_size}"
            retry_count = 0
            success = False

            while retry_count < MAX_RETRIES and not success and (time.time() - start_time) < max_time:
                try:
                    # Make OPTIONS request first to mimic browser behavior
                    self._make_options_request(paginated_url)
                    response = requests.get(paginated_url, headers=headers, timeout=REQUEST_TIMEOUT)
                    response.raise_for_status()

                    # Try to parse JSON response
                    try:
                        # Store the raw response text
                        raw_response_text = response.text

                        # Parse the JSON
                        data = response.json()
                        success = True

                        # Store the raw response
                        raw_responses.append(raw_response_text)
                    except ValueError as json_err:
                        logger.error(f"Failed to parse JSON response: {json_err}")
                        logger.debug(f"Response content: {response.text[:500]}...")  # Log first 500 chars
                        retry_count += 1
                        time.sleep(2 * retry_count)  # Exponential backoff
                        continue

                    if not data:
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
                        if hasattr(e, 'response') and e.response is not None:
                            if e.response.status_code == 429:
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
                            elif e.response.status_code >= 500:
                                # Server error (5xx) - use longer backoff times
                                sleep_time = min(10 ** retry_count + (1.0 * random.random()), 120)  # Much longer wait for server errors
                                logger.warning(f"Server error (status code: {e.response.status_code}). Waiting for {sleep_time:.2f} seconds before retrying.")
                                # Log more details about the error
                                logger.error(f"Server error details: {e}")
                                if hasattr(e.response, 'text'):
                                    logger.error(f"Response content: {e.response.text[:1000]}...")  # Log more content for server errors
                            else:
                                # Standard exponential backoff with jitter for other errors
                                sleep_time = min(2 ** retry_count + (0.1 * random.random()), 15)  # Increased max sleep time
                                logger.info(f"HTTP error {e.response.status_code}. Retrying in {sleep_time:.2f} seconds")
                        else:
                            # Standard exponential backoff with jitter for other errors (no response object)
                            sleep_time = min(2 ** retry_count + (0.1 * random.random()), 15)  # Increased max sleep time
                            logger.info(f"Network error (no status code). Retrying in {sleep_time:.2f} seconds")

                        time.sleep(sleep_time)
                    else:
                        logger.error(f"Failed to retrieve data for page {page} after {MAX_RETRIES} attempts: {e}")
                        if hasattr(e, 'response') and e.response is not None:
                            logger.error(f"Response status code: {e.response.status_code}")
                            logger.error(f"Response content: {e.response.text[:500]}...")  # Log first 500 chars
                        return all_data, raw_responses  # Return what we have so far

            # If we couldn't get data after all retries, break the pagination loop
            if not success:
                logger.warning(f"Failed to get data for page {page} after all retries, stopping pagination")
                break

        return all_data, raw_responses

    def get_user_id(self, token: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Retrieve the authenticated player's ID and email from GameBus.

        Source:
            GET USER_ID_URL (/users/current). The JSON response contains:
              - player.id: the numeric Player ID used for all subsequent API calls
              - email: the account email address

        Args:
            token: OAuth access token

        Returns:
            Tuple containing:
            - Player ID (int) or None if retrieval failed
            - User email (str) or None if not available
        """
        headers = {
            "Authorization": f"Bearer {token}",
            "Origin": "https://base.healthyw8.gamebus.eu",
            "Referer": "https://base.healthyw8.gamebus.eu/",
            "Accept": "application/json, text/plain, */*"
        }

        # Add retry logic with exponential backoff
        retry_count = 0
        max_retries = MAX_RETRIES
        start_time = time.time()
        max_time = 60  # Maximum time in seconds for the player ID retrieval process

        while retry_count < max_retries and (time.time() - start_time) < max_time:
            try:
                # Make OPTIONS request first to mimic browser behavior
                self._make_options_request(self.user_id_url)

                response = requests.get(self.user_id_url, headers=headers, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                user_data = response.json()
                user_id = user_data.get("player", {}).get("id")
                user_email = user_data.get("email")

                # Validate that user_id is an integer
                if not isinstance(user_id, int):
                    logger.error(f"Invalid player ID format: {user_id}. Expected an integer.")
                    return None, None

                return user_id, user_email
            except requests.exceptions.RequestException as e:
                retry_count += 1
                if retry_count < max_retries:
                    logger.warning(f"Failed to get player ID and email, attempt {retry_count}/{max_retries}: {e}")

                    # Determine sleep time based on error type
                    if hasattr(e, 'response') and e.response is not None:
                        if e.response.status_code == 429:
                            # Rate limiting - use longer backoff
                            sleep_time = min(5 ** retry_count + (0.5 * random.random()), 60)
                            logger.warning(f"Rate limit exceeded. Waiting for {sleep_time:.2f} seconds before retrying.")
                        elif e.response.status_code >= 500:
                            # Server error (5xx) - use longer backoff times
                            sleep_time = min(10 ** retry_count + (1.0 * random.random()), 120)
                            logger.warning(f"Server error (status code: {e.response.status_code}). Waiting for {sleep_time:.2f} seconds before retrying.")
                            # Log more details about the error
                            logger.error(f"Server error details: {e}")
                            if hasattr(e.response, 'text'):
                                logger.error(f"Response content: {e.response.text[:1000]}...")
                        else:
                            # Standard exponential backoff for other errors
                            sleep_time = min(2 ** retry_count + (0.1 * random.random()), 15)
                            logger.info(f"HTTP error {e.response.status_code}. Retrying in {sleep_time:.2f} seconds")
                    else:
                        # Network error - standard backoff
                        sleep_time = min(2 ** retry_count + (0.1 * random.random()), 15)
                        logger.info(f"Network error (no status code). Retrying in {sleep_time:.2f} seconds")

                    time.sleep(sleep_time)
                else:
                    logger.error(f"Failed to get player ID and email after {max_retries} attempts: {e}")
                    if hasattr(e, 'response') and e.response is not None:
                        logger.error(f"Response status code: {e.response.status_code}")
                        logger.error(f"Response content: {e.response.text[:500]}...")
                    return None, None

        # If we've exhausted all retries or exceeded max time
        logger.error(f"Failed to get player ID and email after {retry_count} attempts or {max_time} seconds")
        return None, None

    def get_user_data(self, token: str, user_id: int, game_descriptor: str, 
                       page_size: int = DEFAULT_PAGE_SIZE, try_all_descriptors: bool = False) -> tuple[List[Dict[str, Any]], str, List[str]]:
        """
        Get user data for a specific game descriptor.

        Args:
            token: Access token
            user_id: Player ID (must be an integer)
            game_descriptor: Type of data to retrieve (e.g., "GEOFENCE", "LOG_MOOD")
            page_size: Number of items per page
            try_all_descriptors: If True, try all valid game descriptors if the specified one doesn't return data

        Returns:
            Tuple containing:
            - List of user data points
            - The actual game descriptor used to fetch the data
            - List of raw JSON responses
        """
        # Validate that user_id is an integer
        if not isinstance(user_id, int):
            logger.error(f"Invalid player ID format: {user_id}. Expected an integer.")
            return [], game_descriptor, []

        # Create a cache key based on user_id and game_descriptor
        cache_key = f"{user_id}_{game_descriptor}"

        # Check if data is already in cache
        if cache_key in self._cache:
            # Cache now stores a tuple of (data, raw_responses)
            data, raw_responses = self._cache[cache_key]
            return data, game_descriptor, raw_responses

        if game_descriptor not in VALID_GAME_DESCRIPTORS:
            logger.warning(f"Game descriptor '{game_descriptor}' not in VALID_GAME_DESCRIPTORS. Attempting to use it anyway.")

        headers = {
            "Authorization": f"Bearer {token}",
            "Origin": "https://base.healthyw8.gamebus.eu",
            "Referer": "https://base.healthyw8.gamebus.eu/",
            "Accept": "application/json, text/plain, */*"
        }

        # Construct URL based on game descriptor
        urls_to_try = []

        # Standard URL format with gds parameter
        urls_to_try.append((self.activities_url + "&gds={}").format(user_id, game_descriptor))

        # Try without any game descriptor filter
        urls_to_try.append(self.activities_url.format(user_id))

        # Use the first URL as the default
        data_url = urls_to_try[0]

        # If we're trying all descriptors, we'll only use the first URL for now
        # The other URLs will be tried later if needed
        if not try_all_descriptors:
            # Try each URL until we get data
            for url in urls_to_try:
                # Check if this URL is already in cache
                url_cache_key = f"{user_id}_{url}"
                if url_cache_key in self._cache:
                    temp_data, temp_raw_responses = self._cache[url_cache_key]
                else:
                    temp_data, temp_raw_responses = self._fetch_paginated_data(url, token, page_size)
                    # Cache the result
                    self._cache[url_cache_key] = (temp_data, temp_raw_responses)

                if temp_data:
                    # Cache the result for the original cache key
                    self._cache[cache_key] = (temp_data, temp_raw_responses)
                    return temp_data, game_descriptor, temp_raw_responses

            # If we didn't find any data, use the default URL and continue with the normal flow
            logger.warning(f"No data found with any URL format. Using default URL: {data_url}")

        # Use the _fetch_paginated_data method to get the data
        all_user_data, all_raw_responses = self._fetch_paginated_data(data_url, token, page_size)
        # Cache the result
        self._cache[cache_key] = (all_user_data, all_raw_responses)
        # If data was found, return it with the original game descriptor
        if all_user_data:
            return all_user_data, game_descriptor, all_raw_responses

        # If no data was found and try_all_descriptors is True, try all valid game descriptors
        if try_all_descriptors:
            # Try all valid game descriptors
            descriptors_to_try = [d for d in VALID_GAME_DESCRIPTORS if d != game_descriptor]

            for descriptor in descriptors_to_try:
                # Check if this descriptor is already in cache
                descriptor_cache_key = f"{user_id}_{descriptor}"
                if descriptor_cache_key in self._cache:
                    descriptor_data, descriptor_raw_responses = self._cache[descriptor_cache_key]
                    if descriptor_data:
                        # Cache the result for the original cache key
                        self._cache[cache_key] = (descriptor_data, descriptor_raw_responses)
                        return descriptor_data, descriptor, descriptor_raw_responses

                try:
                    # Recursive call with the new descriptor, but don't try all descriptors again
                    descriptor_data, actual_descriptor, descriptor_raw_responses = self.get_user_data(token, user_id, descriptor, page_size, try_all_descriptors=False)
                    if descriptor_data:
                        # Cache the result for the original cache key
                        self._cache[cache_key] = (descriptor_data, descriptor_raw_responses)
                        return descriptor_data, actual_descriptor, descriptor_raw_responses
                except Exception as e:
                    logger.warning(f"Error trying game descriptor '{descriptor}': {e}")
                    continue

            logger.warning("No data found with any of the valid game descriptors")

        return all_user_data, game_descriptor, all_raw_responses
