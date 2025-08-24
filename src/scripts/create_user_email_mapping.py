r"""
Manual script to generate data_raw\user_email_mapping.txt.

It logs in each user from config\users.xlsx, retrieves:
- player_id (GameBus player.id)
- email (from /users/current)
- account user_id (nested user.id) either from existing data_raw\player_<pid>_all_raw.json
  or by making a minimal data request and parsing raw JSON responses in-memory (no files written).

Usage (from project root):
    python -m src.scripts.create_user_email_mapping
or
    python src\scripts\create_user_email_mapping.py

Notes:
- The extraction pipeline no longer creates this mapping automatically.
- This script is idempotent: running it multiple times will overwrite the mapping file consistently.
"""
from __future__ import annotations

import os
import json
from typing import Optional, Tuple, Dict, Any

import pandas as pd

# Ensure imports work whether running as a module or script
try:
    from config.paths import RAW_DATA_DIR, USERS_FILE_PATH
    from config.credentials import AUTHCODE
    from src.extraction.gamebus_client import GameBusClient
    from config.settings import VALID_GAME_DESCRIPTORS
except Exception:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from config.paths import RAW_DATA_DIR, USERS_FILE_PATH  # type: ignore
    from config.credentials import AUTHCODE  # type: ignore
    from src.extraction.gamebus_client import GameBusClient  # type: ignore
    from config.settings import VALID_GAME_DESCRIPTORS  # type: ignore

MAPPING_FILE = os.path.join(RAW_DATA_DIR, "user_email_mapping.txt")


def _find_user_id(obj: Any) -> Optional[str]:
    if isinstance(obj, dict):
        u = obj.get("user")
        if isinstance(u, dict):
            uid = u.get("id")
            if isinstance(uid, (int, str)) and str(uid).strip() != "":
                return str(uid).strip()
        for key in ("creator", "player", "account", "owner"):
            if key in obj:
                res = _find_user_id(obj[key])
                if res:
                    return res
        for v in obj.values():
            res = _find_user_id(v)
            if res:
                return res
    elif isinstance(obj, list):
        for item in obj:
            res = _find_user_id(item)
            if res:
                return res
    return None


def _extract_uid_from_existing_all_raw(pid: int | str) -> Optional[str]:
    path = os.path.join(RAW_DATA_DIR, f"player_{pid}_all_raw.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return _find_user_id(data)
    except Exception:
        return None


def _extract_uid_from_raw_responses(raw_responses: list[str]) -> Optional[str]:
    for raw in raw_responses:
        try:
            data = json.loads(raw)
        except Exception:
            continue
        uid = _find_user_id(data)
        if uid:
            return uid
    return None


def _read_users_xlsx(users_file: str) -> pd.DataFrame:
    df = pd.read_excel(users_file)
    # Normalize columns
    required = ["email", "password"]
    for req in required:
        if req not in df.columns:
            raise ValueError(f"Users file must contain '{req}' column")
    cols = ["email", "password"]
    if "UserID" in df.columns:
        cols.append("UserID")
    return df[cols]


def _sanitize_credential(value: Any, allow_numeric: bool = False) -> Optional[str]:
    """Coerce a cell value to a clean string or return None if truly missing.
    - Trims whitespace
    - Treats None/NaN/""/"nan" as missing
    - Optionally converts numeric values to strings (e.g., 12345.0 -> "12345")
    """
    # Handle pandas-na if pandas is available
    try:
        import pandas as _pd  # local import to avoid issues if pandas isn't present at runtime
        if _pd.isna(value):
            return None
    except Exception:
        pass

    if value is None:
        return None

    # Strings
    if isinstance(value, str):
        s = value.strip()
        if s == "" or s.lower() == "nan":
            return None
        return s

    # Numbers (when allowed)
    if allow_numeric and isinstance(value, (int, float)):
        try:
            if isinstance(value, float):
                # NaN guard without numpy
                if value != value:
                    return None
                if value.is_integer():
                    return str(int(value)).strip()
            return str(value).strip()
        except Exception:
            return None

    # Other types: attempt string conversion
    try:
        s = str(value).strip()
        if s == "" or s.lower() == "nan":
            return None
        return s
    except Exception:
        return None


def _determine_user_id(client: GameBusClient, token: str, player_id: int) -> Optional[str]:
    # 1) Try existing all_raw file
    uid = _extract_uid_from_existing_all_raw(player_id)
    if uid:
        return uid

    # 2) Try minimal data fetch without saving files
    # Iterate over a small subset of descriptors first, then all
    descriptors_to_try = list(VALID_GAME_DESCRIPTORS[:])
    # Heuristic: quickly try a few common ones first by placing them at front if present
    common_first = ["LOG_MOOD", "STEPS", "GEOFENCE", "NUTRITION_SUMMARY"]
    for d in reversed(common_first):
        if d in descriptors_to_try:
            descriptors_to_try.remove(d)
            descriptors_to_try.insert(0, d)

    for descriptor in descriptors_to_try:
        try:
            _, _, raw_responses = client.get_user_data(token, player_id, descriptor, try_all_descriptors=False)
            uid = _extract_uid_from_raw_responses(raw_responses)
            if uid:
                return uid
        except Exception:
            continue

    # 3) As a last resort, try scanning across all descriptors by enabling try_all_descriptors
    try:
        _, _, raw_responses = client.get_user_data(token, player_id, descriptors_to_try[0] if descriptors_to_try else "LOG_MOOD", try_all_descriptors=True)
        uid = _extract_uid_from_raw_responses(raw_responses)
        if uid:
            return uid
    except Exception:
        pass

    return None


def _write_mapping(mappings: Dict[str, Tuple[Optional[str], str]]):
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    with open(MAPPING_FILE, "w", encoding="utf-8") as f:
        for pid, (uid, email) in mappings.items():
            safe_uid = uid if uid not in (None, "") else "-"
            f.write(f"player_id={pid} user_id={safe_uid}: {email}\n")


# Type annotation forward declaration for Tuple
from typing import Tuple as _TupleAlias  # noqa: E402


def main() -> None:
    users_file = USERS_FILE_PATH
    if not os.path.exists(users_file):
        print(f"Users file not found: {users_file}")
        return

    try:
        df = _read_users_xlsx(users_file)
    except Exception as e:
        print(f"Failed to read users file: {e}")
        return

    client = GameBusClient(AUTHCODE)
    mappings: Dict[str, _TupleAlias[Optional[str], str]] = {}

    processed = 0
    for _, row in df.iterrows():
        raw_email = row.get("email")
        raw_password = row.get("password")
        raw_provided_uid = row.get("UserID") if "UserID" in row else None
        processed += 1

        email = _sanitize_credential(raw_email, allow_numeric=False)
        password = _sanitize_credential(raw_password, allow_numeric=True)
        if not email or not password:
            pwd_disp = "(missing)" if not password else "***"
            print(f"Skipping row due to missing credentials: email={email or '(missing)'} password={pwd_disp}")
            continue

        token = client.get_user_token(email, password)
        if not token:
            print(f"  Failed to get token for {email}")
            continue

        player_id, api_email = client.get_user_id(token)
        if player_id is None:
            print(f"  Failed to get player ID for {email}")
            continue

        # Prefer API email if provided
        final_email = api_email or email

        # Determine account user_id
        uid = _determine_user_id(client, token, player_id)
        provided_uid = _sanitize_credential(raw_provided_uid, allow_numeric=True)
        if not uid and provided_uid is not None:
            uid = provided_uid

        mappings[str(player_id)] = (uid, final_email)

    _write_mapping(mappings)
    print(f"Wrote mapping for {len(mappings)} users to {MAPPING_FILE}")


if __name__ == "__main__":
    main()
