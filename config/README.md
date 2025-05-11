# Configuration Files

This directory contains configuration files for the GameBus-HealthBehaviorMining project.

## Files

- `credentials.py`: API credentials and endpoints
- `paths.py`: File paths used throughout the project
- `settings.py`: General settings and constants

## Important Note

The API key is now exclusively loaded from the `.env` file in the root directory for security reasons.

## For Development

To set up your development environment:

1. Create a `.env` file in the root directory with your GameBus API key:

```
GAMEBUS_API_KEY=your_api_key_here
```

2. For user data, you can either:
   - Specify a custom path in `config/paths.py`
   - Or specify a custom path using the `--users-file` parameter when running the pipeline

3. The users XLSX file must be in the correct format:
   - It must include a header row with exact column names: `email` and `password` (both lowercase)
   - The `UserID` column is optional but needed if you want to use the `--user-id` parameter
   - Other additional columns in the XLSX file will be ignored
   - Example:
     ```
     email               | password      | UserID
     ---------------------|--------------|-------
     user@example.com     | password123  | 12345
     ```

4. Default configurations:
   - Users file is expected at `config/users.xlsx`
   - Output will go to `data_raw/`
   - API key must be provided in the `.env` file

## Git Behavior

These configuration files are tracked by Git, but their content (especially XLSX files with credentials) is ignored through `.gitignore` rules. This allows the structure to be shared while keeping sensitive data private.
