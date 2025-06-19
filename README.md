# GameBus Data Analyzer

A tool for extracting and analyzing health behavior data from the GameBus platform.

## Overview

This project extracts user activity data from the GameBus API and performs various analyses to generate insights about user behavior, activity patterns, and engagement.

## Key Features

- **Data Extraction**: Retrieves data from GameBus API with robust error handling and retry mechanisms
- **Parallel Processing**: Processes multiple users concurrently for improved performance
- **Comprehensive Analysis**: Generates statistics and visualizations for activity patterns, user engagement, and more
- **Flexible Configuration**: Supports various game descriptors and data types

## Quick Start

1. **Install Python 3.8+** and clone this repository
2. **Set up environment**: 
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Configure**:
   - Create `.env` file with `GAMEBUS_API_KEY=your_api_key_here`
   - Add user credentials to `config/users.xlsx` (see Configuration section below for format)
   - For analysis, add campaign files to `config/campaign_desc.xlsx` and `config/campaign_data.xlsx`

4. **Run**:
   ```
   python pipeline.py                    # Full pipeline (extraction + analysis)
   python pipeline.py --extract          # Only extract data
   python pipeline.py --analyze          # Only analyze existing data
   python pipeline.py --log-level DEBUG  # Verbose logging
   ```

## Project Structure

```
gamebus-data-analyzer/
├── config/                # Configuration files
├── src/                   # Source code
│   ├── extraction/        # Data extraction from GameBus
│   ├── analysis/          # Data analysis
│   ├── scripts/           # Utility scripts
│   └── utils/             # Utility functions
├── data_raw/              # Raw extracted data
├── data_analysis/         # Analysis output
├── logs/                  # Log files
└── pipeline.py            # Main pipeline runner
```

## Available Analyses

The tool provides various analyses including:
- Activity patterns and distributions
- User engagement metrics
- Movement types and physical activity statistics
- Challenge completion rates
- Heart rate data analysis (when available)

See the documentation in `src/analysis/data_analysis.py` for details on specific analyses.

## Configuration

### Configuration Files

- `credentials.py`: API endpoints and authentication settings
- `paths.py`: File paths used throughout the project
- `settings.py`: General settings including valid game descriptors and API parameters
- `users.xlsx`: User credentials for GameBus API access (you must create this)
- `campaign_desc.xlsx`: Campaign description data (required for analysis)
- `campaign_data.xlsx`: Campaign activity data (required for analysis)

### API Key

The GameBus API key must be stored in a `.env` file in the root directory:

```
GAMEBUS_API_KEY=your_api_key_here
```

### User Credentials

Create a `users.xlsx` file in the config directory with the following format:

```
email               | password      | UserID (optional)
---------------------|--------------|----------------
user@example.com     | password123  | 12345
```

Requirements:
- Must include header row with exact column names: `email` and `password` (lowercase)
- The `UserID` column is optional
- !: You can use the Excel file with users generated for GameBus campaigns without any modifications

### Campaign Data

For analysis functionality, copy these files from the GameBus Campaigns website:
- `campaign_desc.xlsx`: Contains challenge descriptions and levels
- `campaign_data.xlsx`: Contains activity data for all users

### Game Descriptors

The `settings.py` file contains a list of valid game descriptors that the system will extract data for. You can modify this list to focus on specific types of activities.

## Troubleshooting

### Common Issues

1. **Authentication Failures**:
   - Ensure your API key in `.env` is correct and up-to-date
   - Check that user credentials in `users.xlsx` are valid

2. **Missing Data**:
   - Verify that the game descriptors you need are enabled in `settings.py`
   - Check that campaign files are correctly formatted and in the right location

3. **Analysis Errors**:
   - Ensure both campaign files are present and properly formatted
   - Check that raw data has been extracted before running analysis

### Performance Tips

- Reduce the number of users processed in parallel if experiencing rate limiting
- Comment out unused game descriptors in `settings.py` to speed up extraction
- Use the DEBUG log level to identify specific issues

### Security Note

Configuration files are tracked by Git, but sensitive content (XLSX files with credentials) is ignored through `.gitignore` rules. Always check that your credentials aren't accidentally committed.

## License

This project is licensed under the MIT License.

## Acknowledgments

- GameBus platform for providing the API
