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
- Frequency, timing, and type of tasks completed
- Completion rates by activity type/task (e.g., walking, fruit intake)

See the documentation in `src/analysis/data_analysis.py` for details on specific analyses.

### Activity Type by Day: data source

The heatmap saved as `activity_type_by_day.png` is built from the activities DataFrame loaded from `config/campaign_data.xlsx` (sheet name: `activities`). The loader (`load_excel_files` in `src/analysis/data_analysis.py`) reads that sheet into `csv_data['activities']`, which is then copied to `activities_df` inside `analyze_visualizations_challenges_tasks` to compute per-hour/day aggregations and the heatmap.

## Data fields: challenge_id and challenge_name

These two fields are used in reports/plots about challenge participation and completion. They come from two data sources during analysis:

- Primary (from activities): For each activity, we parse the rewardedParticipations column (a JSON-like string) into a list of reward dicts. From each reward we try to extract:
  - challenge_id from one of these keys: challengeId, challenge_id, challenge, cid, or id
  - challenge_name from one of these keys: challengeName, name, or label
  This parsing is done by extract_detailed_rewards in src/analysis/data_analysis.py and then processed in the analysis flow that builds completion_records.

- Secondary (from campaign description): If an ID is missing but we have a name, we map the name to the desc_challenges sheet (campaign_desc.xlsx → desc_challenges) to recover the numeric id. We also use desc_challenges to attach the official challenge name when available.

In summary, challenge_id primarily originates from each reward in activities.rewardedParticipations, and challenge_name is taken from that same reward object when present; otherwise they are completed by looking up id/name in the desc_challenges sheet.

## Data fields: tasks (source)

Tasks used in the analyses come from the campaign description Excel file and are loaded as a DataFrame named desc_tasks:

- Source file and sheet: config/campaign_desc.xlsx → sheet named "tasks". When loaded by the tool, all sheets from campaign_desc.xlsx are prefixed with desc_, therefore the tasks sheet becomes desc_tasks.
- Where it is loaded: In src/analysis/data_analysis.py, load_excel_files reads campaign_desc.xlsx and maps each sheet to csv_data with the desc_ prefix. Later, analyze_visualizations_challenges_tasks accesses tasks via tasks_df = csv_data['desc_tasks'].
- Typical fields: task id, challenge (foreign key to desc_challenges.id), name/label, points, timing/constraints depending on your campaign file.
- How tasks are used: The analyzer uses desc_tasks to summarize counts and points, and to relate tasks to their parent challenges and visualizations. Some sections also infer "completed tasks" patterns directly from activities (e.g., activity type, rewardedParticipations) for ranking and timing plots, but the canonical definitions and metadata of tasks originate from desc_tasks.

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


## FAQ

### What is the difference between points_by_activity_type and rewards_by_activity_type?
- points_by_activity_type: Sums the rewarded points across all activities of each type. This shows which activity types contributed the most total points overall (SUM per type).
- rewards_by_activity_type: Averages the rewarded points per activity instance, grouped by type. This shows how many points are typically awarded when a given activity type occurs (MEAN per type).