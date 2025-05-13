# GameBus Data Analyzer

A script for extracting and analyzing health behavior data from the GameBus platform.

## Project Structure

```
gamebus-data-analyzer/
├── config/                         # Configuration files
│   ├── credentials.py              # GameBus API credentials
│   ├── paths.py                    # Path configurations
│   └── settings.py                 # General settings
├── src/                            # Source code
│   ├── extraction/                 # Data extraction from GameBus
│   │   ├── gamebus_client.py       # GameBus API client
│   │   └── data_collectors.py      # Data collectors for different data types
│   ├── analysis/                   # Data analysis
│   │   └── data_analysis.py        # Data analysis script
│   └── utils/                      # Utility functions
│       └── logging.py              # Logging utilities
├── data_raw/                       # Raw extracted data
├── data_analysis/                  # Analysis output
│   ├── statistics/                 # Statistical analysis output
│   └── visualizations/             # Visualization output
├── logs/                           # Log files
└── pipeline.py                     # Main pipeline runner
```

## Framework Workflow

1. **Data Extraction**: Extract raw data from the GameBus API, including:
   - GPS location data
   - Activity type data
   - Heart rate data
   - Accelerometer data
   - Mood data
   - Notification data

   The extraction process includes:
   - Authentication with the GameBus API
   - Retrieving player data for multiple users in parallel
   - Parsing and processing the raw data
   - Saving the processed data to JSON files in the `data_raw` directory

2. **Data Analysis**: Analyze the extracted data to generate insights, including:
   - Activity patterns and distributions
   - User engagement and participation
   - Challenge types and completion rates
   - Movement types and physical activity metrics
   - Heart rate data and patterns

   The analysis process includes:
   - Loading data from Excel files in the `config` directory and JSON files in the `data_raw` directory
   - Generating descriptive statistics 
   - Creating visualizations to provide insights
   - Performing bivariate analyses to explore relationships between variables
   - Saving visualizations to the `data_analysis/visualizations` directory
   - Saving statistics to the `data_analysis/statistics` directory

## Getting Started

### Prerequisites

- Python 3.8+
- GameBus API credentials

### Installation

1. Clone the repository
2. Install dependencies:
```
pip install -r requirements.txt
```

3. Configure the project:
   For data extraction:
   - Create a `.env` file in the root directory with your GameBus API key (see below)
   - Copy the .xlsx file with user credentials exported from GameBus Campaigns website into the `config/` directory and rename to `users.xlsx`

   For data analysis:
   - Copy campaign description downloaded from GameBus Campaigns website into the `config/` directory and rename to `campaign_desc.xlsx`
   - Copy campaign data downloaded from GameBus Campaigns website into the `config/` directory and rename to `campaign_data.xlsx`

### Configuration

This project is configured using environment variables and config files:

1. **Environment Variables (required):**
   Create a `.env` file in the root directory with:
   ```
   GAMEBUS_API_KEY=your_api_key_here
   ```
   The API key is only loaded from this file for security reasons.

2. **Config Files:**
   - Output will go to the `data_raw` directory by default

**Important Note about Users XLSX File Format:**
- The users XLSX file must have the header row with exact column names: `email` and `password`
- Note that both column names are lowercase
- Other additional columns in the XLSX file will be ignored
- Example:
  ```
  email               | password      | 
  ---------------------|--------------|-------
  user@example.com     | password123  | 
  ```

For detailed instructions, see `config/README.md`.

### Usage

#### Using the Pipeline

Run the data extraction pipeline with all data types:

```
python pipeline.py
```

Specify specific data types to collect:

```
python pipeline.py --data-types location mood activity heartrate
```

Specify a custom XLSX file containing user credentials:

```
python pipeline.py --users-file path/to/your/users.xlsx
```

Process a specific user by their UserID:

```
python pipeline.py --user-id 12345
```

Change the logging level:

```
python pipeline.py --log-level DEBUG
```

Combine multiple options:

```
python pipeline.py --data-types location heartrate --users-file custom_users.xlsx --log-level INFO
```

Run data analysis on existing data (skips extraction if data already exists):

```
python pipeline.py --analyze
```

Extract data and run analysis in one command:

```
python pipeline.py --data-types location mood activity heartrate --analyze
```

Note: When using only the `--analyze` flag without specifying `--user-id` or `--data-types`, the pipeline will skip the extraction step if data already exists in the `data_raw` folder. This allows you to run analysis on previously extracted data without re-extracting it.

### Performance Optimizations

The framework includes several optimizations to improve performance:

1. **Parallel Processing**: Multiple users are processed concurrently using thread pools
2. **Caching**: API responses are cached to reduce redundant API calls
3. **Shared Data**: Raw data is shared between collectors to minimize API requests
4. **Timeout Handling**: Automatic timeout for API requests to prevent hanging
5. **Rate Limiting**: Smart handling of API rate limits with exponential backoff
6. **Flexible Data Parsing**: Robust parsing of different data formats and structures



## License

This project is licensed under the MIT License.

## Acknowledgments

- GameBus platform for providing the API
