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

## Script Workflow

1. **Data Extraction**: Extract raw data from the GameBus API:

   The extraction process includes:
   - Authentication with the GameBus API
   - Retrieving player data for multiple users in parallel
   - Parsing and processing the raw data
   - Saving the processed data to JSON files in the `data_raw` directory

2. **Data Analysis**: Analyze the extracted data to generate insights, including:
   - Activity patterns and distributions
   - User engagement and participation
   - Movement types and physical activity metrics
   - Heart rate data and patterns

   The analysis process includes:
   - Loading data from Excel files in the `config` directory and JSON files in the `data_raw` directory
   - Generating descriptive statistics 
   - Creating visualizations to provide insights
   - Saving visualizations to the `data_analysis/visualizations` directory
   - Saving statistics to the `data_analysis/statistics` directory

## Available Analyses

The following analyses are available in the GameBus Data Analyzer:

### Activity Analysis

| Analysis | Description | Data Source | Game Descriptors/Parameters |
|----------|-------------|-------------|---------------------------|
| Activity Types Distribution | Shows the frequency of different activity types | campaign_data.xlsx (activities sheet) | 'type' column |
| Activities Over Time | Shows the number of activities recorded each day | campaign_data.xlsx (activities sheet) | 'createdAt' column |
| Points by Activity Type | Shows the total points awarded for each activity type | campaign_data.xlsx (activities sheet) | 'type', 'rewardedParticipations' columns |
| User Activity Distribution | Shows the number of activities recorded by each user | campaign_data.xlsx (activities sheet) | 'pid' column |
| Activity Type by User | Shows how different users engage with different activity types | campaign_data.xlsx (activities sheet) | 'pid', 'type' columns |
| Dropout Rates | Shows the distribution of time between first and last activity for each user | campaign_data.xlsx (activities sheet) | 'createdAt', 'pid' columns |
| Active vs Passive Usage | Shows the distribution of app usage time between active and passive usage (Note: This analysis is only performed if VISIT_APP_PAGE events with DURATION_MS parameter are available) | campaign_data.xlsx (activities sheet), VISIT_APP_PAGE events with DURATION_MS parameter | 'createdAt', 'pid', 'type', 'properties' columns |
| Usage by Day of Week | Shows the number of activities by day of week | campaign_data.xlsx (activities sheet) | 'createdAt', 'date' columns |
| Activity Heatmap by Time | Shows when users are most active during the week | campaign_data.xlsx (activities sheet) | 'createdAt', 'hour', 'day_of_week' columns |
| Activity Types Over Time | Shows how different types of activities contribute to overall engagement | campaign_data.xlsx (activities sheet) | 'date', 'type' columns |
| User Engagement Heatmap | Shows the engagement patterns of different users over time | campaign_data.xlsx (activities sheet) | 'pid', 'date' columns |

### User Data Analysis

| Analysis | Description | Data Source | Game Descriptors/Parameters |
|----------|-------------|-------------|---------------------------|
| Movement Type Distribution | Shows the proportion of different movement types | JSON files in data_raw directory | WALK, RUN, BIKE, TRANSPORT, PHYSICAL_ACTIVITY, GENERAL_ACTIVITY |
| Calories by Movement Type | Shows the average calories burned for each movement type | JSON files in data_raw directory | WALK, RUN, BIKE, TRANSPORT, PHYSICAL_ACTIVITY, GENERAL_ACTIVITY with KCALORIES parameter |
| Steps Trend | Shows the distribution and trend of steps over time | JSON files in data_raw directory | WALK, RUN, GENERAL_ACTIVITY with STEPS or STEPS_SUM parameter |
| Calories Trend | Shows the distribution and trend of calories burned over time | JSON files in data_raw directory | WALK, RUN, BIKE with KCALORIES parameter |
| Activity Providers | Shows the distribution of activities by provider/source | JSON files in data_raw directory | 'data_provider_name' column in activity data |
| App Page Visits | Shows the distribution of app page visits | JSON files in data_raw directory | VISIT_APP_PAGE with APP_NAME, PAGE_NAME parameters |
| Heart Rate Distribution | Shows the overall distribution of heart rate measurements | JSON files in data_raw directory | LOG_MOOD with heart rate data |
| Heart Rate Over Time | Shows how heart rate changes over time for a specific activity | JSON files in data_raw directory | LOG_MOOD with heart rate and timestamp data |
| Heart Rate by Hour | Shows how heart rate varies by hour of day | JSON files in data_raw directory | LOG_MOOD with heart rate and timestamp data |

### Challenge Analysis

| Analysis | Description | Data Source | Game Descriptors/Parameters |
|----------|-------------|-------------|---------------------------|
| Challenge Levels Distribution | Shows the distribution of challenges by level | campaign_desc.xlsx (challenges sheet) | 'level', 'is_initial_level', 'success_next' columns |
| Challenge Types Distribution | Shows the frequency of different challenge types | campaign_desc.xlsx (challenges sheet) | 'type' column |
| Challenge Completions | Shows the number of completions for each challenge | campaign_data.xlsx (activities sheet) | SCORE_GAMEBUS_POINTS with FOR_CHALLENGE parameter |
| Challenge Completions Over Time | Shows how challenge completions are distributed over time | campaign_data.xlsx (activities sheet) | SCORE_GAMEBUS_POINTS with FOR_CHALLENGE parameter, 'createdAt' column |
| Challenges Ranked by Points | Shows challenges ranked by the total points earned | campaign_data.xlsx (activities sheet) | SCORE_GAMEBUS_POINTS with FOR_CHALLENGE, NUMBER_OF_POINTS parameters |
| Challenges Ranked by Users | Shows challenges ranked by the number of users completing them | campaign_data.xlsx (activities sheet) | SCORE_GAMEBUS_POINTS with FOR_CHALLENGE parameter, 'pid' column |
| User Challenge Points Heatmap | Shows a heatmap of points earned by users for each challenge | campaign_data.xlsx (activities sheet) | SCORE_GAMEBUS_POINTS with FOR_CHALLENGE, NUMBER_OF_POINTS parameters, 'pid' column |
| Task Completion Frequency | Shows the frequency of task completions over time | campaign_data.xlsx (activities sheet) | SCORE_GAMEBUS_POINTS with FOR_CHALLENGE parameter, 'createdAt' column |
| Daily Task Completion | Shows the number of tasks completed per day | campaign_data.xlsx (activities sheet) | SCORE_GAMEBUS_POINTS with FOR_CHALLENGE parameter, 'createdAt' column |


## Complete Setup Guide

### Step 1: Install Python

If you don't have Python installed yet, follow these steps:

1. **Download Python**:
   - Go to [python.org](https://www.python.org/downloads/)
   - Click on the "Download Python" button (download version 3.8 or newer)
   - Make sure to check the box that says "Add Python to PATH" during installation

2. **Verify Installation**:
   - Open Command Prompt (search for "cmd" in the Start menu)
   - Type `python --version` and press Enter
   - You should see the Python version displayed (e.g., "Python 3.10.0")
   - If you see an error, try typing `py --version` instead

### Step 2: Download the Project

1. **Using Git** (recommended if you have Git installed):
   - Open Command Prompt
   - Navigate to the folder where you want to store the project
   - Type: `git clone https://github.com/username/gamebus-data-analyzer.git`
   - Navigate into the project folder: `cd gamebus-data-analyzer`

2. **Without Git**:
   - Download the project as a ZIP file from the repository
   - Extract the ZIP file to a location on your computer
   - Open Command Prompt
   - Navigate to the extracted folder location

### Step 3: Set Up a Virtual Environment (Recommended)

Creating a virtual environment keeps your project dependencies separate from other Python projects:

1. **Create a virtual environment**:
   - In Command Prompt, navigate to the project folder
   - Type: `python -m venv .venv`
   - This creates a virtual environment in a folder named `.venv`

2. **Activate the virtual environment**:
   - On Windows: `.venv\Scripts\activate`
   - You'll see (.venv) at the beginning of your command prompt line when it's activated

### Step 4: Install Dependencies

1. With your virtual environment activated, install all required packages:
   ```
   pip install -r requirements.txt
   ```

2. This will install all necessary Python libraries including:
   - pandas: For data manipulation
   - numpy: For numerical operations
   - matplotlib and seaborn: For creating visualizations
   - requests: For API communication
   - python-dotenv: For loading environment variables
   - And other dependencies

### Step 5: Configure the Project

1. **Set up API credentials**:
   - Create a text file named `.env` in the project's root directory
   - Add your GameBus API key to this file:
     ```
     GAMEBUS_API_KEY=your_api_key_here
     ```
   - Save the file

2. **Prepare user data**:
   - Create an Excel file with user credentials
   - The file must have columns named `email` and `password` (both lowercase)
   - Optionally include a `UserID` column for reference (not used for filtering)
   - Save this file as `users.xlsx` in the `config/` directory

3. **For data analysis**:
   - Copy campaign description file from GameBus Campaigns website to `config/campaign_desc.xlsx`
   - Copy campaign data file from GameBus Campaigns website to `config/campaign_data.xlsx`

Example of `users.xlsx` format:
```
email               | password      | UserID (optional)
---------------------|--------------|----------------
user@example.com     | password123  | 12345
```

### Step 6: Running the Script

With your virtual environment activated, you can run the script using various commands:

1. **Basic usage** (extracts all data for all users and runs analysis):
   ```
   python pipeline.py
   ```

2. **Only extract data** (extracts all data for all users without running analysis):
   ```
   python pipeline.py --extract
   ```

3. **Only run data analysis** on existing data (without extracting new data):
   ```
   python pipeline.py --analyze
   ```

4. **Change logging level** (for more or less detailed output):
   ```
   python pipeline.py --log-level DEBUG
   ```

5. **Generate test data** (creates synthetic test data for development and testing):
   ```
   python src\scripts\generate_test_data.py
   ```
   This script generates test data for multiple users with various game descriptors. It uses property schemes from the `config\property_schemes.py` file to enhance data generation. The script:
   - Loads property schemes from the config file
   - Uses these schemes to determine appropriate parameters for each game descriptor
   - Generates random values based on property types, units, and other attributes
   - Saves the generated data to JSON files in the `data_raw` directory

   If the config file doesn't contain information for a particular game descriptor, the script falls back to hardcoded parameters.

   To update the property schemes in the config file from the Excel file:
   ```
   python src\scripts\update_property_schemes.py
   ```
   This script reads the `HW8-database-bootstrap.xlsx` file in the `api docs` folder and updates the `config\property_schemes.py` file with the extracted property schemes. This is only needed when the Excel file has been updated with new property schemes.

### Troubleshooting Common Issues

1. **"Python is not recognized as an internal or external command"**:
   - Python wasn't added to your PATH during installation
   - Solution: Reinstall Python and check "Add Python to PATH"

2. **"No module named 'package_name'"**:
   - A required package is missing
   - Solution: Make sure you've activated your virtual environment and run `pip install -r requirements.txt`

3. **"Could not find a version that satisfies the requirement"**:
   - You might be using an incompatible Python version
   - Solution: Make sure you're using Python 3.8 or newer

4. **"Error loading .env file"**:
   - The .env file is missing or in the wrong location
   - Solution: Create the .env file in the project's root directory

5. **"Failed to get token for user"**:
   - API credentials might be incorrect
   - Solution: Check your API key in the .env file and user credentials in users.xlsx

### Performance Optimizations

The script includes several optimizations to improve performance:

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
