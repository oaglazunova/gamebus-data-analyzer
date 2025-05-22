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
   - Challenge types and completion rates
   - Movement types and physical activity metrics
   - Heart rate data and patterns

   The analysis process includes:
   - Loading data from Excel files in the `config` directory and JSON files in the `data_raw` directory
   - Generating descriptive statistics 
   - Creating visualizations to provide insights
   - Saving visualizations to the `data_analysis/visualizations` directory
   - Saving statistics to the `data_analysis/statistics` directory

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
   - Optionally include a `UserID` column if you want to process specific users
   - Save this file as `users.xlsx` in the `config/` directory

3. **For data analysis**:
   - Copy campaign description file from GameBus Campaigns website to `config/campaign_desc.xlsx`
   - Copy campaign data file from GameBus Campaigns website to `config/campaign_data.xlsx`

Example of `users.xlsx` format:
```
email               | password      | 
---------------------|--------------|-------
user@example.com     | password123  | 
```

### Step 6: Running the Script

With your virtual environment activated, you can run the script using various commands:

1. **Basic usage** (extracts all data for all users):
   ```
   python pipeline.py
   ```

2. **Process a specific user**:
   ```
   python pipeline.py --user-id 12345
   ```

3. **Run data analysis on existing data**:
   ```
   python pipeline.py --analyze
   ```

4. **Change logging level** (for more or less detailed output):
   ```
   python pipeline.py --log-level DEBUG
   ```

5. **Extract data and run analysis in one command**:
   ```
   python pipeline.py --analyze
   ```

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
