"""
Script to update property schemes in the config file from the Excel file.
This script reads the HW8-database-bootstrap.xlsx file and updates the
property_schemes.py config file with the extracted data.
"""

import os
import sys
import pandas as pd
import pprint

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Add the project root to the Python path
sys.path.insert(0, PROJECT_ROOT)

# Path to the Excel file with property schemes
EXCEL_FILE_PATH = os.path.join(PROJECT_ROOT, 'api docs', 'HW8-database-bootstrap.xlsx')

# Path to the config file
CONFIG_FILE_PATH = os.path.join(PROJECT_ROOT, 'config', 'property_schemes.py')

def extract_property_schemes_from_excel():
    """
    Extract property schemes from the Excel file.

    Returns:
        Tuple containing:
        - Dictionary mapping activity schemes to their property schemes
        - Dictionary mapping property names to their details (type, unit, etc.)
    """
    print(f"Extracting property schemes from Excel file: {EXCEL_FILE_PATH}")

    try:
        # Read the Excel file
        excel_file = pd.ExcelFile(EXCEL_FILE_PATH)

        # Read the activityschemes sheet
        activity_df = pd.read_excel(excel_file, sheet_name='activityschemes')

        # Read the propertyschemes sheet
        property_df = pd.read_excel(excel_file, sheet_name='propertyschemes')

        # Create a dictionary mapping activity schemes to their property schemes
        activity_to_properties = {}
        for _, row in activity_df.iterrows():
            activity_name = row['name']
            if pd.notna(row['propertyschemes']):
                # Extract property schemes from the cell
                properties = [prop.strip() for prop in str(row['propertyschemes']).split(',')]
                activity_to_properties[activity_name] = properties

        # Create a dictionary mapping property names to their details
        property_details = {}
        for _, row in property_df.iterrows():
            # Skip rows with NaN property names
            if pd.isna(row['name']):
                continue

            property_name = row['name']
            property_details[property_name] = {
                'type': row['type'] if pd.notna(row['type']) else 'STRING',
                'unit': row['unit'] if pd.notna(row['unit']) else None,
                'values': row['values'] if pd.notna(row['values']) else None,
                'input_as': row['input_as'] if pd.notna(row['input_as']) else None
            }

        print(f"Extracted {len(activity_to_properties)} activity schemes and {len(property_details)} property schemes")
        return activity_to_properties, property_details

    except Exception as e:
        print(f"Error extracting property schemes from Excel: {e}")
        return {}, {}

def update_config_file(activity_to_properties, property_details):
    """
    Update the config file with the extracted property schemes.

    Args:
        activity_to_properties: Dictionary mapping activity schemes to their property schemes
        property_details: Dictionary mapping property names to their details
    """
    print(f"Updating config file: {CONFIG_FILE_PATH}")

    try:
        # Create the content for the config file
        content = [
            '"""',
            'Property schemes configuration for the GameBus data analyzer.',
            'This file contains property schemes extracted from the HW8-database-bootstrap.xlsx file.',
            '"""',
            '',
            '# Dictionary mapping activity schemes to their property schemes',
            f'ACTIVITY_TO_PROPERTIES = {pprint.pformat(activity_to_properties, width=120)}',
            '',
            '# Dictionary mapping property names to their details (type, unit, etc.)',
            f'PROPERTY_DETAILS = {pprint.pformat(property_details, width=120)}',
            ''
        ]

        # Write the content to the config file
        with open(CONFIG_FILE_PATH, 'w') as f:
            f.write('\n'.join(content))

        print(f"Config file updated successfully")

    except Exception as e:
        print(f"Error updating config file: {e}")

def main():
    """
    Extract property schemes from the Excel file and update the config file.
    """
    print("Updating property schemes in the config file...")

    # Extract property schemes from the Excel file
    activity_to_properties, property_details = extract_property_schemes_from_excel()

    # Update the config file
    update_config_file(activity_to_properties, property_details)

    print("Property schemes update complete!")

if __name__ == "__main__":
    main()
