"""
This script processes survey data by performing the following tasks:
1. Loads and preprocesses survey data from a CSV file.
2. Extracts IP-related information (such as city, region, country, etc.) for each response using the IPinfo API.
3. Classifies each response as 'complete', 'partly', or 'empty' based on certain criteria.
4. Cleans the survey data by removing duplicates, incomplete responses, and invalid entries.
5. Extracts and saves contact information and response locations to CSV files.
6. Anonymizes the survey data by removing sensitive information and saves the anonymized data to a CSV file.
7. Calculates and prints the average time to complete a survey.

Usage:
- Copy the env.sample file and save it as .env. Then get an access-token from https://ipinfo.io/ and add it
- Run the script while ensuring you have the right raw survey file selected
"""

import ipinfo
import os
import sys
import pandas as pd
import numpy as np
from alive_progress import alive_bar
from dotenv import load_dotenv
from pathlib import Path

# Define the project and data directories
project_dir = Path(os.path.abspath('')).resolve()
data_dir = project_dir.joinpath("data")
output_dir = data_dir.joinpath("output")


def get_ip_info_for_dataframe(responses: pd.DataFrame):
    """
    Get IP information for each row in the DataFrame using the IPinfo API.

    Args:
        responses (pd.DataFrame): The input DataFrame containing IP addresses.

    Returns:
        pd.DataFrame: The original DataFrame with additional IP-related columns.
    """
    print("For the IP to country conversion to work, copy the .env.sample file to .env and paste the access key there.")
    load_dotenv(project_dir.joinpath(".env"))
    IPINFO_ACCESS_TOKEN = os.getenv("IPINFO_ACCESS_TOKEN")
    
    # Create an IPinfo handler using the access token
    handler = ipinfo.getHandler(IPINFO_ACCESS_TOKEN)
    
    # Use alive_bar to display a progress bar for the operation
    with alive_bar(len(responses)) as update_bar:
        responses[["ip_city", "ip_region", "ip_country", "ip_org", "ip_postal", "ip_timezone"]] = responses.apply(
            get_ip_info, 
            axis=1, 
            result_type='expand', 
            ipinfo_handler=handler,
            update_bar=update_bar
        )
    
    return responses


def get_ip_info(row, ipinfo_handler, update_bar):
    """
    Retrieve IP information for a given row.

    Args:
        row (pd.Series): A row of the DataFrame containing the IP address.
        ipinfo_handler (ipinfo.Handler): The handler for making requests to IPinfo.
        update_bar (alive_progress.alive_bar): The progress bar for updating the status.

    Returns:
        tuple: A tuple containing IP-related information (city, region, country, etc.).
    """
    update_bar()
    
    # Fetch the details for the IP address
    details = ipinfo_handler.getDetails(row.ipaddr)
    
    # Attempt to return relevant IP details or return NA if not available
    try:
        return details.city, details.region, details.country, details.org, details.postal, details.timezone
    except AttributeError:
        return pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA


def get_is_complete(row):
    """
    Determine if the response is complete, partly filled, or empty based on certain conditions.

    Args:
        row (pd.Series): A row from the DataFrame containing response data.

    Returns:
        str: The status of the response ('complete', 'partly', or 'empty').
    """
    if row.submitdate != row.submitdate:
        # If the submitdate is missing, check interview time
        if row.interviewtime > 0:
            return 'partly'
        else:
            return 'empty'
    else:
        return 'complete'


def print_response_locations(raw_survey_df: pd.DataFrame, filename):
    """
    Print and save the count of responses based on their completion status and country.

    Args:
        raw_survey_df (pd.DataFrame): The raw survey DataFrame containing responses.
        filename (str): The name of the file where the results will be saved.
    """
    raw_survey_df['is_complete'] = pd.Categorical(raw_survey_df['is_complete'], ["complete", "partly", "empty"])
    raw_survey_df.sort_values("is_complete", inplace=True)
    raw_survey_df = raw_survey_df[~raw_survey_df['ipaddr'].duplicated()]
    grouped_df = raw_survey_df.groupby(["is_complete", "ip_country"], observed=False).count()["ipaddr"]
    grouped_df.to_csv(filename, sep=';')
    print(f"Saved to CSV: {filename}")


def clean_responses(raw_survey_df: pd.DataFrame):
    """
    Clean the raw survey DataFrame by keeping complete responses and removing duplicates.

    Args:
        raw_survey_df (pd.DataFrame): The raw survey DataFrame.

    Returns:
        pd.DataFrame: The cleaned survey DataFrame with duplicates and incomplete responses removed.
    """
    # Keep only complete responses
    clean_survey_df = raw_survey_df[raw_survey_df['is_complete'] == 'complete']
    
    # Remove hand-checked duplicates
    clean_survey_df = clean_survey_df[~clean_survey_df.index.isin([476, 464, 588, 666])]
    
    # Remove too-fast replies (interview time < 300 seconds)
    clean_survey_df = clean_survey_df[~clean_survey_df.index.isin([659, 600, 577])]
    
    return clean_survey_df


def preprocess(input_file_survey: Path, output_dir: Path):
    """
    Preprocess the raw survey data, extract IP information, clean the data, and save results to CSV.

    Args:
        input_file_survey (Path): Path to the raw survey CSV file.
        output_dir (Path): Directory where the output CSV files will be saved.

    Returns:
        pd.DataFrame: The cleaned and anonymized survey DataFrame.
    """
    # Read the survey CSV file into a DataFrame
    raw_survey_df = pd.read_csv(input_file_survey, sep=';', na_values=['N/A', ''], keep_default_na=False)
    
    # Replace 'None' values in the 'WLplants' column with 'Of none'
    raw_survey_df['WLplants'] = raw_survey_df['WLplants'].replace('None', 'Of none')
    
    # Set 'id' as the index of the DataFrame
    raw_survey_df.set_index("id", inplace=True)
    
    # Determine if each response is complete or not
    raw_survey_df['is_complete'] = raw_survey_df.apply(get_is_complete, axis=1)

    # Get IP information for each row in the DataFrame
    raw_survey_with_ip_info_df = get_ip_info_for_dataframe(raw_survey_df)

    # Define columns related to contacts
    contact_cols = {
        'startlanguage': 'Language',
        'COUDapply[SQ001]': 'Participate in the early adopter group of innovative utilities',
        'COUDapply[SQ002]': 'Contact me with information on further implementation actions, discuss policy suggestions, evaluate scientific developments, etc.',
        'CIsurvey[SQ001]': 'First Name',
        'CIsurvey[SQ002]': 'Surname',
        'CIsurvey[SQ003]': 'Wastewater Association',
        'CIsurvey[SQ004]': 'Email',
        'CIsurvey[SQ005]': 'Phone number'
    }

    # Extract contact information and save to a CSV file
    contacts_df = raw_survey_df[contact_cols.keys()].copy()
    contacts_df = contacts_df[contacts_df.iloc[:, -5:].any(axis=1)]
    contacts_df = contacts_df.rename(columns=contact_cols)
    contacts_file = output_dir.joinpath('contacts.csv')
    contacts_df.to_csv(contacts_file, sep=';', encoding='utf-8-sig')

    # Save response location information to a CSV file
    response_locations_file = output_dir.joinpath('response_locations.csv')
    print_response_locations(raw_survey_with_ip_info_df, response_locations_file)

    # Clean responses and remove unwanted columns
    clean_survey_df = clean_responses(raw_survey_with_ip_info_df)

    sensitive_cols = ['ipaddr', 'CIsurvey[SQ001]', 'CIsurvey[SQ002]', 'CIsurvey[SQ003]', 'CIsurvey[SQ004]', 'CIsurvey[SQ005]',
                      'ip_city', 'ip_org', 'ip_postal', 'ip_timezone']
    clean_anon_survey_df = clean_survey_df.drop(columns=sensitive_cols)

    # Save the anonymized survey data to a CSV file
    clean_anon_file = output_dir.joinpath('clean_anon_survey.csv')
    clean_anon_survey_df.to_csv(clean_anon_file, sep=';', encoding='utf-8-sig')
    print(f"Saved to CSV: {clean_anon_file}")

    # Print the average time to complete a survey
    print('Average time to complete a survey: ', round(clean_survey_df['interviewtime'].mean() / 60, 1), 'min')

    return clean_anon_survey_df


if __name__ == "__main__":
    # Define the input survey file and output directory
    input_file_survey = data_dir.joinpath("results-survey515326.csv")

    # Preprocess the survey data and obtain the cleaned anonymized survey DataFrame
    clean_anon_survey_df = preprocess(input_file_survey, output_dir)