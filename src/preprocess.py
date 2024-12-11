import ipinfo
import os 
import sys
import pandas as pd
import numpy as np

from alive_progress import alive_bar
from dotenv import load_dotenv
from pathlib import Path

project_dir = Path(os.path.abspath('')).resolve()
data_dir = project_dir.joinpath("data")
output_dir = data_dir.joinpath("output")

def get_ip_info_for_dataframe(responses: pd.DataFrame):
    print("For the ip to country conversion to work, copy the .env.sample file to .env and paste the access key there.")
    load_dotenv("/home/jovyan/work/co-ud-labs-survey-analysis/.env")
    IPINFO_ACCESS_TOKEN = os.getenv("IPINFO_ACCESS_TOKEN")
    handler = ipinfo.getHandler(IPINFO_ACCESS_TOKEN)
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
    update_bar()
    # print(row)
    details = ipinfo_handler.getDetails(row.ipaddr)
    
    try:
        return details.city, details.region, details.country, details.org, details.postal, details.timezone
    except AttributeError:
        return pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA
    
def get_is_complete(row):
    if row.submitdate != row.submitdate:
        if row.interviewtime > 0:
            return 'partly'
        else:
            return 'empty'
    else:
        return 'complete'
    
def print_response_locations(raw_survey_df: pd.DataFrame, filename):
    raw_survey_df['is_complete'] = pd.Categorical(raw_survey_df['is_complete'], ["complete", "partly", "empty"])
    raw_survey_df.sort_values("is_complete", inplace=True)
    raw_survey_df = raw_survey_df[~raw_survey_df['ipaddr'].duplicated()]
    grouped_df = raw_survey_df.groupby(["is_complete","ip_country"], observed = False).count()["ipaddr"]
    grouped_df.to_csv(filename, sep=';')
    print(f"Saved to CSV: {filename}") 

def clean_responses(raw_survey_df: pd.DataFrame):
    # keep complete responses
    clean_survey_df = raw_survey_df[raw_survey_df['is_complete'] == 'complete']

    # remove duplicates (hand-checked)
    clean_survey_df = clean_survey_df[~clean_survey_df.index.isin([476, 464, 588, 666])]

    # remove too fast replies (raw_survey_df['interviewtime'] < 300)
    clean_survey_df = clean_survey_df[~clean_survey_df.index.isin([659, 600, 577])]

    return clean_survey_df

def preprocess(input_file_survey: Path, output_dir: Path):
    raw_survey_df = pd.read_csv(input_file_survey, sep=';', na_values=['N/A', ''], keep_default_na=False)
    raw_survey_df['WLplants'] = raw_survey_df['WLplants'].replace('None', 'Of none')
    raw_survey_df.set_index("id", inplace=True)
    raw_survey_df['is_complete'] = raw_survey_df.apply(get_is_complete, axis=1)

    raw_survey_with_ip_info_df = get_ip_info_for_dataframe(raw_survey_df)
    # raw_survey_with_ip_info_df = raw_survey_df

    contact_cols = {'startlanguage': 'Language',
                    'COUDapply[SQ001]': 'Participate in the early adopter group of innovative utilities',
                    'COUDapply[SQ002]': 'Contact me with information on further implementation actions, discuss policy suggestions, evaluate scientific developments, etc.',
                    'CIsurvey[SQ001]': 'First Name',
                    'CIsurvey[SQ002]': 'Surname',
                    'CIsurvey[SQ003]': 'Wastewater Association',
                    'CIsurvey[SQ004]': 'Email',
                    'CIsurvey[SQ005]': 'Phone number'}

    contacts_df = raw_survey_df[contact_cols.keys()].copy()
    contacts_df = contacts_df[contacts_df.iloc[:,-5:].any(axis=1)]
    contacts_df = contacts_df.rename(columns=contact_cols)
    contacts_file = output_dir.joinpath('contacts.csv')
    contacts_df.to_csv(contacts_file, sep=';', encoding='utf-8-sig')

    response_locations_file = output_dir.joinpath('response_locations.csv')
    print_response_locations(raw_survey_with_ip_info_df, response_locations_file)
    
    clean_survey_df = clean_responses(raw_survey_with_ip_info_df)

    sensitive_cols = ['ipaddr', 'CIsurvey[SQ001]', 'CIsurvey[SQ002]', 'CIsurvey[SQ003]', 'CIsurvey[SQ004]', 'CIsurvey[SQ005]', 'ip_city', 'ip_org', 'ip_postal', 'ip_timezone']
    clean_anon_survey_df = clean_survey_df.drop(columns=sensitive_cols)
    clean_anon_file = output_dir.joinpath('clean_anon_survey.csv')
    clean_anon_survey_df.to_csv(clean_anon_file, sep=';', encoding='utf-8-sig')
    print(f"Saved to CSV: {clean_anon_file}") 

    print('Average time to complete a survey: ', round(clean_survey_df['interviewtime'].mean()/60,1), 'min')

    return clean_anon_survey_df

    
if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print("Needs exactly 2 arguments!")
    #     print("Please provide the paths to the responses CSV and the output directory.")
    #     exit(1)

    input_file_survey = data_dir.joinpath("results-survey515326.csv")
    # input_file_survey = output_dir.joinpath("raw_survey.csv")

    clean_anon_survey_df = preprocess(input_file_survey, output_dir)

