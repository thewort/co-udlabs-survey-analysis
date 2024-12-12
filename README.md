# Co-UD Labs Survey Analysis

## Introduction

This is the evaluation of a survey among wastewater associations conducted in the
course of Co-UDlabs:


Deliverable D2.5 - Report on smart governance and public access to data
DOI: [10.5281/zenodo.14280836](https://doi.org/10.5281/zenodo.14280836)


## Setup - Export

When exporting the survey responses from LimeSurvey, you need to select 
"Question code" and export all answers to CSV format.
Make sure to name the files and export with "Semicolon" as CSV Field Separator.
This will make it easy to open the file in Excel.
The file will be downloaded with the name:
results-survey515326.csv
Paste it into the directory "data".

Install the necessary packages from requirements.txt (Python version 3.12.3)

To get the IP-adresses in the preprocessing, it is necessary to copy the 
env.sample file, save it as .env and replace the token by your own that
you can get from ipinfo.io.

## Preprocessing

After getting the raw data from LimeSurvey as described above, you can use
the preprocess.py script to get the IP-location of the respondents and clean
the data. These files are generated:

* contacts.csv: names and adresses of the respondents who gave it
* response_locations.csv: summary of the IP-origins
* clean_anon_survey.csv: cleaned dataset without sensitive info

The notebook countries_overview.ipynb conducts some further analysis on the
origin of the responses and compares different countries.

## Generate Appendix

generate_appendix.py takes the clean data and question list and generates 
an appendix for each country as well as all of them together. The output
are a folder for each appendix containing plots (PDF) and CSV files for 
each question and a TEX-file to embed them all in a Latex-Document. Use
Overleaf to generate the document.

## Regression Analysis

regression_analysis.py performs a logistic regression as an explanatory
analysis to investigate the influence of different wastewater 
association characteristics on attitudes of survey respondents. It uses
clean_anon_survey.csv and generates the output regression_analysis.csv

## Plots for the Deliverable

These are produced by the jupyter notebook deliverable_plots.ipynb from 
the cleaned data.
