# Co-UD Labs Survey Analysis

## Introduction

This is a Renku project - basically a git repository with some
bells and whistles. You'll find we have already created some
useful things like `data` and `notebooks` directories and
a `Dockerfile`.


## Setup - Export

When exporting the survey responses, you need to select "Question code" and export 
all answers to csv format.
Make sure to name the files and export with "Semicolon" as CSV Field Separator.
This will make it easy to open the file in Excel.
The file will be downloaded with the name:
results-survey515326.csv


## Import
Upload the files and overwrite the existing ones.

Then, add the datasets:

```shell
renku dataset add -o raw-co-ud-survey-515326 data/completed_responses.csv
renku dataset add -o raw-co-ud-survey-515326 data/incomplete_responses.csv
```

Commit the changes before continuing.

### Updating the data

To run a new analysis, update the ip_info CSV file with the following command:

```shell
renku update data/output/ip_info_responses.csv 
```