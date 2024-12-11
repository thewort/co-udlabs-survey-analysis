"""
This script performs data cleaning, categorization, and logistic regression analysis
on the survey data to understand relationships between wastewater organization characteristics
and survey responses. 

Key functionalities:
1. **Data Loading**: Reads survey and question list datasets.
2. **Data Cleaning**: Filters invalid data and replaces complex responses for readability.
3. **Categorization**: Categorizes year and number of employees into defined ranges.
4. **Regression Analysis**:
   - Defines dependent variables (survey responses) and independent variables.
   - Prepares data for regression analysis using logistic regression with K-Fold cross-validation.
   - Evaluates model accuracy and calculates coefficients for each independent variable.
5. **Output Results**: Saves regression analysis results.

Usage: 
Execute the script directly to perform the analysis, assuming required input files 
are present in the specified 'data' directory.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

# Paths and directory setup
PROJECT_DIR = Path().resolve()
DATA_DIR = PROJECT_DIR / "data"

# Constants for year and employee categories
YEAR_SMALL_UNTIL = 1970
YEAR_LARGE_FROM = 1980

NEMPLOYEES_SMALL_UNTIL = 4
NEMPLOYEES_LARGE_FROM = 20

def categorize_year(year):
    """
    Categorizes the given year into one of three categories
    """
    if year < YEAR_SMALL_UNTIL:
        return f'until_{YEAR_SMALL_UNTIL}'
    elif YEAR_SMALL_UNTIL <= year <= YEAR_LARGE_FROM:
        return f' {YEAR_SMALL_UNTIL}_to_{YEAR_LARGE_FROM}' # space before so it drops out in get_dummies
    else:
        return f'{YEAR_LARGE_FROM}_to_now'
    

def categorize_Nemployees(Nemployees):
    """
    Categorizes the given year into one of three categories
    """
    if Nemployees < NEMPLOYEES_SMALL_UNTIL:
        return f'less_than_{NEMPLOYEES_SMALL_UNTIL}'
    elif NEMPLOYEES_SMALL_UNTIL <= Nemployees <= NEMPLOYEES_LARGE_FROM:
        return f'{NEMPLOYEES_SMALL_UNTIL}_to_{NEMPLOYEES_LARGE_FROM}' # space before so it drops out in get_dummies
    else:
        return f'more_than_{NEMPLOYEES_LARGE_FROM}'
    

if __name__ == "__main__":
    # File paths
    survey_file_path = DATA_DIR.joinpath("output/clean_anon_survey.csv")
    questions_file_path = DATA_DIR.joinpath("EN_questionlist.csv")

    # Load datasets
    survey_df = pd.read_csv(survey_file_path, sep=';')
    questions_df = pd.read_csv(questions_file_path, sep=';')

    pd.set_option('display.max_rows', None)

    # Load the dataset
    data = survey_df.copy()

    # Clean Year data
    data = data[data['year'] > 1700]

    # Categorize year
    data['year_category'] = data['year'].apply(categorize_year)
    print(data.groupby('year_category')['id'].count())

    # Categorize Nemployees
    data['Nemployees_category'] = data['Nemployees'].apply(categorize_Nemployees)
    print(data.groupby('Nemployees_category')['id'].count())

    # Replace long answers for readability and simplification
    replacements = {
        'No, but I would like to know more or to do this in future': 'Motivated',
        'Yes, I do uncertainty calculations on my data': 'Motivated',
        "No, I don't think this is needed": 'Unmotivated',
        'I am in favor of this vision, because I generally fin that transparency improves effectiveness and aids innovation': 'Yes',
        'I reject this vision because I think data from wastewater infrastructures should not be openly available': 'No',
        'I support this vision on the following condition:': 'Yes'
        }
    data_clean = data.replace(replacements, regex=False)

    # Dependent variables (Y1, Y2, Y3)
    y_variables = ['ORGshare[SQ004]', 'MEASplants[SQ009]', 'CSOpublic']

    # Independent variables (X1 to X5)
    x_variables = ['year_category', 'Nemployees_category', 'management', 'EMPevent', 'MEunc']

    # select x variables
    x_data = data_clean[x_variables]

    # Keep rows with only one element missing
    x_data = x_data[x_data.count(axis=1) >= x_data.shape[1]-1]
    print(f"removed {data_clean.shape[0] - x_data.shape[0]} rows from x due to too many nans")
    print(f"Xdata size: {x_data.shape}")

    # Impute missing x values with the most frequent element (mode) of each column
    for column in x_data.columns:
        mode_value = x_data[column].mode()[0]  # Get the mode of the column
        x_data[column] = x_data[column].fillna(mode_value)

    # Prepare independent variables (X) 
    X = pd.get_dummies(x_data, drop_first=True)

    results = {}  # Dictionary to store regression results

    for y in y_variables:
        print(f"\n--- Regression Analysis for Dependent Variable: {y} ---")

        # Only use rows where there is a value for Y
        Y = data_clean[y].dropna()
        X_filtered = X.loc[X.index.isin(Y.index)]
        Y_filtered = Y.loc[Y.index.isin(X_filtered.index)]

        # Reset the index of the filtered DataFrame
        X_filtered = X_filtered.reset_index(drop=True)
        Y_filtered = Y_filtered.reset_index(drop=True)
        print(f"\nSample Size: X:{X_filtered.shape[0]} Y:{Y_filtered.shape[0]}")
        
        # Define the dependent variable
        Y_filtered = pd.get_dummies(Y_filtered, drop_first=True)
        
        # Fit a linear regression model
        model = LogisticRegression()

        # Set up K-Fold Cross-Validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Store results
        coefficients = []
        accuracies = []

        # Perform K-Fold Cross-Validation
        for train_index, test_index in kf.split(X_filtered):
            # Split the data
            X_train, X_test = X_filtered.iloc[train_index], X_filtered.iloc[test_index]
            Y_train, Y_test = Y_filtered.iloc[train_index], Y_filtered.iloc[test_index]
            
            # Fit the model
            model.fit(X_train, Y_train.to_numpy().ravel())
            
            # Save coefficients
            coefficients.append(model.coef_[0])  # Coefficients for this fold
            
            # Evaluate accuracy
            accuracy = model.score(X_test, Y_test)
            accuracies.append(accuracy)

        # Convert coefficients to DataFrame for analysis
        coefficients_df = pd.DataFrame(coefficients, columns=X.columns)

        # add to results
        results[y] = coefficients_df.mean().round(2).to_dict()
        results[y]['Accuracy'] = f"{np.round(np.mean(accuracies) * 100, 2)}%"
        results[y]['Sample Size'] = X_filtered.shape[0]

        # Print results
        print("\n Accuracies for each fold:")
        print(accuracies)
        print("Average Accuracy:", np.mean(accuracies))
        print("\nCoefficients for each fold:")
        print(coefficients_df)
        print("\nAverage Coefficients:")
        print(coefficients_df.mean())

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(DATA_DIR.joinpath("output/regression_analysis.csv"),sep=';') 
    
    print("\nCount year category:")
    print(data.groupby('year_category')['id'].count())
    print("\nCount Nemployees category:")
    print(data.groupby('Nemployees_category')['id'].count())
    print("\nRegression Analysis Results:")
    print(results_df)