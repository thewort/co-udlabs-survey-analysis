"""
Survey Data Processing and Appendix Generation

This script processes survey data and generates visualizations, tables, 
and a LaTeX-based appendix for different countries. It reads cleaned survey data, 
filters relevant questions, creates plots and tables for individual questions, 
and compiles them into a country-specific appendix.

Key Features:
- Handles multiple question types (e.g., bar charts, histograms, text responses).
- Supports multi-question types and ranking calculations.
- Generates LaTeX documents with figures and tables.
- Configurable for specific countries or all available data.

Inputs:
- Cleaned survey data in CSV format.
- Question list files for each country in CSV format.

Outputs:
- Visualizations saved as PDF or CSV in the `appendix` directory.
- LaTeX files for each country with all the processed data.

Usage:
1. Place the cleaned survey data in the `data/output` directory.
2. Ensure question lists for each country are available in the `data` directory.
3. Configure the `COUNTRIES` set if needed; leave it empty to process all countries.
4. Run the script. Outputs are generated in the `data/output/appendix` directory.

Dependencies:
- pandas
- altair
- pylatex
"""

import os
from pathlib import Path
from textwrap import wrap
import pandas as pd
pd.set_option('future.no_silent_downcasting', True) 
import altair as alt
from pylatex import Document, Figure, Table, Tabular, Package, NoEscape

# Paths and directory setup
PROJECT_DIR = Path(os.path.abspath("")).resolve()
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = DATA_DIR / "output"
APPENDIX_DIR = OUTPUT_DIR / "appendix"

# Input file paths
CLEANED_CSV = OUTPUT_DIR / "clean_anon_survey.csv"

# Configuration
COUNTRIES = {"CH", "FR", "DK", "ES"}  # Leave empty to select all available countries
COUNTRY_LANGUAGE = {
    "CH": "DE", "DE": "DE", "DK": "DK", "ES": "ES",
    "FR": "FR", "BE": "NL", "LU": "NL", "ALL": "EN"
}
BAR_COLOR = "#1f78b4"
SINGLE_Q_TYPES = [
    "likert", "bool", "number", "singleselect", "singleselect_text",
    "multiselect_text", "text", "3multiselect_text"
]
MULTI_Q_TYPES = ["multiselect", "multiselect_text", "3multiselect", "3multiselect_text", "rank_text"]

# LaTeX configuration
GEOMETRY_OPTIONS = {"tmargin": "1cm", "lmargin": "2cm"}


def get_percent_df(source_df: pd.DataFrame, code: str) -> pd.DataFrame:
    """
    Calculate percentage distribution of values in a DataFrame column.
    """
    percent_df = source_df.groupby([code]).size() / source_df.shape[0]
    percent_df = percent_df.reset_index(name="percent")
    percent_df["labels"] = percent_df["percent"].apply(lambda x: f"{round(x * 100, 1)}%")
    return percent_df


def get_nlabel(n: int, country: str) -> str:
    """
    Create a label indicating the number of responses (N) and the country.
    """
    return f"{country}, N = {n}" if country else f"N = {n}"


def get_file_name(question: dict, country: str) -> Path:
    """
    Generate file name for charts or data exports based on question and country.
    """
    ext = "csv" if "text" in question["QuestionType"] else "pdf"
    filename = f"{country}/{country}_Q{question['Question_ID']}_G{question['Group_ID']}_.{ext}"
    return APPENDIX_DIR / filename


def get_tex_file_name(question_id, group_id, country, question_type):
    """
    Get the file name that will be referred to in the TEX file to embed plots
    """
    ext = "csv" if "text" in question_type else "pdf"
    filename = f"{country}_Q{question_id}_G{group_id}_.{ext}"
    return filename


# Visualization Functions
def v_bar_chart(source_df: pd.DataFrame, question: dict, country: str):
    """
    Create a vertical bar chart and save it to the appropriate file.
    """
    code = question['Code']
    title_wrapped = wrap(question['ShortQuestion'], 40)
    answers_sort = question['Answers'].split('$')

    # Filter the dataframe for non-null values in the current code column
    source_df = source_df[~source_df[code].isna()]

    # Calculate percentages
    percent_df = get_percent_df(source_df, code)

    # Append N label to the title
    full_title = title_wrapped + [f"N = {source_df.shape[0]}, {country}"]

    # Create the bar chart
    bars = alt.Chart(percent_df, title=full_title).mark_bar(color=BAR_COLOR).encode(
        x=alt.X(f"{code}:N", scale=alt.Scale()).axis(labelLimit=500, title='', labelAngle=0).sort(answers_sort),
        y=alt.Y("percent:Q", scale=alt.Scale(domain=[0, 1])).axis(format='%', title=''),
        color=alt.Color(legend=None)
    ).properties(
        width=100,
        height=150 
    )

    # Add percentage labels to the bars
    text = bars.mark_text(align='left', dx=-15, dy=-10).encode(
        x=alt.X(f"{code}:N").sort(answers_sort),
        color=alt.Color(legend=None),
        text=alt.Text('labels')
    )

    # Save chart
    (bars + text).resolve_scale(color='independent').save(get_file_name(question, country))


def h_bar_chart(source_df: pd.DataFrame, question: dict, country: str):
    """
    Create a horizontal bar chart and save it to the appropriate file.
    """
    code = question['Code']
    title_wrapped = wrap(question['ShortQuestion'], 70)
    answers_sort = question['Answers']
    try:
        answers_sort = answers_sort.split('$') 
    except:
        answers_sort=[]

    # Filter out NaN values in the column
    source_df = source_df[~source_df[code].isna()]
    
    # Generate Label
    full_title = title_wrapped + [get_nlabel(source_df.shape[0], country)]

    # Calculate percentages
    percent_df = get_percent_df(source_df, code)
    
    # Create bar chart
    bars = alt.Chart(percent_df, title=full_title).mark_bar(color=BAR_COLOR).encode(
        x=alt.X("percent:Q").axis(format='%', title=''),
        y=alt.Y(f"{code}:N", scale=alt.Scale()).axis(labelLimit=500, title='', labelAngle=0).sort(answers_sort),
        color=alt.Color(legend=None)
    ).properties(
        height=150 
    )

    # Add percentage labels
    text = bars.mark_text(align="left", dx=2).encode(
        text=alt.Text("labels"),
        x=alt.X("percent:Q"),
        color=alt.Color(legend=None)
    )

    # Save Chart
    (bars + text).resolve_scale(color='independent').save(get_file_name(question, country))


def number_histogram(source_df: pd.DataFrame, question: dict, country: str):
    """
    Create a Histogram with vertical bars and save it to the appropriate file.
    """
    code = question['Code']
    title = question['WrappedQuestion']

    # Ensure data is numeric and filter nans
    source_df[code] = pd.to_numeric(source_df[code], errors='coerce')
    source_df = source_df[~source_df[code].isna()]

    # Generate label
    nlabel = get_nlabel(source_df.shape[0], country)
    title = title + [''] + [nlabel]

    binstep = 10

    # Special adjustments to handle exceptions in the data
    if code == 'year':
        source_df = source_df[source_df['year'] > 1000]
    elif code in ['Npluviom', 'IDPyear']:
        binstep = 1
    elif code == 'Ncommunities':
        source_df = source_df[source_df['Ncommunities'] < 1000]
    elif code == 'AREAtotal' and country != 'BE':
        source_df = source_df[source_df['AREAtotal'] < 100000]
        binstep = 1000

    # Define min and max of the x-axis
    min = source_df[code].min()
    max = source_df[code].max()

    # Create and save chart
    alt.Chart(source_df, title=title).mark_bar().encode(
        x= alt.X(f"{code}:Q", bin=alt.Bin(extent=[min, max], step=binstep)),
        y='count()',
    ).properties(
        height=150 
    ).save(get_file_name(question, country))


def answers_text(source_df: pd.DataFrame, question: dict, country: str):
    """
    Collects answers to a question and prints them to CSV
    """
    code = question['Code']

    source_df = source_df[~source_df[code].isna()]
    text_df = source_df[code]

    text_df.to_csv(get_file_name(question, country), sep=';')


def multi_q_plot (source_df: pd.DataFrame, multi_q_df: pd.DataFrame, multi_q_i, country):
    """
    Handles questions that have span several rows in the questions_df (from EN_questionlist.csv)
    """

    multi_q_df = multi_q_df[multi_q_df['Question_ID'] == multi_q_i]
    q_types = set(multi_q_df['QuestionType'])

    # Determine whether to eliminate answers that do not comply with "select only three"
    only3 = True if '3multiselect' in q_types else False
        
    # Special case for rank calculation
    if 'rank_text' in q_types:
        calculate_rank(source_df, multi_q_df, country)
    else:
        multiselect_bar_chart(source_df, multi_q_df, country, only3)

        
def multiselect_bar_chart(source_df: pd.DataFrame, questions_df: pd.DataFrame, country: str, only3=False):
    """
    Create a horizontal bar chart for questions where multiple answers can be selected.
    """
    question = questions_df.iloc[0].to_dict()
    question_cols = questions_df['Code'].to_list()
    print(question_cols)
    title = question['WrappedQuestion']
    answers_sort = questions_df['Answers'].to_list()

    # Filter out columns and answers without values, then replace 'No' with nan
    # so that the 'other' column counts as yes
    source_df = source_df[question_cols]
    source_df = source_df[source_df.any(axis=1)]
    source_df = source_df.replace('No', pd.NA)

    # Filter out answers that do not comply with "select only three" if necessary
    if only3:
        notvalid_answers_df = source_df[source_df.count(axis='columns') > 3]
        if not notvalid_answers_df.empty:
            print('removed {0} answers with too many options ticked'.format(notvalid_answers_df.shape[0]))
            source_df = source_df[source_df.count(axis='columns') <= 3]

    # Calculate percentage of each column
    percent_df = source_df.count()
    percent_df = percent_df.reset_index(name='count')
    percent_df = percent_df.rename(columns={'index': 'Code'})
    percent_df['percent'] = percent_df['count'] / source_df.shape[0] 
    percent_df['labels'] = percent_df['percent'].apply(lambda x : str(round(x * 100,1)) + '%')
    percent_df = percent_df.merge(questions_df[['Code', 'Answers']], how='left', on='Code')

    # get label
    title = title + [f"N = {source_df.shape[0]}, {country}"]

    # Create bar chart
    bars = alt.Chart(percent_df, title=title).mark_bar(color='#1f78b4').encode(
        x=alt.X("percent:Q").axis(format='%', title=''),
        y=alt.Y('Answers').axis(labelLimit=500, title='').sort(answers_sort),
        color=alt.Color(legend=None)
    ).properties(
        height=150 
    )

    # Add percentage labels
    text = bars.mark_text(align='left', dx=2).encode(
        x=alt.X('percent:Q'),
        color=alt.Color(legend=None),
        text=alt.Text('labels'))

    # Save chart
    (bars + text).properties(
        title=alt.TitleParams(
            text=title,
            anchor="start"  # Align the title to the left
        )
    ).resolve_scale(color='independent').save(get_file_name(question, country))


def calculate_rank(source_df: pd.DataFrame, questions_df: pd.DataFrame, country: str):
    """
    Calculates the ranking of a policy using the first 3 priorities of an answer and weighing them
    and saves it to CSV
    """
    question = questions_df.iloc[0].to_dict()
    question_cols  = ['CSOmonit[1]', 'CSOmonit[2]', 'CSOmonit[3]']
    print(question_cols)

    source_df = source_df[question_cols]
    source_df = source_df.assign(ones=1)
    rank_df = pd.DataFrame(columns=['Policy'])

    # Count the occurrence of a Policy
    for i in range(3):
        col_code = question_cols[i]
        rank_i_df = source_df[[col_code,'ones']].groupby(by=col_code).count()
        rank_i_df.index.rename('Policy', inplace=True)
        rank_i_df.rename(columns={"ones": 'Rank {}'.format(i+1)}, inplace=True)
        rank_i_df['Rank {}'.format(i+1)] = rank_i_df['Rank {}'.format(i+1)].astype('Int32')
        rank_df = rank_df.merge(rank_i_df, on='Policy', how='outer')

    # Calculate the rank
    rank_df = round(rank_df.fillna(value=0))
    rank_df['Score'] = 3 * rank_df['Rank 1'] + 2 * rank_df['Rank 2'] + rank_df['Rank 3']
    rank_df['Normalized score'] = round(rank_df['Score'] / rank_df['Score'].max(), 2)
    rank_df = rank_df.sort_values(by='Score', ascending=False, ignore_index=True)
    rank_df.drop(labels='Score', axis=1, inplace=True)

    # Save to CSV
    rank_df.to_csv(get_file_name(question, country), sep=';', index=False)


def create_tex(questions_df: pd.DataFrame, country: str, country_dir: Path):
        """
        Generates a TEX file to embed the plots and tables and saves it
        """
        groups_df = questions_df[['Group_ID','Group']]
        groups_df = groups_df.groupby(['Group_ID','Group']).count()
        groups_df = groups_df.reset_index()
        groups_dict = groups_df.to_dict('records')

        print(f"Generating TEX file for {country}")
        doc = Document(geometry_options=GEOMETRY_OPTIONS)
        doc.packages.append((Package("morefloats", options=["morefloats=500"])))
        doc.packages.append((Package("hyperref")))
        doc.append(NoEscape(r"\maxdeadcycles=1000"))
        doc.append(NoEscape(r"\listoffigures"))
        doc.append(NoEscape(r"\listoftables"))

        # Iterate through Groups and questions
        for group in groups_dict:
            group_questions_df = questions_df[questions_df['Group'] == group['Group']]
            for question_id in set(group_questions_df['Question_ID']):
                question_df = questions_df[questions_df['Question_ID'] == question_id]
                question_text = "; ".join(str(x) for x in set(question_df['ShortQuestion']))
                for question_type in set(question_df['QuestionType']):
                    filename =  get_tex_file_name(question_id, group['Group_ID'], country, question_type)
                    
                    # Embed plot
                    if 'text' not in question_type:
                        with doc.create(Figure(position="h!")) as figure:
                            if 'select' in question_type:
                                width = "450px"
                            elif question_type == 'bool':
                                width="150px"
                            else:
                                width="250px"
                            figure.add_image(filename, width=width)
                            figure.add_caption(question_text)

                    # Make rank table
                    elif 'rank' in question_type:
                        answers_df = pd.read_csv(country_dir.joinpath(filename), sep=';')
                        with doc.create(Table(position="h!")) as table:
                            with doc.create(Tabular("|p{10cm}|c|c|c|r|")) as tabular:
                                tabular.add_hline()
                                tabular.add_row(answers_df.columns)
                                tabular.add_hline()
                                tabular.add_hline()
                                for answer in answers_df.to_dict('records'):
                                    tabular.add_row(answer.values())
                                    tabular.add_hline()
                            table.add_caption([question_text])

                    # Make table for open questions
                    else:
                        answers_df = pd.read_csv(country_dir.joinpath(filename), sep=';')
                        with doc.create(Table(position="h!")) as table:
                            with doc.create(Tabular("|p{15cm}|")) as tabular:
                                tabular.add_hline()
                                tabular.add_row([question_text])
                                tabular.add_hline()
                                tabular.add_hline()
                                for answer in answers_df.iloc[:,-1:].values:
                                    tabular.add_row(answer)
                                    tabular.add_hline()
                            table.add_caption([question_text])

        # Generate and save to country folder
        doc.generate_tex(os.path.join(country_dir, f"appendix_{country}"))

def create_plots_and_tables(questions_df: pd.DataFrame, survey_country_df: pd.DataFrame, country: str):
        """
        Iterate through questions and create plots or do calculations according to type
        """

        # Iterate through questions with a single row in questions_df
        for question in questions_df[questions_df['QuestionType'].isin(SINGLE_Q_TYPES)].to_dict('records'):
            print(question['Code'], end='\n')
            if question['QuestionType'] == 'bool':
                v_bar_chart(survey_country_df, question, country)
            elif question['QuestionType'] in ['singleselect', 'likert']:
                h_bar_chart(survey_country_df, question, country)
            elif question['QuestionType'] == 'number':
                number_histogram(survey_country_df, question, country)
            elif question['QuestionType'] in ['singleselect_text', 'multiselect_text', 'text','3multiselect_text']:
                answers_text(survey_country_df, question, country)

        # Iterate through questions with a multiple in questions_df
        multi_q_df = questions_df[questions_df['QuestionType'].isin(MULTI_Q_TYPES)]
        multi_q_is = set(multi_q_df['Question_ID'])
        for i in multi_q_is:
            print(i, end=': ')
            multi_q_plot(survey_country_df, multi_q_df.copy(), i, country)

if __name__ == "__main__":      
    # load survey data
    survey_df = pd.read_csv(CLEANED_CSV, sep=';')

    # create appendix folder if necessary
    if not os.path.exists(APPENDIX_DIR):
        os.mkdir(APPENDIX_DIR)

    # if no countries selected manually, select all available in survey data
    if not COUNTRIES:
        COUNTRIES = set(survey_df['ip_country'])
    COUNTRIES.add('ALL')

    print(f'Creating appendix for these countries:\n{COUNTRIES}')

    for country in COUNTRIES:
        #read questionlist for language
        questions_csv = DATA_DIR.joinpath(f'{COUNTRY_LANGUAGE[country]}_questionlist.csv')
        questions_df = pd.read_csv(questions_csv, sep=';')
        questions_df['WrappedQuestion'] = questions_df['ShortQuestion'].apply(wrap)

        # remove unused questions
        questions_df = questions_df[(questions_df['Question_ID'] != 54) & (questions_df['Group'] != 'Metadata')]

        # create country folder if necessary
        country_dir = APPENDIX_DIR.joinpath(country)
        if not os.path.exists(country_dir):
            os.mkdir(country_dir)

        # filter survey data
        print(f'Including survey data from {country}')
        if country != "ALL": 
            survey_country_df = survey_df[survey_df['ip_country'] == country].copy()
        else:
            print('No filter applied')
            survey_country_df = survey_df.copy()

        # create pdf and csv files to import in latex
        create_plots_and_tables(questions_df, survey_country_df, country)

        # create tex document for country
        create_tex(questions_df, country, country_dir)



