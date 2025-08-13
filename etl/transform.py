import numpy as np
import pandas as pd

from logger_config import my_logger

def standardize_tumor_values(diagnostics_df):
    """Converts tumor grade to a standardized format
    
    Parameters:
    - diagnostics_df: diagnostics data frame
    
    Returns:
    None"""

    mapping = {"High Grade": "G3",
               "Low Grade": "G2",
               "Intermediate Grade": "G1",
               "unknown": np.nan}
    
    diagnostics_df_copy = diagnostics_df.copy()
    diagnostics_df_copy['Tumor Grade'] = diagnostics_df_copy['Tumor Grade'].map(mapping)

    return diagnostics_df_copy

def standardize_race(demographics_df):
    """Converts race to a one hot encoded variable
    
    Parameters:
    - demographics_df: demographics data frame
    
    Returns:
    None"""
    mapping = {"Alive": 1,
               "Dead": 0,
               "unkown": np.nan}
    
    demographics_df_copy = demographics_df.copy()
    demographics_df_copy['Race'] = demographics_df_copy['Race'].map(mapping)
    return demographics_df_copy


def standardize_ethnicity(demographics_df):
    """Converts ethnicity to a one hot encoded variable
    
    Parameters:
    - demographics_df: demographics data frame
    
    Returns:
    None"""
    mapping = {"hispanic or latino": 1,
               "not hispanic or latino": 0,
               "unkown": np.nan}
    
    demographics_df_copy = demographics_df.copy()
    demographics_df_copy['Ethnicity'] = demographics_df_copy['Ethnicity'].map(mapping)
    return demographics_df_copy


def tidy_molecular_test(molecular_test_df):
    """Tidy's the molecular test data frame so each row is a participant with specific gene measurements
    
    Parameters:
    - molecular_test_df: molecular test data frame
    
    Returns:
    None"""

    cols = ["HTAN Participant ID", "Timepoint Label", "Gene Symbol", "Test Result"]
    molecular_test_df = molecular_test_df.loc[:, cols]

    molecular_test_df['Test Result'].apply(lambda row: np.nan if row == "unknown" else row)

    # Step 4: Pivot into tidy format
    tidy_molecular_test_df = molecular_test_df.pivot_table(
        index=["HTAN Participant ID", "Timepoint Label"],
        columns="Gene Symbol",
        values="Test Result",
        aggfunc="first"
    ).reset_index()

    return tidy_molecular_test_df

def standardize_age_at_diagnosis(diagnostics_df):
    """Standardizes age at diagnosis by converting to float in years
    
    Parameters:
    - diagnostics_df: diagnostics data frame
    
    Returns:
    None"""

    diagnostics_df['Age at Diagnosis'] = diagnostics_df['Age at Diagnosis'].apply(lambda row: np.nan if row == 'Not Applicable' else row)
    diagnostics_df['Age at Diagnosis'] = diagnostics_df['Age at Diagnosis'].astype(np.float32) / 365.25

    return diagnostics_df

def create_age_tumor_interaction(merged_df):
    """Creates a tumor and age interaction term as a feature
    
    Parameters:
    - merged_df: the merged data frame that was outputted from the etl
    
    Returns:
    None"""


    mapping = {"G3": 1.75,
               "G2": 1.5,
               "G1": 1.25}
    numerical_grade = merged_df['Tumor Grade'].map(mapping)

    merged_df['Age_Grade_Score'] = numerical_grade * merged_df['Age at Diagnosis']
    return merged_df



def transform_data(demographics, diagnostics, molecular_test):
    """Transform and load the demographics, diagnostics, and molecular_test datasets
    into one data set for analysis
    
    Parameters:
    demographics: demographics dataset that contains demographic info of DCIS patients
    diagnostics: diagnostics dataset that contains diagnostics info of the DCIS patients
    molecular_test: molecular testing dataset containing molecular tests done on DCIS patients
    
    Returns:
    merged_df: the transformed / cleaned dataset that has info from the demographics,
    diagnostics, and molecular test data"""


    demographics = standardize_ethnicity(demographics_df=demographics)
    demographics = standardize_race(demographics_df=demographics)
    diagnostics = standardize_tumor_values(diagnostics_df=diagnostics)
    diagnostics = standardize_age_at_diagnosis(diagnostics_df=diagnostics)
    molecular_test = tidy_molecular_test(molecular_test_df=molecular_test)

    # Merging data on Participant ID
    merged_df = pd.merge(demographics, diagnostics, on='HTAN Participant ID', how='left')
    merged_df = pd.merge(merged_df, molecular_test, on='HTAN Participant ID', how='left')

    # Keeping only relavant columns
    cols = ['HTAN Participant ID','Ethnicity', 'Age at Diagnosis' ,'Year of Diagnosis', 'Tumor Grade', 
            'Days to Last Known Disease Status','Days to Recurrence', 'Days to Last Follow up','ERBB2', 'ESR1','HER2', 'PGR']
    merged_df = merged_df.loc[:, cols]

    # Convert 'Not Applicable' to NA:
    merged_df = merged_df.map(lambda x: np.nan if x == "Not Applicable" else x)

    # Remove rows where tumor grade is NA
    merged_df = merged_df.dropna(subset=['Tumor Grade'])

    # Remove rows where ethnicity is NA
    merged_df = merged_df.dropna(subset=['Ethnicity'])

    # One hot encoding the molecular test for the 4 genes
    merged_df['ERBB2'] = merged_df['ERBB2'].apply(lambda row: 1 if row == 'positive' else 0)
    merged_df['ESR1'] = merged_df['ESR1'].apply(lambda row: 1 if row == 'positive' else 0)
    merged_df['HER2'] = merged_df['HER2'].apply(lambda row: 1 if row == 'positive' else 0)
    merged_df['PGR'] = merged_df['PGR'].apply(lambda row: 1 if row == 'positive' else 0)

    # Turn NAs in Days to Recurrence to 0
    merged_df['Days to Recurrence'] = merged_df['Days to Recurrence'].fillna(0)

    my_logger.info(f'### Data has any NAs: {merged_df.isna().any().any()} ###')

    # Convert to appropriate data types
    merged_df['Days to Recurrence'] = merged_df['Days to Recurrence'].astype(int)
    merged_df['Days to Last Known Disease Status'] = merged_df['Days to Last Known Disease Status'].astype(int)
    merged_df['Days to Last Follow up'] = merged_df['Days to Last Follow up'].astype(int)

    # New feature that counts the number of positive biomarkers
    merged_df['Num_Positive_Biomarkers'] = merged_df['ERBB2'] + merged_df['ESR1'] + merged_df['HER2'] + merged_df['PGR']

    # New feature for age vs tumor grade interaction:
    merged_df = create_age_tumor_interaction(merged_df)
 
    # Export to .csv
    merged_df.to_csv('data/transformed_data/merged_df.csv', index=False)
    return merged_df


    




