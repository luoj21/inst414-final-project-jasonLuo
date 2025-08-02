import numpy as np
import pandas as pd


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




def transform_data():

    demographics = pd.read_csv('data/raw_data/demographics.csv')
    diagnostics = pd.read_csv('data/raw_data/diagnostics.csv')
    molecular_test = pd.read_csv('data/raw_data/molecular_test.csv')

    demographics = standardize_ethnicity(demographics_df=demographics)
    demographics = standardize_race(demographics_df=demographics)
    diagnostics = standardize_tumor_values(diagnostics_df=diagnostics)
    diagnostics = standardize_age_at_diagnosis(diagnostics_df=diagnostics)
    molecular_test = tidy_molecular_test(molecular_test_df=molecular_test)

    # Merging data on Participant ID
    merged_df = pd.merge(demographics, diagnostics, on='HTAN Participant ID', how='left')
    merged_df = pd.merge(merged_df, molecular_test, on='HTAN Participant ID', how='left')

    # Keeping only relavant columns
    cols = ['HTAN Participant ID','Ethnicity', 'Race', 'Age at Diagnosis' ,'Year of Diagnosis', 'Tumor Grade',
            'Days to Last Follow up', 'Days to Last Known Disease Status','Days to Recurrence', 'ERBB2', 'ESR1','HER2', 'PGR' ]
    merged_df = merged_df.loc[:, cols]

    # Convert 'Not Applicable' to NA:
    merged_df = merged_df.map(lambda x: np.nan if x == "Not Applicable" else x)

    # Export to .csv
    merged_df.to_csv('data/transformed_data/merged_df.csv', index=False)
    return merged_df
    


if __name__ == "__main__":
    transform_data()

    




