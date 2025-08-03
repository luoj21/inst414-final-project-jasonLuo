import pandas as pd



def extract_data():
    """Extracts datasets fron HTAN and stores to raw_data folder
    
    Parameters:
    None
    
    Returns:
    None
    """

    # Extract from HTAN
    demographics = pd.read_csv('https://d13ch66cwesneh.cloudfront.net/metadata/syn39263164.csv')
    diagnostics = pd.read_csv('https://d13ch66cwesneh.cloudfront.net/metadata/syn39263335.csv')
    molecular_test = pd.read_csv('https://d13ch66cwesneh.cloudfront.net/metadata/syn39266384.csv')

    print('### Extracting datasets... ### \n')
    demographics.to_csv('data/raw_data/demographics.csv', index=False)
    diagnostics.to_csv('data/raw_data/diagnostics.csv', index = False)
    molecular_test.to_csv('data/raw_data/molecular_test.csv', index = False)
    print('### Done extracting ###')

    return demographics, diagnostics, molecular_test