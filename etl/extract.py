import pandas as pd
from logger_config import my_logger



def extract_data():
    """Extracts datasets fron HTAN and stores to raw_data folder
    
    Parameters:
    None
    
    Returns:
    None
    """

    # Extract from HTAN
    try:
        demographics = pd.read_csv('https://d13ch66cwesneh.cloudfront.net/metadata/syn39263164.csv')
        diagnostics = pd.read_csv('https://d13ch66cwesneh.cloudfront.net/metadata/syn39263335.csv')
        molecular_test = pd.read_csv('https://d13ch66cwesneh.cloudfront.net/metadata/syn39266384.csv')

        my_logger.info('### Extracting datasets... ### \n')
        demographics.to_csv('data/raw_data/demographics.csv', index=False)
        diagnostics.to_csv('data/raw_data/diagnostics.csv', index = False)
        molecular_test.to_csv('data/raw_data/molecular_test.csv', index = False)
        my_logger.info('### Done extracting ###')
    
    except:
        my_logger.info("Invalid link(s)")
        raise IOError()

    return demographics, diagnostics, molecular_test