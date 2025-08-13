import pandas as pd
import numpy as np

from analysis.evaluate_model import *
from vis.visualizations import *
from etl.extract import *
from etl.transform import *

def main(): 
    ## transform data:
    demographics, diagnostics, molecular_test = extract_data()
    data = transform_data(demographics, diagnostics, molecular_test)

    ## create and evaluate model(s)
    evaluate_model(data)

    ## create visualizations
    plot_class_counts(data)
    plot_ages(data)
    plot_ethnicity(data)
    plot_days_to_last_follow_up(data)


if __name__ == "__main__":
    main()