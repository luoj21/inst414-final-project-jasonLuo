import pandas as pd
import numpy as np

import etl.transform
import analysis.evaluate_model
from vis.visualizations import *


def main(): 
    ## transform data:
    data = etl.transform.transform_data()

    ## create and evaluate model(s)
    analysis.evaluate_model.evaluate_model()

    ## create visualizations
    plot_class_counts(data)
    plot_ages(data)
    plot_ethnicity(data)
    plot_days_to_last_follow_up(data)
    



if __name__ == "__main__":
    main()