import pandas as pd
import numpy as np

import etl.transform
import analysis.evaluate_model
from vis.visualizations import *


def main():
    
    ## transform data:
    etl.transform.transform_data()

    ## create and evaluate model(s)
    analysis.evaluate_model.evaluate_model()

    ## create visualizations



if __name__ == "__main__":
    main()