import numpy as np
import matplotlib.pyplot as plt

from analysis.models import DCIS_classification_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from vis.visualizations import plot_confusion_matrix
from logger_config import my_logger

def create_numerical_classes(y):
    """Converts tumor grade to an integer between 0 and 2
    
    Parameters:
    - y: the labels that are in string form, ie, G1, G2, G3
    
    Returns:
    - numerical_classes: list of integers between 0 and 2, representing class lables"""

    mapping = {"G1": 0,
               "G2": 1,
               "G3": 2}
    
    numerical_classes = y.copy()
    numerical_classes = y.map(mapping)

    return numerical_classes



def evaluate_model(merged_df):
    """Trains and evaluates a model and exports a confusion matrix with results
    
    Parameters:
    - merged_df: the merged data frame that was outputted from the etl
    
    Returns:
    None"""

    X = merged_df.loc[:, ~merged_df.columns.isin(['HTAN Participant ID', 'Tumor Grade'])]
    y = merged_df['Tumor Grade']
    y = create_numerical_classes(y)

    model = DCIS_classification_model("forest")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)
    model.fit(X_train, y_train, use_grid_search=False)
    y_pred = model.predict(X_test)

    my_logger.info(f"Displaying classification report for {model.model_type.upper()}: \n")
    print(classification_report(y_pred=y_pred, y_true=y_test))
    my_logger.info(f"The overall accuracy of {model.model_type.upper()} is {accuracy_score(y_pred=y_pred,y_true=y_test)}")

    plot_confusion_matrix(y_pred=y_pred, y_true=y_test, title = f'Confusion Matrix For {model.model_type.upper()}')

