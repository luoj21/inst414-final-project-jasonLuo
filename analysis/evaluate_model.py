import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from models import DCIS_classification_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def plot_confusion_matrix(y_true, y_pred, normalize=False, title='Confusion Matrix'):
    """ Plots a confusion matrix using seaborn heatmap.

    Parameters: 
    - y_true: list or array of true labels
    - y_pred: list or array of predicted labels
    - labels: list of label names (optional)
    - normalize: if True, show percentages instead of raw counts
    - title: title of the confusion matrix plot
    
    Returns:
    None"""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    plt.show()



def evaluate_model():
    # path = 'data/transformed_data/merged_df.csv'
    # merged_df = pd.read_csv(path)
    # merged_df = merged_df.iloc[:, 1:]

    # merged_df.dropna(inplace = True)
    # print(merged_df.shape)

    X = np.random.rand(3000, 10)
    y = np.random.randint(0, 3, size=(3000, 1)).ravel()
    model = DCIS_classification_model("forest")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)
    model.fit(X_train, y_train, use_grid_search=True)
    y_pred = model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, title= f"Confusion Matrix For {model.model_type.upper()} Model", normalize=True)
    print(classification_report(y_pred=y_pred, y_true=y_test))
