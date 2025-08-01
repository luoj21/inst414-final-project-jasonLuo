import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix



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


def plot_class_counts(merged_df):
    pass


def plot_ages(merged_df):
    pass


def plot_ethnicity(merged_df):
    pass


def plot_race(merged_df):
    pass