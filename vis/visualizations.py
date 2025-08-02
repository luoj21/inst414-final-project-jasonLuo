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
    plt.savefig(f'data/outputs{title}.png', dpi = 200)
    plt.show()


def plot_class_counts(merged_df):
    """Plots number of patients per DCIS risk on bar plot
    
    Parameter:
    - merged_df: the merged df that is the output from the etl

    Returns:
    None
    """
    class_counts = merged_df.groupby('Tumor Grade').size().reset_index(name = 'counts')

    sns.barplot(data = class_counts,
                x = 'Tumor Grade',
                y = 'counts')
    plt.title('Number of Patients With Specific DCIS Risk')
    plt.savefig('data/outputs/class_counts_plot.png', dpi = 200)
    plt.show()



def plot_ages(merged_df):
    """Plots histogram of ages for DCIS patients
    
    Parameter:
    - merged_df: the merged df that is the output from the etl

    Returns:
    None
    """
    sns.histplot(data = merged_df,
                 x = 'Age at Diagnosis')
    plt.title('Distribtion of DCIS Patient Ages')
    plt.savefig('data/outputs/age_hist_plot.png', dpi = 200)
    plt.show()


def plot_ethnicity(merged_df):
    """Plots number of patients per ethnicity on bar plot
    
    Parameter:
    - merged_df: the merged df that is the output from the etl

    Returns:
    None
    """
    class_counts = merged_df.groupby('Ethnicity').size().reset_index(name = 'counts')

    sns.barplot(data = class_counts,
                x = 'Ethnicity',
                y = 'counts')
    plt.title('Number of Patients With Specific Ethnicity')
    plt.savefig('data/outputs/ethnicity_counts_plot.png', dpi = 200)
    plt.show()


def plot_days_to_last_follow_up(merged_df):
    """Plots histogram of ages for DCIS patients
    
    Parameter:
    - merged_df: the merged df that is the output from the etl

    Returns:
    None
    """
    sns.histplot(data = merged_df,
                 x = 'Days to Last Follow up',
                 kde=True,
                 bins=30)
    plt.title('Distribtion of Days to Last Follow Up Across Patients')
    plt.savefig('data/outputs/days_last_follow_up_hist_plot.png', dpi = 200)
    plt.show()


# def plot_race(merged_df):
#     """Plots number of patients per race on bar plot
    
#     Parameter:
#     - merged_df: the merged df that is the output from the etl

#     Returns:
#     None
#     """
#     class_counts = merged_df.groupby('Race').size().reset_index(name = 'counts')

#     sns.barplot(data = class_counts,
#                 x = 'Race',
#                 y = 'counts')
#     plt.title('Number of Patients With Specific Race')
#     plt.savefig('data/outputs/race_counts_plot.png', dpi = 200)
#     plt.show()