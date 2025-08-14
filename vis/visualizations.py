import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize



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
    plt.savefig(f'data/outputs/{title}.png', dpi = 200)
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
                 x = 'Age at Diagnosis',
                 kde=True)
    plt.title('Distribtion of DCIS Patient Ages')
    plt.savefig('data/outputs/age_hist_plot.png', dpi = 200)
    plt.show()

def plot_tumors_by_age(merged_df):
    """Plots grouped barplot that counts tumor grade for each age group
    
    Parameter:
    - merged_df: the merged df that is the output from the etl

    Returns:
    None
    """
    gap = (max(merged_df['Age at Diagnosis']) - min(merged_df['Age at Diagnosis'])) / 4
    age_bins = list(np.arange(min(merged_df['Age at Diagnosis']), max(merged_df['Age at Diagnosis']), gap))
    age_bins.append(99)
    labels = [1,2,3,4]

    merged_df_copy = merged_df.copy()
    merged_df_copy['age_group'] = pd.cut(merged_df_copy['Age at Diagnosis'], bins=age_bins, labels=labels, right=True)
    result = merged_df_copy.groupby(['age_group', 'Tumor Grade'], observed=True).size().reset_index(name='Count')

    g = sns.barplot(data=result, x='age_group', y='Count', hue='Tumor Grade')
    g.set_xticks(range(0, 4))
    g.set_xticklabels(['[32 - 45]', '[45 - 59]', '[59 - 73]', '[73 - 99]'])
    plt.title('Number of Patients With Specific DCIS Risk Separated By Age Group')
    plt.savefig('data/outputs/tumors_by_age.png', dpi = 200)
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
                 x = 'Days to Last Known Disease Status',
                 kde=True,
                 bins=30)
    plt.title('Distribtion of Days to Last Known Disease Status')
    plt.savefig('data/outputs/days_last_follow_up_hist_plot.png', dpi = 200)
    plt.show()


def plot_roc_curve(y_pred_proba, y_test):
    """ Plots multi-class ROC curve
    
    Parameters:
    - y_pred_proba: the predicted probability of being in a certain class
    - y_test: the true labels of the test set
    
    Returns:
    None"""

    y_test_binarized=label_binarize(y_test, classes=np.unique(y_test))
    fpr = {}
    tpr = {}
    thresh ={}
    roc_auc = dict()

    classes = ['G1', 'G2', 'G3']

    for i in range(0,3):    
        fpr[i], tpr[i], thresh[i] = roc_curve(y_test_binarized[:,i], y_pred_proba[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])   
        plt.plot(fpr[i], tpr[i], linestyle='--', label='%s vs Rest (AUC=%0.2f)'%(classes[i], roc_auc[i]))

    plt.plot([0,1],[0,1],'b--')
    plt.xlim([0,1])
    plt.ylim([0,1.05])
    plt.title('Multiclass ROC Curve')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend()
    plt.savefig('data/outputs/multiclass_roc.png', dpi = 200)
    plt.show()