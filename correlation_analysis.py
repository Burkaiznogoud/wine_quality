import prepare_file as f
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from seaborn.objects import Plot as plot
from seaborn.objects import Dots as dot
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from misc.evaluation import evaluation


data = f.PrepareFile(file='.\datafiles\Redwine.csv', Y_column='Recommended',columns_to_exclude=['Quality'])
d = f.Data()
print(d.data)
X_train, X_test, Y_train, Y_test, columns = d.data
ridge = RidgeClassifier()
ridge.fit(X_train, Y_train)
y_hat = ridge.predict(X_test)

def plot_correlation_matrix(columns_to_exclude=['Quality']):
    data = f.PrepareFile(file='.\datafiles\Redwine.csv', Y_column='Recommended', columns_to_exclude=columns_to_exclude)
    correlation_matrix = pd.DataFrame(data.X).corr()
    plt.figure(figsize=(10, 7))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", xticklabels=data.X_columns, yticklabels=data.X_columns)
    plt.title(f"Excluded columns : {columns_to_exclude}")
    plt.savefig(fname="plots\correlation_matrix.png")
    plt.close()

def calculate_variance_inflation_factors():
    x = pd.DataFrame(X_train, columns=d.columns)
    y = pd.DataFrame(X_test, columns=d.columns)
    vif_data = pd.concat([x, y])
    vif = pd.DataFrame()
    vif["Variable"] = d.columns
    vif["VIF"] = [f"{variance_inflation_factor(vif_data, i):.4f}" for i in range(vif_data.shape[1])]
    return vif

def plot_meshgrid():
    raw_data = d.get_raw()
    features = pd.DataFrame(raw_data['X_raw'])
    pairplot = sns.pairplot(features, kind='scatter', diag_kind='kde', height=1, aspect=1)
    for axes in pairplot.axes.flat:
        axes.set_ylabel(axes.get_ylabel(), rotation=60)
        axes.set_xlabel(axes.get_xlabel(), rotation=45)
    plt.savefig(fname="plots\meshgrid_feature.png")
    plt.close()

@evaluation
def results():
    results = {
    'accuracy': f"{accuracy_score(Y_test, y_hat):.4f}",
    'precision': f"{precision_score(Y_test, y_hat):.4f}",
    'recall':  f"{recall_score(Y_test, y_hat):.4f}",
    'f1': f"{f1_score(Y_test, y_hat):.4f}",
    'classif_report':  f"{classification_report(Y_test, y_hat):.4f}"
    }
    return results
    
def plot_confusion_matrix():
    """Set option = show to plot confusion matrix. Default set to save."""
    cm = confusion_matrix(Y_test, y_hat)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, fmt='d')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['Not recommended', 'Recommended']) 
    ax.yaxis.set_ticklabels(['Not recommended', 'Recommended'])
    plt.savefig(fname='plots\cm_features.png')
    plt.close()

plot_correlation_matrix(columns_to_exclude=['Quality', 'Free sulfur dioxide', 'Density', 'Citric acid', 'Ph'])
# print(calculate_variance_inflation_factors())        
# results()
# plot_meshgrid()
# plot_confusion_matrix()