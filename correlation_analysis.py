import prepare_file as f
import numpy as np
import pandas as pd
import seaborn as sns
from seaborn.objects import Plot as plot
from seaborn.objects import Dots as dot
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from misc.evaluation import evaluation

data = f.PrepareFile(file='.\datafiles\Redwine.csv', Y_column='Recommended', columns_to_exclude=['Quality', 'Ph', 'Free sulfur dioxide', 'Density', 'Fixed acidity', 'Volatile acidity'])

correlation_matrix = pd.DataFrame(data.X).corr()

# plt.figure(figsize=(10, 7))
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", xticklabels=data.X_columns, yticklabels=data.X_columns)
# plt.show()

d = f.Data()

X_train, X_test, Y_train, Y_test, columns = d.data
ridge = RidgeClassifier()
ridge.fit(X_train, Y_train)
y_hat = ridge.predict(X_test)

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
    'accuracy': accuracy_score(Y_test, y_hat),
    'precision': precision_score(Y_test, y_hat),
    'recall':  recall_score(Y_test, y_hat),
    'f1': f1_score(Y_test, y_hat),
    'classif_report':  classification_report(Y_test, y_hat)
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
        
results()
plot_meshgrid()
plot_confusion_matrix()