from prepare_file import PrepareFile, Data
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from misc.evaluation import evaluation



data = PrepareFile(file='.\datafiles\Redwine.csv', Y_column='Recommended', columns_to_exclude=['Quality', 'Ph', 'Free sulfur dioxide', 'Density', 'Fixed acidity', 'Volatile acidity'])

correlation_matrix = pd.DataFrame(data.X).corr()

plt.figure(figsize=(10, 7))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", xticklabels=data.X_columns, yticklabels=data.X_columns)
plt.show()

d = Data()

X_train, X_test, Y_train, Y_test, columns = d.data
ridge = RidgeClassifier()
ridge.fit(X_train, Y_train)
y_hat = ridge.predict(X_test)

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
    plt.show()
        
results()
plot_confusion_matrix()