from sklearn.metrics import accuracy_score, classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from misc.processing import processing
from misc.evaluation import evaluation

class Algorithm:
    def __init__(self, columns, X_train, Y_train, X_test, Y_test):
        self.columns = columns
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

    @processing
    def instantiate_estimator(self):
        pass

    @processing
    def calculate_Y_hat(self):
        self.estimator.fit(self.X_train, self.Y_train)
        self.Y_hat = self.estimator.predict(self.X_test)
        return self.Y_hat
    
    @processing
    def train_classification_metrics(self):
        self.estimator.fit(self.X_train, self.Y_train)
        Y_hat = self.estimator.predict(self.X_train)
        accuracy = accuracy_score(self.Y_train, Y_hat)
        precision = precision_score(self.Y_train, Y_hat)
        recall = recall_score(self.Y_train, Y_hat)
        f1 = f1_score(self.Y_train, Y_hat)
        classif_report = classification_report(self.Y_train, Y_hat)
        self.train_report = {
            'test accuracy': f"{accuracy:.4f}",
            'test precision': f"{precision:.4f}",
            'test recall': f"{recall:.4f}",
            'test f1': f"{f1:.4f}",
            'test classification report': f"{classif_report}",
        }
        return self.train_report
    
    @processing
    def evaluate_classification_metrics(self):
        self.accuracy = accuracy_score(self.Y_test, self.Y_hat)
        self.precision = precision_score(self.Y_test, self.Y_hat)
        self.recall = recall_score(self.Y_test, self.Y_hat)
        self.f1 = f1_score(self.Y_test, self.Y_hat)
        self.classification_report = classification_report(self.Y_test, self.Y_hat)
        return self.accuracy, self.precision, self.recall, self.f1, self.classification_report
    
    @evaluation
    def evaluation_results(self):
        results =   {
                    'results': f"{__name__} of {__class__}",
                    'parameters': f"{self.estimator.get_params}",
                    'accuracy': f"{self.accuracy:.4f}",
                    'precision': f"{self.precision:.4f}",
                    'recall': f"{self.recall:.4f}",
                    'f1': f"{self.f1:.4f}",
                    'classification report': f"{self.classification_report}",
                    }
        return results
    
    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.Y_test, self.Y_hat)
        ax= plt.subplot()
        sns.heatmap(cm, annot=True, ax = ax, fmt='d')
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix'); 
        ax.xaxis.set_ticklabels(['Not recommended', 'Recommended']) 
        ax.yaxis.set_ticklabels(['Not recommended', 'Recommended'])
        plt.show()