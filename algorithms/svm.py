from sklearn.svm import SVC # Support Vector Machine algorithm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance
from sklearn.exceptions import UndefinedMetricWarning
import pandas as pd
import numpy as np
from misc.timing import timing
from misc.evaluation import evaluation
from misc.processing import processing


# Support Vector Machine
class SVM_Algorithm:
    def __init__(self, columns, X_train, Y_train, X_test, Y_test, init_params = 'default'):
        self.parameters =   { 
                            'kernel': ['linear', 'poly'],
                            'C': [0.001, 0.01, 0.1, 1, 10],
                            'gamma':[0.001, 0.01, 0.1, 1, 10],
                            'degree': [3, 4 , 5],
                            'decision_function_shape': ['ovo', 'ovr']
                            }
        self.columns = columns
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.instantiate_SVC(init_params = init_params)
        self.evaluate_classification_metrics()
        self.get_feature_importance()
        self.evaluation_results()

    @processing
    def instantiate_SVC(self, init_params):
        params = {'default':{  
                            'kernel': 'linear',
                            'C' : 1, 
                            'gamma': 0.1, 
                            'degree': 3, 
                            'decision_function_shape': 'ovo'
                            }
                  }
        if init_params == 'default':
            self.estimator = SVC(**params['default'])
            print(self.estimator)
            return self.estimator
        else:
            self.estimator = SVC(**init_params)
            print(self.estimator)
            return self.estimator

    @processing
    def evaluate_classification_metrics(self):
        self.estimator.fit(self.X_train, self.Y_train)
        Y_hat = self.estimator.predict(self.X_test)
        self.accuracy = accuracy_score(self.Y_test, Y_hat)
        self.precision = precision_score(self.Y_test, Y_hat)
        self.recall = recall_score(self.Y_test, Y_hat)
        self.f1 = f1_score(self.Y_test, Y_hat)
        return self.accuracy, self.precision, self.recall, self.f1

    @processing
    def get_feature_importance(self):
        self.estimator.fit(self.X_train, self.Y_train)
        if len(self.X_test) != len(self.Y_test):
            raise ValueError(f"X_test {self.X_test.shape} and Y_test {self.Y_test.shape} must have the same number of samples.")
        try:
            if self.estimator.kernel == 'linear':
                coefficients = self.estimator.coef_[0]
                self.selected_features = pd.DataFrame(
                    {
                    'Feature': self.columns, 
                    'Coefficient': coefficients
                    }
                )
                self.selected_features = self.selected_features.reindex(self.selected_features['Coefficient'].abs().sort_values(ascending=False).index)
                return self.selected_features
            else:
                result = permutation_importance(self.estimator, self.X_train, self.Y_train, n_repeats=30, random_state=4)
                importance_std = result.importances_std
                importances = result.importances_mean
                self.selected_features = pd.DataFrame(
                    {
                    'Feature': self.columns, 
                    'Importance': importances, 
                    "Importance_std": importance_std
                    }
                )
                self.selected_features.reindex(self.selected_features['Feature'])
                self.selected_features = self.selected_features.sort_values(
                                                                            by='Importance', 
                                                                            ascending=False
                                                                            )
                return self.selected_features
        except AttributeError as e:
            if self.estimator.kernel == 'linear' and not hasattr(self.estimator, 'probability'):
                print(f"The kernel '{self.estimator.kernel}' does not support feature coefficients.")
            elif hasattr(self.estimator, 'probability') and self.estimator.probability:
                print("Coefficients not available when using SVM with probability=True.")
            else:
                print(f"Exception: {e}")
                print(f"The kernel '{self.estimator.kernel}' does not support feature coefficients.")
        
    @evaluation
    def evaluation_results(self):
        results =   {
                    'results': f"{__name__} of {__class__}",
                    'parameters': f"{self.estimator.get_params}",
                    'accuracy': f"{self.accuracy:.4f}",
                    'precision': f"{self.precision:.4f}",
                    'recall': f"{self.recall:.4f}",
                    'f1': f"{self.f1:.4f}",
                    'features': f"{self.selected_features}"
                    }
        return results
        


""" 
A Support Vector Machine (SVM) is a powerful supervised machine learning algorithm 
that can be used for both classification and regression tasks. 
The primary goal of an SVM is to find a hyperplane that best separates different classes in the feature space. 
It does so by maximizing the margin between the classes while minimizing the classification error. 
Here's an overview of SVM, including its key concepts, parameters, and how to fit the model with the best parameters:

Key Concepts:

Hyperplane: In a binary classification problem, 
an SVM seeks to find the hyperplane that best separates 
the data points of different classes while maximizing 
the margin (distance) between the hyperplane and the nearest data points.

Support Vectors: 
Support vectors are the data points that are closest to the hyperplane and influence its position. 
These support vectors are crucial for defining the decision boundary.

Kernel Trick: 
SVMs can use a kernel function to transform the data into a higher-dimensional space. 
This allows the SVM to find non-linear decision boundaries in the original feature space.

Parameters:
Some of the key parameters for tuning an SVM model include:

Kernel: The choice of kernel function, such as linear, polynomial, radial basis function (RBF), or custom kernels.

C (Regularization parameter): 
The regularization parameter controls the trade-off between maximizing the margin and minimizing the classification error. 
A smaller value results in a larger margin but may allow some misclassification. 
A larger value makes the model try to classify all training points correctly.

Gamma (for RBF kernel): 
The gamma parameter influences the shape of the decision boundary when using the RBF kernel. 
Smaller values make the decision boundary smoother, while larger values result in a more complex boundary.

Degree (for polynomial kernel): 
The degree parameter determines the degree of the polynomial kernel function. 
Higher degrees can capture more complex relationships but may lead to overfitting.

Decision Function Shape: 
Some SVM implementations allow you to control the shape of the decision function, 
such as 'ovo' (one-vs-one) or 'ovr' (one-vs-rest) for multiclass classification.

Fitting the Model with Best Parameters:

To fit an SVM model with the best parameters, you typically follow these steps:

Data Preparation: 
Load and preprocess your data, splitting it into training and testing sets.

Hyperparameter Tuning: 
Use techniques like grid search or randomized search to find 
the best combination of hyperparameters for your SVM. 
This involves trying different combinations of kernel, C, gamma, and other relevant parameters.

Model Training: 
Fit the SVM model using the best hyperparameters on the training data. 
You can use libraries like scikit-learn to do this.

Model Evaluation: 
Evaluate the model's performance on the testing data 
using appropriate metrics such as accuracy, precision, recall, F1-score, or others, depending on your task.

Prediction: 
Once the model is trained and validated, you can use it to make predictions on new, unseen data.
"""