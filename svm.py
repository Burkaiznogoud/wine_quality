from sklearn.svm import SVC # Support Vector Machine algorithm
import numpy as np

# Support Vector Machine
class SVM_Algorithm:
    def __init__(self):
        self.parameters = { 'kernel':('linear', 'rbf','poly','rbf', 'sigmoid'),
                            'C': np.logspace(-3, 3, 5),
                            'gamma':np.logspace(-3, 3, 5)}
        self.estimator = SVC()

""" Tuned hyperparameters :(best parameters)  {'C': 0.001, 'gamma': 0.001, 'kernel': 'linear'}
Accuracy :  1.0, Score: 1.0
svc = SVM_Algorithm(X_train, X_test, Y_train, Y_test)
svc.hyperparameters_score()
"""