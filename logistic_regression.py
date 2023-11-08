from sklearn.linear_model import LogisticRegression # Linear Regression algorithm

# Logistic Regression
class LogisticRegression_Algorithm:
    def __init__(self):
        self.estimator = LogisticRegression()
        self.parameters = {"C":[0.01,0.1,1],'penalty':['l2'], 'solver':['lbfgs']}

"""
Tuned hyperparameters :(best parameters)  {'C': 0.01, 'penalty': 'l2', 'solver': 'lbfgs'}
Accuracy :  1.0
1.0
lr = LogisticRegression_Algorithm(X_train, X_test, Y_train, Y_test)
lr.hyperparameters_score()
"""
