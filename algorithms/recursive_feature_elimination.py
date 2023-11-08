from sklearn.feature_selection import RFE # Recursie Feature Elimination algorithm

# Recursive Feature Elimination
class RFE_Algorithm:
    def __init__(self, estimator):
        self.parameters = { 'n_features_to_select': [3, 6, 9], 'step': [1, 3]}
        self.estimator = RFE(estimator = estimator)


"""
rfe = Algorithm(X_train, X_test, Y_train, Y_test, estimator = lr.estimator)
rfe.hyperparameters_score()
rfe.plot_confusion_matrix()
Tuned hyperparameters :(best parameters)  {'n_features_to_select': 3, 'step': 1}
Accuracy :  1.0
1.0
"""