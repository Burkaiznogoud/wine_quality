from sklearn.tree import DecisionTreeClassifier # Decision Tree Classifier algorithm

# Decision Tree Classifier
class DecisionTree_Algorithm:
    def __init__(self):
        self.parameters = { 'criterion': ['gini', 'entropy'],
                            'splitter': ['best', 'random'],
                            'max_depth': [2, 4, 6],
                            'max_features': ['auto', 'sqrt', 'log2'],
                            'min_samples_leaf': [1, 2, 4],
                            'min_samples_split': [2, 5, 10]}
        self.estimator = DecisionTreeClassifier()

"""Tuned hyperparameters :(best parameters)  {'criterion': 'gini', 'max_depth': 6, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 5, 'splitter': 'best'}
Accuracy :  0.9989010989010989
Score: 1.0
tree = DecisionTree_Algorithm(X_train, X_test, Y_train, Y_test)
tree.hyperparameters_score()
""" 