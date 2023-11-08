
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # Allows us to split our data into training and testing data
from sklearn.model_selection import GridSearchCV # Allows us to test parameters of classification algorithms and find the best one
from sklearn.linear_model import LogisticRegression # Linear Regression algorithm
from sklearn.svm import SVC # Support Vector Machine algorithm
from sklearn.tree import DecisionTreeClassifier # Decision Tree Classifier algorithm
from sklearn.feature_selection import RFE # Recursie Feature Elimination algorithm
from sklearn.feature_selection import SelectKBest, f_classif # Select K Best algorithm
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score


class PrepareFile:
    def __init__(self, file, column) -> None:
        self.file = self.options(file)
        self.X = self.prepare_X()
        self.Y = self.prepare_Y(column=column)

    def options(self, file):
        file = pd.read_csv(file)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_colwidth', None)
        return file
    
    def prepare_Y(self, column):
        Y = self.file[column]
        Y = Y.replace({True: np.float64(1), False: np.float64(0)})
        Y = Y.to_numpy()
        self.file = self.file.drop(column, axis=1)
        self.file.set_index('Id', inplace=True)
        return Y

    def prepare_X(self):
        X = self.file.to_numpy()
        return X


class Algorithm:
    def __init__(self, _X_train, _X_test, _Y_train, _Y_test, estimator, params):
        self.parameters = params
        self.X = _X_train
        self.Y = _Y_train
        self.x = _X_test
        self.y = _Y_test
        self.estimator = estimator
        self.cv = GridSearchCV(cv = 10, param_grid = self.parameters, estimator = self.estimator)
        self.fit, self.score, self.Y_hat = self.fit_score_predict()

    def fit_score_predict(self):
        self.fit = self.cv.fit(self.X, self.Y)
        self.score = self.cv.score(self.x, self.y)
        self.Y_hat = self.cv.predict(self.x)
        return self.fit, self.score, self.Y_hat

    def hyperparameters_score(self):
        print("Tuned hyperparameters :(best parameters) ", self.cv.best_params_)
        print("Accuracy : ", self.cv.best_score_)
        print(self.score)

    def plot_confusion_matrix(self):
        "this function plots the confusion matrix"
        cm = confusion_matrix(self.y, self.Y_hat)
        ax= plt.subplot()
        sns.heatmap(cm, annot=True, ax = ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix'); 
        ax.xaxis.set_ticklabels(['Not recommended', 'Recommended']); ax.yaxis.set_ticklabels(['Not recommended', 'Recommended'])
        plt.show()
        

# Support Vector Machine
class SVM_Algorithm:
    def __init__(self):
        self.parameters = { 'kernel':('linear', 'rbf','poly','rbf', 'sigmoid'),
                            'C': np.logspace(-3, 3, 5),
                            'gamma':np.logspace(-3, 3, 5)}
        self.estimator = SVC()
    

# Select K Best
class SKB_Algorithm:
    def __init__(self):
        self.parameters = {'k': [3, 5, 7]}
        self.scorers = {'precision_score': make_scorer(precision_score),
                        'recall_score': make_scorer(recall_score),
                        'accuracy_score': make_scorer(accuracy_score)
                        }
        self.estimator = SelectKBest(score_func=f_classif)

# Logistic Regression
class LogisticRegression_Algorithm:
    def __init__(self):
        self.estimator = LogisticRegression()
        self.parameters = {"C":[0.01,0.1,1],'penalty':['l2'], 'solver':['lbfgs']}



# Recursive Feature Elimination
class RFE_Algorithm:
    def __init__(self, estimator):
        self.parameters = { 'n_features_to_select': [3, 6, 9], 'step': [1, 3]}
        self.estimator = RFE(estimator = estimator)


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

        
# Initialization of StandardScaler object
transform = StandardScaler()    
data = PrepareFile(file='RedWine.csv', column='Recommended')

Y = data.Y
# Normalize data using StandardScaler
X = transform.fit_transform(data.X)

# train, test, split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4)


""" Tuned hyperparameters :(best parameters)  {'C': 0.001, 'gamma': 0.001, 'kernel': 'linear'}
Accuracy :  1.0, Score: 1.0
svc = SVM_Algorithm(X_train, X_test, Y_train, Y_test)
svc.hyperparameters_score()
"""

"""ValueError: Classification metrics can't handle a mix of binary and continuous targets !
# lr = LinearRegression_Algorithm(X_train, X_test, Y_train, Y_test)
# lr.hyperparameters_score()
# lr.y_hat()
"""

"""Tuned hyperparameters :(best parameters)  {'criterion': 'gini', 'max_depth': 6, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 5, 'splitter': 'best'}
Accuracy :  0.9989010989010989
Score: 1.0
tree = DecisionTree_Algorithm(X_train, X_test, Y_train, Y_test)
tree.hyperparameters_score()
"""

"""
Tuned hyperparameters :(best parameters)  {'k': 3}
Accuracy :  nan
skb = SKB_Algorithm(X_train, X_test, Y_train, Y_test)
skb.hyperparameters_score()
"""


"""
Tuned hyperparameters :(best parameters)  {'C': 0.01, 'penalty': 'l2', 'solver': 'lbfgs'}
Accuracy :  1.0
1.0
lr = LogisticRegression_Algorithm(X_train, X_test, Y_train, Y_test)
lr.hyperparameters_score()
"""

"""
rfe = Algorithm(X_train, X_test, Y_train, Y_test, estimator = lr.estimator)
rfe.hyperparameters_score()
rfe.plot_confusion_matrix()
Tuned hyperparameters :(best parameters)  {'n_features_to_select': 3, 'step': 1}
Accuracy :  1.0
1.0
"""

lr = LogisticRegression_Algorithm()
rfe = RFE_Algorithm(estimator=lr.estimator)
algorithm = Algorithm(X_train, X_test, Y_train, Y_test, estimator = rfe.estimator, params = rfe.parameters)
algorithm.hyperparameters_score()
algorithm.plot_confusion_matrix()