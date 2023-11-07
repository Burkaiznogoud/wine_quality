
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # Allows us to split our data into training and testing data
from sklearn.model_selection import GridSearchCV # Allows us to test parameters of classification algorithms and find the best one
from sklearn.linear_model import LinearRegression # Linear Regression algorithm
from sklearn.svm import SVC # Support Vector Machine algorithm
from sklearn.tree import DecisionTreeClassifier # Decision Tree Classifier algorithm
from sklearn.feature_selection import RFE # Recursie Feature Elimination algorithm
from sklearn.feature_selection import SelectKBest, f_classif # Select K Best algorithm
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score

def plot_confusion_matrix(_Y, _Y_hat):
    "this function plots the confusion matrix"
    cm = confusion_matrix(_Y, _Y_hat)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['Not recommended', 'Recommended']); ax.yaxis.set_ticklabels(['Not recommended', 'Recommended'])
    plt.show()

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


# LinearRegression
class LinearRegression_Algorithm:
    def __init__(self, _X_train, _X_test, _Y_train, _Y_test):
        self.parameters = { 'fit_intercept': [True],
                            'copy_X': [True]}
        self.estimator = LinearRegression()
        self.lr_cv = GridSearchCV(cv = 10, param_grid=self.parameters, estimator=self.estimator)
        self.lr_cv.fit(_X_train, _Y_train)
        self.score = self.lr_cv.score(_X_test, _Y_test)
        self.Y_hat = self.lr_cv.predict(_X_test)

    def hyperparameters_score(self):
        print("Tuned hyperparameters :(best parameters) ", self.lr_cv.best_params_)
        print("Accuracy : ", self.lr_cv.best_score_)
        print(self.score)


# Support Vector Machine
class SVM_Algorithm:
    def __init__(self, _X_train, _X_test, _Y_train, _Y_test):
        self.parameters = { 'kernel':('linear', 'rbf','poly','rbf', 'sigmoid'),
                            'C': np.logspace(-3, 3, 5),
                            'gamma':np.logspace(-3, 3, 5)}
        self.estimator = SVC()
        self.svc_cv = GridSearchCV(cv = 10, param_grid=self.parameters, estimator=self.estimator)
        self.svc_cv.fit(_X_train, _Y_train)
        self.score = self.svc_cv.score(_X_test, _Y_test)
        self.Y_hat = self.svc_cv.predict(_X_test)

    def hyperparameters_score(self):
        print("Tuned hyperparameters :(best parameters) ", self.svc_cv.best_params_)
        print("Accuracy : ", self.svc_cv.best_score_)
        print(self.score)
    

# Select K Best
class SKB_Algorithm:
    def __init__(self, _X_train, _X_test, _Y_train, _Y_test):
        self.parameters = {'k': [3, 5, 7]}
        self.scorers = {'precision_score': make_scorer(precision_score),
                        'recall_score': make_scorer(recall_score),
                        'accuracy_score': make_scorer(accuracy_score)
                        }
        self.estimator = SelectKBest(score_func=f_classif)
        self.skb_cv = GridSearchCV(cv = 10, param_grid=self.parameters, scoring=self.scorers, estimator=self.estimator, refit='accuracy_score')
        self.skb_cv.fit(_X_train, _Y_train)

    def hyperparameters_score(self):
        print("Tuned hyperparameters :(best parameters) ", self.skb_cv.best_params_)
        print("Accuracy : ", self.skb_cv.best_score_)


# Recursive Feature Elimination
class RFE_Algorithm:
    def __init__(self, _X_train, _X_test, _Y_train, _Y_test):
        self.parameters = { 'n_features_to_select': [3, 6, 9],
                            'step': [1, 3]}
        self.estimator = RFE()
        self.rfe_cv = GridSearchCV(cv = 10, param_grid=self.parameters, estimator=self.estimator)
        self.rfe_cv.fit(_X_train, _Y_train)
        self.score = self.rfe_cv.score(_X_test, _Y_test)
        self.Y_hat = self.rfe_cv.predict(_X_test)

    def hyperparameters_score(self):
        print("Tuned hyperparameters :(best parameters) ", self.rfe_cv.best_params_)
        print("Accuracy : ", self.rfe_cv.best_score_)
        print(self.score)


# Decision Tree Classifier
class DecisionTree_Algorithm:
    def __init__(self, _X_train, _X_test, _Y_train, _Y_test):
        self.parameters = { 'criterion': ['gini', 'entropy'],
                            'splitter': ['best', 'random'],
                            'max_depth': [2, 4, 6],
                            'max_features': ['auto', 'sqrt', 'log2'],
                            'min_samples_leaf': [1, 2, 4],
                            'min_samples_split': [2, 5, 10]}
        self.estimator = DecisionTreeClassifier()
        self.tree_cv = GridSearchCV(cv = 10, param_grid=self.parameters, estimator=self.estimator)
        self.tree_cv.fit(_X_train, _Y_train)
        self.score = self.tree_cv.score(_X_test, _Y_test)
        self.Y_hat = self.tree_cv.predict(_X_test)

    def hyperparameters_score(self):
        print("Tuned hyperparameters :(best parameters) ", self.tree_cv.best_params_)
        print("Accuracy : ", self.tree_cv.best_score_)
        print(self.score)
        
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
plot_confusion_matrix(Y_test, svc.Y_hat)
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
plot_confusion_matrix(Y_test, tree.Y_hat)
"""

"""
Tuned hyperparameters :(best parameters)  {'k': 3}
Accuracy :  nan
skb = SKB_Algorithm(X_train, X_test, Y_train, Y_test)
skb.hyperparameters_score()
"""

rfe = RFE_Algorithm(X_train, X_test, Y_train, Y_test)
rfe.hyperparameters_score()
plot_confusion_matrix(Y_test, rfe.Y_hat)