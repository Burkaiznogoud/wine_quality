import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # Allows us to split our data into training and testing data
from sklearn.preprocessing import StandardScaler
import algorithm as a
from algorithms import select_K_Best as skb
from algorithms import svm
from algorithms import recursive_feature_elimination as rfe
from algorithms import logistic_regression as lr
from algorithms import decision_tree as dt
import prepare_file as prep

   
# Initialization of StandardScaler object
transform = StandardScaler()    
data = prep.PrepareFile(file='datafiles\RedWine.csv', Y_column='Recommended', columns_to_exclude=['Quality', 'Id'])

Y = data.Y
# Normalize data using StandardScaler
X = transform.fit_transform(data.X)

# train, test, split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

print(data.X_columns)
# lr = lr.LogisticRegression_Algorithm()
# rfe = rfe.RFE_Algorithm(estimator=lr.estimator)
d_tree = dt.DecisionTree_Algorithm()
algorithm = a.Algorithm(X_train, X_test, Y_train, Y_test, estimator = d_tree.estimator, params = d_tree.parameters)
# print(rfe.get_feature_ranking(X, Y, columns=data.X_columns))
# print(rfe.selected_features)
# print(rfe.estimator.ranking_)
print(d_tree.get_feature_importance(X, Y, columns=data.X_columns))
algorithm.hyperparameters_score()
algorithm.plot_confusion_matrix()