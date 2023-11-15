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

select_k_best = skb.SKB_Algorithm(columns = data.X_columns, k = 4,  X_train = X_train, Y_train = Y_train, X_test = X_test, Y_test = Y_test)
select_k_best.evaluation_results()

# select_k_best.score_f_classif()
# print(select_k_best.f_classif, select_k_best.estimator, select_k_best.parameters)
# algorithm = a.Algorithm(X_train, X_test, Y_train, Y_test, params = select_k_best.parameters, estimator = select_k_best.estimator)
# algorithm.calculate_hyperparameters()


# select_k_best.score_mutual_classif()
# algorithm = a.Algorithm(X_train, X_test, Y_train, Y_test, params = select_k_best.parameters, estimator = select_k_best.estimator)
# algorithm.calculate_hyperparameters()


# select_k_best.score_mutual_regression()
# algorithm = a.Algorithm(X_train, X_test, Y_train, Y_test, params = select_k_best.parameters, estimator = select_k_best.estimator)
# algorithm.calculate_hyperparameters()

# print(f"f_classif : {select_k_best.f_classif}\n mutual_classif : {select_k_best.mutual_classification} \n mutual_regression : {select_k_best.mutual_regression}")
# lr = lr.LogisticRegression_Algorithm()
# rfe = rfe.RFE_Algorithm(estimator=lr.estimator)
# d_tree = dt.DecisionTree_Algorithm()

# svm = svm.SVM_Algorithm(columns= data.X_columns, X_train = X_train, Y_train = Y_train, X_test = X_test, Y_test = Y_test)
# algorithm = a.Algorithm(X_train, X_test, Y_train, Y_test, estimator = svm.estimator, params = svm.parameters)
# svm.model_info()
# algorithm.calculate_hyperparameters()
# algorithm.hyperparameters_info()
# algorithm.plot_confusion_matrix()


# print(rfe.get_feature_ranking(X, Y, columns=data.X_columns))
# print(rfe.selected_features)
# print(rfe.estimator.ranking_)
# print(d_tree.get_feature_importance(X, Y, columns=data.X_columns))
# algorithm.hyperparameters_score()
# algorithm.plot_confusion_matrix()


