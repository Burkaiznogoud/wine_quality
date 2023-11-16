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
data = prep.PrepareFile(file='datafiles\RedWine.csv', Y_column='Recommended', columns_to_exclude=['Quality', 'Id', 'Alcohol'])

Y = data.Y
# Normalize data using StandardScaler
X = transform.fit_transform(data.X)

# train, test, split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4)


### Select K Best ### TESTED and WORKS!
# select_k_best = skb.SKB_Algorithm(  columns = data.X_columns,
#                                     k = 3,
#                                     X_train = X_train, Y_train = Y_train,
#                                     X_test = X_test, Y_test = Y_test
#                                     )
# select_k_best.evaluation_results()

# algorithm = a.Algorithm(X_train, X_test, Y_train, Y_test,
#                         params = select_k_best.parameters,
#                         estimator = select_k_best.estimator
#                         )
# cv_select_k_best = skb.SKB_Algorithm(columns = data.X_columns, 
#                                      k = 3, 
#                                      init_params = algorithm.best_params, 
#                                      X_train = X_train, Y_train = Y_train, 
#                                      X_test = X_test, Y_test = Y_test
#                                      )
# cv_select_k_best.evaluation_results()

### Recursive Feature Elimination ### TESTED and WORKS but further ivestigation is needed.
# recursive_selection = rfe.RFE_Algorithm(columns= data.X_columns,
#                                         X_train = X_train, Y_train = Y_train,
#                                         X_test = X_test, Y_test = Y_test
#                                         )
# recursive_selection.evaluation_results()
# algorithm = a.Algorithm(X_train, X_test, Y_train, Y_test,
#                         estimator = recursive_selection.estimator,
#                         params = recursive_selection.parameters
#                         )
# cv_recursive_selection = rfe.RFE_Algorithm( columns= data.X_columns,
#                                             init_params = algorithm.best_params,
#                                             X_train = X_train, Y_train = Y_train,
#                                             X_test = X_test, Y_test = Y_test
#                                             )
# cv_recursive_selection.evaluation_results()

### Logistic Regression ### TESTED and WORKS but further investigation is needed.
# logistic = lr.LogisticRegression_Algorithm( columns= data.X_columns,
#                                             X_train = X_train, Y_train = Y_train,
#                                             X_test = X_test, Y_test = Y_test
#                                             )
# algorithm = a.Algorithm(X_train, X_test, Y_train, Y_test,
#                         estimator = logistic.estimator,
#                         params = logistic.parameters
#                         )
# cv_logistic = lr.LogisticRegression_Algorithm(  columns= data.X_columns,
#                                                 X_train = X_train, Y_train = Y_train,
#                                                 X_test = X_test, Y_test = Y_test
#                                                 )

### Decision Tree Classifier ### TESTED and WORKS. Not sure about results.

# decision_tree = dt.DecisionTree_Algorithm(  columns= data.X_columns,
#                                             X_train = X_train, Y_train = Y_train,
#                                             X_test = X_test, Y_test = Y_test
#                                             )
# algorithm = a.Algorithm(X_train, X_test, Y_train, Y_test,
#                         estimator = decision_tree.estimator,
#                         params = decision_tree.parameters
#                         )
# cv_decision_tree = dt.DecisionTree_Algorithm(   columns= data.X_columns,
#                                                 X_train = X_train, Y_train = Y_train,
#                                                 X_test = X_test, Y_test = Y_test
#                                                 )

# svm.model_info()
# algorithm.calculate_hyperparameters()
# algorithm.hyperparameters_info()
# algorithm.plot_confusion_matrix()

# print(d_tree.get_feature_importance(X, Y, columns=data.X_columns))
# algorithm.hyperparameters_score()
# algorithm.plot_confusion_matrix()


