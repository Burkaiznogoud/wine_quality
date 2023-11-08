import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # Allows us to split our data into training and testing data
from sklearn.preprocessing import StandardScaler
import algorithm as a
import select_K_Best as skb
import svm
import recursive_feature_elimination as rfe
import logistic_regression as lr
import decision_tree as dt
import prepare_file as prep

   
# Initialization of StandardScaler object
transform = StandardScaler()    
data = prep.PrepareFile(file='RedWine.csv', column='Recommended')

Y = data.Y
# Normalize data using StandardScaler
X = transform.fit_transform(data.X)

# train, test, split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4)


lr = lr.LogisticRegression_Algorithm()
rfe = rfe.RFE_Algorithm(estimator=lr.estimator)
algorithm = a.Algorithm(X_train, X_test, Y_train, Y_test, estimator = rfe.estimator, params = rfe.parameters)
algorithm.hyperparameters_score()
algorithm.plot_confusion_matrix()