
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing

from sklearn.model_selection import train_test_split # Allows us to split our data into training and testing data
from sklearn.model_selection import GridSearchCV # Allows us to test parameters of classification algorithms and find the best one
from sklearn.linear_model import LinearRegression # Linear Regression algorithm
from sklearn.svm import SVC # Support Vector Machine algorithm
from sklearn.tree import DecisionTreeClassifier # Decision Tree Classifier algorithm
from sklearn.feature_selection import RFE # Recursie Feature Elimination algorithm
from sklearn.feature_selection import SelectKBest # Select K Best algorithm