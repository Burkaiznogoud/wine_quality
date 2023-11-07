
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # Allows us to split our data into training and testing data
from sklearn.model_selection import GridSearchCV # Allows us to test parameters of classification algorithms and find the best one
from sklearn.linear_model import LinearRegression # Linear Regression algorithm
from sklearn.svm import SVC # Support Vector Machine algorithm
from sklearn.tree import DecisionTreeClassifier # Decision Tree Classifier algorithm
from sklearn.feature_selection import RFE # Recursie Feature Elimination algorithm
from sklearn.feature_selection import SelectKBest # Select K Best algorithm

filename = 'RedWine.csv'

df = pd.read_csv(filename)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


df.set_index('Id', inplace=True)
print(df.head())

# Normalize data using StandardScaler
# extracted Recommended column as Y
Y = df['Recommended']
Y = Y.replace({True: 1, False: 0})
df = df.drop('Recommended', axis=1)

# Initialization of StandardScaler object
transform = StandardScaler()

# Data to normalize and use for models creation
X = df.to_numpy()
x = transform.fit_transform(X)

print("Y: ", Y)
print("X: ", X)