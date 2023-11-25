from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class PrepareFile:
    def __init__(self, file, Y_column, columns_to_exclude=None):
        self.exclude = columns_to_exclude
        self.options(file)
        self.prepare_X(columns_to_exclude)
        self.prepare_Y(Y_column)
        self.Y_column = Y_column
        self.X_columns = [col for col in self.file.columns if col != self.Y_column and col not in self.exclude]

    def options(self, file):
        print(20 * "-")
        print(f"Reading {file} file from CSV.")
        print(20 * "-")
        self.file = pd.read_csv(file)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_colwidth', None)
        return self.file
    
    def prepare_Y(self, Y_column):
        print(20 * "-")
        print(f"Extracting {Y_column} and converting it to numpy.")
        print(20 * "-")
        Y = self.file[Y_column]
        Y = Y.replace({True: np.float64(1), False: np.float64(0)})
        self.Y = Y.to_numpy()
        self.file = self.file.drop(Y_column, axis=1)
        self.file.set_index('Id', inplace=True)
        return self.Y

    def prepare_X(self, columns_to_exclude=None):
        if columns_to_exclude == None:
            print(20 * "-")
            print(f"No columns were excluded. Left dataframe is being converted to numpy array.")
            print(20 * "-")
            X = self.file.reset_index(drop=True)
            X.set_index(X.columns[0], inplace=True)
            self.X = X.to_numpy()
        else:
            print(20 * "-")
            print(f"Columns {columns_to_exclude} were excluded. Left dataframe is being converted to numpy array.")
            print(20 * "-")
            X = self.file.drop(columns_to_exclude, axis=1).reset_index(drop=True)
            X.set_index(X.columns[0], inplace=True)
            self.X = X.to_numpy()
        return self.X
    

class Data:
    def __init__(self):
        self.prepare_data()

    def prepare_data(self):
        data = PrepareFile(file='.\datafiles\Redwine.csv', Y_column='Recommended', columns_to_exclude=['Quality', 'Id'])
        Y = data.Y
        X = data.X
        columns = data.X_columns
        transform = StandardScaler()
        X = transform.fit_transform(data.X)  
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4)
        self.data = X_train, X_test, Y_train, Y_test, columns
        return self.data