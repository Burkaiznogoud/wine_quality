from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

_ = 20 * '-'

class PrepareFile:
    def __init__(self, file, Y_column, columns_to_exclude=None):
        self.exclude = columns_to_exclude
        self.options(file)
        self.prepare_Y(Y_column)
        self.prepare_X(columns_to_exclude)
        self.Y_column = Y_column
        self.X_columns = [col for col in self.file.columns if col != self.Y_column and col not in self.exclude]

    def options(self, file):
        text = f"{_}\nReading {file} file from CSV..\n{_}"
        print(text)
        self.file = pd.read_csv(file)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_colwidth', None)
        return self.file
    
    def prepare_Y(self, Y_column):
        text = f"{_}\nExtracting {Y_column} and converting it to numpy.\n{_}"
        print(text)
        Y = self.file[Y_column]
        Y = Y.replace({True: np.float64(1), False: np.float64(0)})
        self.Y_raw = Y
        self.Y = Y.to_numpy()
        self.file.drop(Y_column, axis=1, inplace=True)
        self.file.set_index('Id', inplace=True)
        return self.Y, self.Y_raw

    def prepare_X(self, columns_to_exclude=None):
        if columns_to_exclude == None:
            text = f"{_}\nNo columns were excluded. Left dataframe is being converted to numpy array.\n{_}"
            print(text)
            X = self.file
            self.X_raw = X
            self.X = X.to_numpy()
        else:
            text = f"{_}\nColumns {columns_to_exclude} were excluded. Left dataframe is being converted to numpy array.\n{_}"
            print(text)
            X = self.file.drop(columns_to_exclude, axis=1).reset_index(drop=True)
            self.X_raw = X
            self.X = X.to_numpy()
        return self.X, self.X_raw
    

class Data:
    def __init__(self):
        self.extract_values()
        self.prepare_data()

    def extract_values(self):
        data = PrepareFile(file='.\datafiles\Redwine.csv', Y_column='Recommended', columns_to_exclude=['Quality'])
        self.Y = data.Y
        self.X = data.X
        self.Y_raw = data.Y_raw
        self.X_raw = data.X_raw
        self.columns = data.X_columns
        return self.Y, self.X, self.Y_raw, self.X_raw, self.columns

    def get_raw(self):
        return {'Y_raw': self.Y_raw, 'X_raw': self.X_raw}

    def prepare_data(self):
        transform = StandardScaler()
        X = transform.fit_transform(self.X)  
        X_train, X_test, Y_train, Y_test = train_test_split(X, self.Y, test_size=0.2, random_state=4)
        self.data = X_train, X_test, Y_train, Y_test, self.columns
        return self.data