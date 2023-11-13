import pandas as pd
import numpy as np

class PrepareFile:
    def __init__(self, file, Y_column, columns_to_exclude=None):
        self.exclude = columns_to_exclude
        self.file = self.options(file)
        self.X = self.prepare_X(columns_to_exclude)
        self.Y = self.prepare_Y(Y_column)
        self.Y_column = Y_column
        self.X_columns = [col for col in self.file.columns if col != self.Y_column and col not in self.exclude]
        

    def options(self, file):
        print(20 * "-")
        print(f"Reading {file} file from CSV.")
        print(20 * "-")
        file = pd.read_csv(file)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_colwidth', None)
        return file
    
    def prepare_Y(self, Y_column):
        print(20 * "-")
        print(f"Extracting {Y_column} and converting it to numpy.")
        print(20 * "-")
        Y = self.file[Y_column]
        Y = Y.replace({True: np.float64(1), False: np.float64(0)})
        Y = Y.to_numpy()
        self.file = self.file.drop(Y_column, axis=1)
        self.file.set_index('Id', inplace=True)
        return Y

    def prepare_X(self, columns_to_exclude=None):
        if columns_to_exclude == None:
            print(20 * "-")
            print(f"No columns were excluded. Left dataframe is being converted to numpy array.")
            print(20 * "-")
            X = self.file.reset_index(drop=True)
            X.set_index(X.columns[0], inplace=True)
            X = X.to_numpy()
        else:
            print(20 * "-")
            print(f"Columns {columns_to_exclude} were excluded. Left dataframe is being converted to numpy array.")
            print(20 * "-")
            X = self.file.drop(columns_to_exclude, axis=1).reset_index(drop=True)
            X.set_index(X.columns[0], inplace=True)
            X = X.to_numpy()
        return X