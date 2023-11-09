import pandas as pd
import numpy as np

class PrepareFile:
    def __init__(self, file, Y_column, columns_to_exclude=None):
        self.exclude = columns_to_exclude
        self.file = self.options(file)
        self.X = self.prepare_X(columns_to_exclude)
        self.Y = self.prepare_Y(Y_column=Y_column)
        self.Y_column = Y_column
        self.X_columns = [col for col in self.file.columns if col != self.Y_column and col not in self.exclude]
        

    def options(self, file):
        file = pd.read_csv(file)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_colwidth', None)
        return file
    
    def prepare_Y(self, Y_column):
        Y = self.file[Y_column]
        Y = Y.replace({True: np.float64(1), False: np.float64(0)})
        Y = Y.to_numpy()
        self.file = self.file.drop(Y_column, axis=1)
        self.file.set_index('Id', inplace=True)
        return Y

    def prepare_X(self, columns_to_exclude=None):
        if columns_to_exclude == None:
            X = self.file.to_numpy()
        else:
            X = self.file.drop(columns=columns_to_exclude, axis=1).to_numpy()
        return X