import pandas as pd
import numpy as np

class PrepareFile:
    def __init__(self, file, column) -> None:
        self.file = self.options(file)
        self.Y_column = column
        self.X_columns = [col for col in self.file.columns if col != self.Y_column]
        self.X = self.prepare_X()
        self.Y = self.prepare_Y(column=column)

    def options(self, file):
        file = pd.read_csv(file)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_colwidth', None)
        return file
    
    def prepare_Y(self, column):
        Y = self.file[column]
        Y = Y.replace({True: np.float64(1), False: np.float64(0)})
        Y = Y.to_numpy()
        self.file = self.file.drop(column, axis=1)
        self.file.set_index('Id', inplace=True)
        return Y

    def prepare_X(self):
        X = self.file.to_numpy()
        return X