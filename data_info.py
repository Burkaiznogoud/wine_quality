import pandas as pd
import numpy as np


df_wine = pd.read_csv("WineQT.csv")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

def data_basic_information(data):
    print("Shape: \n", data.shape, "\n",
    "Column names: \n", data.columns, "\n",
    "Data types: \n", data.dtypes, "\n",
    "Any NaNs: \n", data.isna().sum(), "\n",
    f"Statistical data of: \n{data}", data.describe(), "\n",
    "First 5 rows: \n", data.head(n=5), "\n")

capitalized_columns = { name : name.capitalize() for name in df_wine.columns}

df_wine.rename(columns=capitalized_columns, inplace=True)

data_basic_information(df_wine)

df_wine.to_csv("RedWine.csv", index=False)

def show_names(data = df_wine):
    for name in data.columns:
        print(name)