import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
wine = pd.read_csv("RedWine.csv")

def create_scatter_plot(data, x_factor, y_factor):
    plt.figure(figsize = (12 , 6))
    plt.scatter(data[x_factor], data[y_factor], marker='.')
    plt.xlabel(f"{x_factor}", fontsize = 12, fontweight = 'bold', labelpad = 15)
    plt.ylabel(f"{y_factor}", fontsize = 12, fontweight = 'bold', labelpad = 15)
    plt.title(f"{y_factor} on {x_factor}", fontsize = 16, fontweight = 'bold')
    plt.show()

def column_names(dataframe):
    col_names = {number : col_name for number, col_name in enumerate(dataframe.columns)}
    return col_names

def list_options(data, plot):
    col_names = column_names(data)
    for key, value in col_names.items():
        print(key, value)
    try:
        first = int(input("Choose first factor by number: "))
        second = int(input("Choose second factor by number: "))
        plot(data, col_names[first], col_names[second])
    except ValueError:
        print("Value error! Choose number from list.")
    else:
        return
    
list_options(data=wine, plot=create_scatter_plot)
    
