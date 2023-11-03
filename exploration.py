import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filename = "RedWine.csv"

def set_options(file):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.options.display.float_format = '{:.3f}'.format
    return pd.read_csv(file)

def create_scatter_plot(data, x_factor, y_factor, fontsize):
    plt.figure(figsize = (12 , 6))
    plt.scatter(data[x_factor], data[y_factor], marker='.')
    plt.xlabel(f"{x_factor}", fontsize = fontsize, fontweight = 'bold', labelpad = 15)
    plt.ylabel(f"{y_factor}", fontsize = fontsize, fontweight = 'bold', labelpad = 15)
    plt.title(f"{y_factor} on {x_factor}", fontsize = fontsize + 4, fontweight = 'bold')
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
        plot(data, col_names[first], col_names[second], fontsize=12)
    except ValueError:
        print("Value error! Choose number from list.")
    else:
        return
    
loop = True
wine = set_options(file=filename)

while loop:
    chart_overview = input("Do you want to view charts?\n Yes \ No\n")
    print(chart_overview, type(chart_overview))
    if chart_overview == "Yes":
        list_options(data=wine, plot=create_scatter_plot) 
    else:
        loop = False
        
        

wine["Recommended"] = wine["Quality"] >= 7
wine.to_csv(filename, index=False)
