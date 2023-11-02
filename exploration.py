import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
wine = pd.read_csv("RedWine.csv")

print(wine["Ph"])
print(wine[["Volatile acidity"]])

def create_scatter_plot(data, x_factor, y_factor):
    plt.figure(figsize = (12 , 6))
    plt.scatter(data[x_factor], data[y_factor], marker='.')
    plt.xlabel(f"{x_factor}", fontsize = 12, fontweight = 'bold', labelpad = 15)
    plt.ylabel(f"{y_factor}", fontsize = 12, fontweight = 'bold', labelpad = 15)
    plt.title(f"{y_factor} on {x_factor}", fontsize = 16, fontweight = 'bold')
    plt.show()

create_scatter_plot(data = wine , x_factor = "Ph", y_factor= "Volatile acidity")


