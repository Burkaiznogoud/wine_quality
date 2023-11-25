from prepare_file import PrepareFile
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



data = PrepareFile(file='.\datafiles\Redwine.csv', Y_column='Recommended', columns_to_exclude=['Quality', 'Ph', 'Free sulfur dioxide', 'Density', 'Fixed acidity', 'Volatile acidity'])

frame = pd.DataFrame(data.X, columns=data.X_columns)

correlation_matrix = pd.DataFrame(data.X).corr()
print(correlation_matrix)
plt.figure(figsize=(10, 7))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", xticklabels=data.X_columns, yticklabels=data.X_columns)
plt.show()