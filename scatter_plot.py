import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('dataset_train.csv')

x = range(0, 1600)

for column in data.columns:
    if(data[column].dtype == 'int' or data[column].dtype == 'float'):
        plt.scatter(x, data[column], label=column)


plt.legend()
plt.show()
