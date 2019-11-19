import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('dataset_train.csv', index_col="Index")


def getgrade(category):
    houses = {"Gryffindor": [], "Slytherin": [],
              "Ravenclaw": [], "Hufflepuff": []}
    for index, row in data.iterrows():
        if(np.isnan(row[category]) == False):
            houses[row['Hogwarts House']].append(row[category])
    return houses


for column in data.columns:
    if(data[column].dtype == 'int' or data[column].dtype == 'float'):
        fig = plt.figure()
        fig.suptitle(column)
        houses = getgrade(column)
        plt.hist([houses['Gryffindor'], houses['Slytherin'], houses['Ravenclaw'],
                  houses['Hufflepuff']], bins=8, color=['yellow', 'green', 'red', 'blue'],
                 label=["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"], histtype='barstacked', alpha=0.5)

plt.ylabel('Nb élèves')
plt.xlabel('Note')
plt.legend()
plt.show()
