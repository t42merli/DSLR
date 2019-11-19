import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('dataset_train.csv')

houses = {"Gryffindor": [], "Slytherin": [], "Ravenclaw": [], "Hufflepuff": []}

category = 'Care of Magical Creatures'

for index, row in data.iterrows():
    if(np.isnan(row[category]) == False):
        houses[row['Hogwarts House']].append(row[category])

plt.hist([houses['Gryffindor'], houses['Slytherin'], houses['Ravenclaw'],
          houses['Hufflepuff']], bins=8, color=['yellow', 'green', 'red', 'blue'],
         label=["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"], histtype='barstacked')

plt.ylabel('Nb élèves')
plt.xlabel('Note (care of maical creatures)')
plt.legend()
plt.show()
