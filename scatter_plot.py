import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('dataset_train.csv')
np.warnings.filterwarnings('ignore')

x = range(0, 1600)
matrix = data.corr()
print(matrix)

fig = plt.figure()
fig.suptitle('Correlation')
plt.scatter(data['Defense Against the Dark Arts'], data['Astronomy'])

fig = plt.figure()
fig.suptitle('Atronomy')
plt.scatter(x, data['Astronomy'])

fig = plt.figure()
fig.suptitle('Defense Against The Dark Arts')
datda= np.array(data['Defense Against the Dark Arts'])
datda = datda * -1
plt.scatter(x, datda, color='red')



plt.legend()
plt.show()
