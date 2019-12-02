import numpy as np
import pandas as pd
import pickle


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


data = pd.read_csv('dataset_train.csv', index_col="Index",
                   usecols=['Index', 'Hogwarts House', 'Astronomy', 'Herbology', 'Divination', 'Muggle Studies', 'Ancient Runes',
                            'History of Magic', 'Transfiguration', 'Potions', 'Charms', 'Flying'])

houseNames = ['Gryffindor', 'Slytherin', 'Hufflepuff', 'Ravenclaw']

houses = {'Gryffindor': [], 'Slytherin': [], 'Hufflepuff': [], 'Ravenclaw': []}

for houseName in houseNames:
    for h in data['Hogwarts House']:
        houses[houseName].append(1 if h == houseName else 0)

data = data.drop('Hogwarts House', 1)

data = data.fillna(data.mean())

data = ((data-data.min())/(data.max()-data.min()))

model = {'houseNames': houseNames, 'min': data.min(), 'max': data.max(),
         'mean': data.mean()}

data.insert(0, 'ones', 1)

m = len(data.index)

def train(thetas, house):
    tmp = np.array(thetas)
    i = 0
    for column in data.columns:
        if(column != 'houses'):
            summ = np.sum((sigmoid(np.dot(data, thetas)) - houses[house])
                          * data[column])
            tmp[i] = tmp[i] - 1/m * summ
            i += 1
    return tmp


thetas = []

for house in houseNames:
    newThetas = np.zeros(11)
    i = 2000
    while (i):
        i -= 1
        newThetas = train(newThetas, house)
    thetas.append(newThetas)

print(thetas)

model['thetas'] = thetas

modelFile = open('model', 'wb')

pickle.dump(model, modelFile)

modelFile.close()
