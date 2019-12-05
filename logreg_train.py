import numpy as np
import pandas as pd
import pickle

data = pd.read_csv('dataset_train.csv', index_col="Index",
                   usecols=['Index', 'Hogwarts House', 'Astronomy', 'Herbology', 'Divination', 'Muggle Studies', 'Ancient Runes',
                            'History of Magic', 'Transfiguration', 'Potions', 'Charms', 'Flying'])

houseNames = ['Gryffindor', 'Slytherin', 'Hufflepuff', 'Ravenclaw']

houses = {'Gryffindor': [], 'Slytherin': [], 'Hufflepuff': [], 'Ravenclaw': []}

for houseName in houseNames:
    for h in data['Hogwarts House']:
        houses[houseName].append(1 if h == houseName else 0)
    houses[houseName] = np.array(houses[houseName])

data = data.drop('Hogwarts House', 1)

data = data.fillna(data.mean())

model = {'houseNames': houseNames, 'min': data.min(), 'max': data.max(),
         'mean': data.mean()}

data = ((data-data.min())/(data.max()-data.min()))

data.insert(0, 'ones', 1)

m = len(data.index)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cost_f(thetas, y):
    sig = sigmoid(np.dot(data, thetas))
    return -1.0/m * np.sum(y * np.log(sig) + (1-y) * np.log(1-sig))

def grad_desc(thetas, y):
    tmp = np.array(thetas)
    i = 0
    for column in data.columns:
        summ = np.sum((sigmoid(np.dot(data, thetas)) - y) * data[column])
        tmp[i] = tmp[i] - 1/m * summ
        i += 1
    return tmp


thetas = {'Gryffindor': [], 'Slytherin': [], 'Hufflepuff': [], 'Ravenclaw': []}

for house in houseNames:
    newThetas = np.zeros(11)
    cost = cost_f(newThetas, houses[house])
    diff = 1
    while (diff > 0.0001):
        old_cost = cost
        newThetas = grad_desc(newThetas, houses[house])
        cost = cost_f(newThetas, houses[house])
        diff = old_cost - cost
    thetas[house] = newThetas

print(thetas)

model['thetas'] = thetas

modelFile = open('model', 'wb')

pickle.dump(model, modelFile)

modelFile.close()
