import numpy as np
import pandas as pd
import pickle


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


data = pd.read_csv('dataset_train.csv', index_col="Index",
                   usecols=['Index', 'Hogwarts House', 'Astronomy', 'Herbology', 'Divination', 'Muggle Studies', 'Ancient Runes',
                            'History of Magic', 'Transfiguration', 'Potions', 'Charms', 'Flying'])

houseNames = ['Gryffindor', 'Slytherin', 'Hufflepuff', 'Ravenclaw']

data = data.dropna()

houses = data["Hogwarts House"]

data = data.drop('Hogwarts House', 1)

model = {'houses':houses, 'min':data.min(), max:data.max()}

data=((data-data.min())/(data.max()-data.min()))


data.insert(0, 'ones', 1)


m = len(data.index)

np.set_printoptions(threshold=np.inf)


thetas = np.ones(11)


def train(thetas, house):
    tmp = np.array(thetas)
    y = []
    for h in houses:
        y.append(1 if h == house else 0)
    i = 0
    for column in data.columns:
        summ = np.sum((sigmoid(np.dot(data, thetas)) - y) * data[column])
        tmp[i] = tmp[i] - 1/m * summ
        i += 1
    return tmp


thetas = []

for house in houseNames:
    newThetas = np.zeros(11)
    i = 1000
    while (i):
        i -= 1
        newThetas = train(newThetas, house)
    thetas.append(newThetas)

model['thetas'] = thetas

print(pickle.dumps(model))
