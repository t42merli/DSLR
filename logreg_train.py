import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


data = pd.read_csv('dataset_train.csv', index_col="Index",
                   usecols=['Index', 'Hogwarts House', 'Astronomy', 'Herbology', 'Divination', 'Muggle Studies', 'Ancient Runes',
                            'History of Magic', 'Transfiguration', 'Potions', 'Charms', 'Flying'])

houses = data["Hogwarts House"]

houseNames = ['Gryffyndor', 'Slytherin', 'Hufflepuff', 'Ravenclaw']

data = data.drop('Hogwarts House', 1)

data = data.dropna()

data.insert(0, 'ones', 1)

size = len(data.index)

data = data.to_numpy()

thetas = np.ones(11)

def train(thetas, house):
    tmp = np.array(thetas)
    y = []
    for h in houses:
        y.append(1 if h == house else 0)
    i = 0
    while(i < 11):
        summ = 0
        index = 0
        for row in data:
            summ += (sigmoid(np.dot(thetas, row)) - y[index]) * row[i]
            index += 1
        tmp[i] = 1/size * summ
        i += 1
    return tmp


thetas = []

for house in houseNames:
    newThetas = np.zeros(11)
    i = 50
    while (i):
        i -= 1
        newThetas = train(newThetas, house)
    thetas.append(newThetas)

print(thetas)