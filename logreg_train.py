import numpy as np
import pandas as pd
import pickle
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if(len(sys.argv) < 2):
    print('please give the filename of the dataset used to train the model',
          '\n Usage: logreg_train "dataset"')
    exit()


data = pd.read_csv(sys.argv[1], index_col="Index",
                   usecols=['Index', 'Hogwarts House', 'Astronomy', 'Herbology', 'Divination', 'Muggle Studies', 'Ancient Runes',
                            'History of Magic', 'Transfiguration', 'Potions', 'Charms', 'Flying'])

train, test = train_test_split(data, test_size=0.2)

houseNames = ['Gryffindor', 'Slytherin', 'Hufflepuff', 'Ravenclaw']

houses = {'Gryffindor': [], 'Slytherin': [], 'Hufflepuff': [], 'Ravenclaw': []}

for houseName in houseNames:
    for h in train['Hogwarts House']:
        houses[houseName].append(1 if h == houseName else 0)
    houses[houseName] = np.array(houses[houseName])

train = train.drop('Hogwarts House', 1)

train = train.fillna(train.mean())

model = {'houseNames': houseNames, 'min': train.min(), 'max': train.max(),
         'mean': train.mean()}

train = ((train-train.min())/(train.max()-train.min()))

train.insert(0, 'ones', 1)

m = len(train.index)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cost_f(thetas, y):
    sig = sigmoid(np.dot(train, thetas))
    return -1.0/m * np.sum(y * np.log(sig) + (1-y) * np.log(1-sig))

def grad_desc(thetas, y):
    tmp = np.array(thetas)
    i = 0
    for column in train.columns:
        summ = np.sum((sigmoid(np.dot(train, thetas)) - y) * train[column])
        tmp[i] = tmp[i] - 1/m * summ
        i += 1
    return tmp


thetas = {'Gryffindor': [], 'Slytherin': [], 'Hufflepuff': [], 'Ravenclaw': []}

print("Training model...\n")

for house in houseNames:
    newThetas = np.zeros(11)
    cost = cost_f(newThetas, houses[house])
    diff = 1
    while (diff > 0.0005):
        old_cost = cost
        newThetas = grad_desc(newThetas, houses[house])
        cost = cost_f(newThetas, houses[house])
        diff = old_cost - cost
    thetas[house] = newThetas

model['thetas'] = thetas

modelFile = open('model', 'wb')

pickle.dump(model, modelFile)

modelFile.close()

print("Training done\nNew model saved in file 'model'\nHere is the new thetas :\n")

print(thetas)

def predict(vals):
    proba = 0
    house = ''
    for h in model['houseNames']:
        sig = sigmoid(np.dot(model['thetas'][h], vals))
        if sig > proba or proba == 0:
            proba = sig
            house = h
    return house

reality = test['Hogwarts House']

test = test.drop('Hogwarts House', 1)

test = test.fillna(model['mean'])

test = (test-model['min'])/(model['max']-model['min'])

test.insert(0, 'ones', 1)

prediction = []

for _index, row in test.iterrows():
    prediction.append(predict(row))


print("\n Model accuracy score on test set from the provided dataset: ", end='')
print(accuracy_score(reality, prediction))