import pandas as pd
import numpy as np
import pickle
import sys

if(len(sys.argv) < 3):
    print('please give the filename of dataset to make prediction on and the filename of the trained model',
          '\n Usage: logreg_predict "dataset" "model"')
    exit()

try:
    modelFile = open(sys.argv[2], 'rb')
    model = pickle.load(modelFile)
    modelFile.close()
except FileNotFoundError:
    print('please provide a valid training model file')
    exit()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict(vals):
    proba = 0
    house = ''
    i = 0
    for t in model['thetas']:
        sig = sigmoid(np.dot(t, vals))
        if sig > proba or proba == 0:
            proba = sig
            house = model['houseNames'][i]
        i += 1
    return house


data = pd.read_csv(sys.argv[1], index_col="Index",
                   usecols=['Index', 'Astronomy', 'Herbology', 'Divination', 'Muggle Studies', 'Ancient Runes',
                            'History of Magic', 'Transfiguration', 'Potions', 'Charms', 'Flying'])

data.fillna(model['mean'])

data = (data-model['min'])/(model['max']-model['min'])

data.insert(0, 'ones', 1)

for index, row in data.iterrows():
    print("%d,%s"% (index,predict(row)))
