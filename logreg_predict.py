import pandas as pd
import numpy as np
import pickle
import sys


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict(vals, model):
    proba = 0
    house = ''
    for h in model['houseNames']:
        sig = sigmoid(np.dot(model['thetas'][h], vals))
        if sig > proba or proba == 0:
            proba = sig
            house = h
    return house

if __name__ == "__main__":
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

    data = pd.read_csv(sys.argv[1], index_col="Index",
                       usecols=['Index', 'Astronomy', 'Herbology', 'Divination', 'Muggle Studies', 'Ancient Runes',
                                'History of Magic', 'Transfiguration', 'Potions', 'Charms', 'Flying'])

    data = data.fillna(model['mean'])

    data = (data-model['min'])/(model['max']-model['min'])

    data.insert(0, 'ones', 1)

    f = open('houses.csv', 'w')

    f.write('Index,Hogwarts House\n')

    for index, row in data.iterrows():
        f.write("%d,%s\n"% (index,predict(row, model)))
