import pandas as pd
import numpy as np
import warnings

thetas = [[-0.0898128,  1.60903899, -3.12135088,  1.743678, -0.40458261,
           3.38368476, -3.7684895, -4.2133493, -0.94172908, -1.55310091,
           3.82731517], [2.50293707, -3.94242548, -3.7688851, -6.40945842, -1.47031278,
                         -1.90861129,  1.77961827,  2.71339798,  4.08953919, -2.18591727,
                         -1.24669285], [-3.90725299,  6.47279172,  5.87220278,  2.80670677, -4.17793139,
                                        -6.71170644,  2.04050333,  1.17856727, -2.94390873, -0.89887993,
                                        -3.31999965], [-4.12143556, -5.75348331,  2.04965982,  0.7636417,  5.1988597,
                                                       3.26893775, -0.84610574, -1.15258912, -1.44764606,  4.93104392,
                                                       -2.02442072]]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


houseNames = ['Gryffindor', 'Slytherin', 'Hufflepuff', 'Ravenclaw']


def predict(vals):
    proba = 0
    house = ''
    i = 0
    for t in thetas:
        sig = sigmoid(np.dot(t, vals))
        if sig > proba or proba == 0:
            proba = sig
            house = houseNames[i]
        i += 1
    return house


data = pd.read_csv('dataset_train.csv', index_col="Index",
                   usecols=['Index',  'Astronomy', 'Herbology', 'Divination', 'Muggle Studies', 'Ancient Runes',
                            'History of Magic', 'Transfiguration', 'Potions', 'Charms', 'Flying'])


data = data.dropna()

data=((data-data.min())/(data.max()-data.min()))

data.insert(0, 'ones', 1)

# print (len(data.index))

for row in data.itertuples(False):
    print(predict(row))
