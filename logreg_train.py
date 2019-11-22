import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


data = pd.read_csv('dataset_train.csv', index_col="Index",
                   usecols=["Hogwarts House", 'Astronomy', 'Herbology', 'Divination', 'Muggle Studies', 'Ancient Runes',
                            'History of Magic', 'Transfiguration', 'Potions', 'Charms', 'Flying'])

data.dropna()

def train(house):
    thetas = np.zeros(11)
    y = []
    for i in data["Hogwarts House"]:
        y.append(1 if i == house else 0)
    i = 1
    while(i)
    

