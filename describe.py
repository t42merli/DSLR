import pandas as pd
import numpy as np
import math


data = pd.read_csv('dataset_train.csv')


def count(array):
    counter = 0
    for i in array:
        if(np.isnan(i) != True):
            counter += 1
    return counter


def mean(array):
    sum = 0
    counter = 0
    for i in array:
        if(np.isnan(i) != True):
            sum += i
            counter += 1
    return (sum / counter)


def std(array):
    m = mean(array)
    arr = np.array(array)
    arr = arr - m
    arr = arr * arr
    m = mean(arr)
    return np.sqrt(m)


def min(array):
    minimum = array[0]
    for i in array:
        if(np.isnan(i) != True and (i < minimum or np.isnan(minimum))):
            minimum = i
    return minimum


def max(array):
    minimum = array[0]
    for i in array:
        if(np.isnan(i) != True and (i > minimum or np.isnan(minimum))):
            minimum = i
    return minimum


def percentile(array, percent):
    sorted = np.sort(array)
    index = percent/100 * count(sorted) - 1
    if(index.is_integer()):
        return ((sorted[int(index)] + sorted[int(index) + 1]) / 2)
    return sorted[math.ceil(index)]


funcArr = [count, mean, std, min]
nameArr = ["count", "mean", "std", "min", "25%", "50%", "75%", "Max"]

print('     ', end='')
for column in data.columns:
    if(data[column].dtype == 'int' or data[column].dtype == 'float'):
        print('%s' % '  ' if len(column) > 13 else "", '%13s' % column, end='')

print()

i = 0

while(i < 8):
    print('%-5s' % nameArr[i], end='')
    for column in data.columns:
        if(data[column].dtype == 'int' or data[column].dtype == 'float'):
            if(i > 3 and i < 7):
                print("%*.5f" % (len(column)+3 if len(column) > 13 else 14,
                                 percentile(data[column], (i - 3) * 25)), end='')
            elif (i > 6):
                print("%*.5f" % (len(column)+3 if len(column)
                                 > 13 else 14, max(data[column])), end='')
            else:
                print("%*.5f" %
                      (len(column)+3 if len(column) > 13 else 14, funcArr[i](data[column])), end='')
    i += 1
    print()
