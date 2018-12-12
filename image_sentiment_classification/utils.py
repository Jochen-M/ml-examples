# coding: utf8

import numpy as np
from math import floor


def shuffle(x, y):
    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    return x[randomize], y[randomize]


def split_data_set(x_all, y_all, percentage):
    split_size = floor(x_all.shape[0] * percentage)

    x_all, y_all = shuffle(x_all, y_all)

    x_1, y_1 = x_all[split_size:], y_all[split_size:]
    x_2, y_2 = x_all[:split_size], y_all[:split_size]

    return x_1, y_1, x_2, y_2


def split_testing_set(x_all, y_all, percentage=0.3):
    return split_data_set(x_all, y_all, percentage)
