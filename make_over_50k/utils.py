# coding: utf8

import numpy as np
from math import floor


def normalize(x_train, x_test):
    """ Normalization:
        x_norm = (x - avg) / std
    """
    x_train_test = np.concatenate((x_train, x_test))
    mu = np.sum(x_train_test, axis=0) / x_train_test.shape[0]
    mu = np.tile(mu, (x_train_test.shape[0], 1))
    sigma = np.std(x_train_test, axis=0)
    sigma = np.tile(sigma, (x_train_test.shape[0], 1))
    x_train_test_normed = (x_train_test - mu) / sigma

    x_train = x_train_test_normed[: x_train.shape[0]]
    x_test = x_train_test_normed[x_train.shape[0]:]
    return x_train, x_test


def shuffle(x, y):
    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    return x[randomize], y[randomize]


def sigmoid(z):
    s = 1.0 / (1.0 + np.exp(-z))
    return np.clip(s, 1e-8, 1-(1e-8))


def split_data_set(x_all, y_all, percentage):
    split_size = floor(x_all.shape[0] * percentage)

    x_all, y_all = shuffle(x_all, y_all)

    x_1, y_1 = x_all[split_size:], y_all[split_size:]
    x_2, y_2 = x_all[:split_size], y_all[:split_size]

    return x_1, y_1, x_2, y_2


def split_validation_set(x_all, y_all, percentage=0.25):
    return split_data_set(x_all, y_all, percentage)
