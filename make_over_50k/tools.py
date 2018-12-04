# coding: utf8

import numpy as np
import pandas as pd
from math import floor


def load_data(train_data_path, train_label_path, test_data_path, test_label_path):
    x_all = pd.read_csv(train_data_path, sep=',', header=0)
    x_all = np.array(x_all.values)
    y_all = pd.read_csv(train_label_path, sep=',', header=0)
    y_all = np.array(y_all.values)
    x_test = pd.read_csv(test_data_path, sep=',', header=0)
    x_test = np.array(x_test.values)
    y_test = pd.read_csv(test_label_path, sep=',', header=0)
    y_test = np.array(y_test)[:, -1]
    return x_all, y_all, x_test, y_test


def normalize(x_all, x_test):
    """ Normalization:
        x_norm = (x - avg) / std
    """
    x_train_test = np.concatenate((x_all, x_test))
    mu = np.sum(x_train_test, axis=0) / x_train_test.shape[0]
    mu = np.tile(mu, (x_train_test.shape[0], 1))
    sigma = np.std(x_train_test, axis=0)
    sigma = np.tile(sigma, (x_train_test.shape[0], 1))
    x_train_test_normed = (x_train_test - mu) / sigma

    x_all = x_train_test_normed[: x_all.shape[0]]
    x_test = x_train_test_normed[x_all.shape[0]:]
    return x_all, x_test


def shuffle(x, y):
    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    return x[randomize], y[randomize]


def split_validation_set(x_all, y_all, percentage):
    validation_set_size = floor(x_all.shape[0] * percentage)

    x_all, y_all = shuffle(x_all, y_all)

    x_train, y_train = x_all[validation_set_size:], y_all[validation_set_size:]
    x_valid, y_valid = x_all[:validation_set_size], y_all[:validation_set_size]

    return x_train, y_train, x_valid, y_valid


def sigmoid(z):
    s = 1.0 / (1.0 + np.exp(-z))
    return np.clip(s, 1e-8, 1-(1e-8))
