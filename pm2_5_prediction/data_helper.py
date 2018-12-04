# coding: utf8

import csv
import numpy as np


def load_train_data(train_data_path='data/train.csv', encoding='big5'):
    data = []
    for i in range(18):
        data.append([])

    n_row = 0
    text = open(train_data_path, 'r', encoding=encoding)
    rows = csv.reader(text, delimiter=',')
    for row in rows:
        if n_row != 0:
            for i in range(3, 27):
                if row[i] != 'NR':
                    data[(n_row - 1) % 18].append(float(row[i]))
                else:
                    data[(n_row - 1) % 18].append(float(0))
        n_row += 1
    text.close()

    x_train, y_train = parse_data_to_x_y(data)
    return x_train, y_train


def parse_data_to_x_y(data):
    x = []
    y = []
    # 12 个月
    for i in range(12):
        # 10 个小时为一组，每个月共有 471 组训练数据
        for j in range(471):
            x.append([])
            # 18 个观测项目
            for t in range(18):
                for s in range(9):
                    x[471 * i + j].append(data[t][480 * i + j + s])
            y.append(data[9][480 * i + j + 9])
    x = np.array(x)
    y = np.array(y)

    # add square term
    x = np.concatenate((x, x**2), axis=1)

    # add bias
    x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

    return x, y


def load_test_data(
        test_data_path='data/x_test.csv',
        test_label_path='data/y_test.csv',
        encoding='big5'):
    x_test = []
    n_row = 0
    text = open(test_data_path, 'r', encoding=encoding)
    rows = csv.reader(text, delimiter=',')
    for row in rows:
        if n_row % 18 == 0:
            x_test.append([])
        for i in range(2, 11):
            if row[i] != 'NR':
                x_test[n_row // 18].append(float(row[i]))
            else:
                x_test[n_row // 18].append(float(0))
        n_row += 1
    text.close()

    x_test = np.array(x_test)
    x_test = np.concatenate((x_test, x_test ** 2), axis=1)
    x_test = np.concatenate((np.ones((x_test.shape[0], 1)), x_test), axis=1)

    y_test = []
    n_row = 0
    text = open(test_label_path, 'r', encoding=encoding)
    rows = csv.reader(text, delimiter=',')
    for row in rows:
        if n_row != 0:
            y_test.append(float(row[1]))
        n_row += 1
    y_test = np.array(y_test)

    return x_test, y_test
