# coding: utf8

import csv
import keras
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout

import utils as U


def load_data(train_data_path="data/train.csv"):
    x_all = []
    y_all = []

    n_row = 0
    text = open(train_data_path, "r")
    rows = csv.reader(text)
    for row in rows:
        if n_row != 0:
            x_all.append(list(map(int, row[1].split())))
            y_all.append(int(row[0]))
        n_row += 1

    x_all = np.array(x_all)
    y_all = np.array(y_all)

    x_train, y_train, x_test, y_test = U.split_testing_set(x_all, y_all)

    return x_train, y_train, x_test, y_test


def cnn():
    num_class = 7
    img_rows = 48
    img_cols = 48

    batch_size = 128
    epochs = 100

    x_train, y_train, x_test, y_test = load_data()

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train = np.reshape(x_train, (x_train.shape[0], img_rows, img_cols, 1))
    x_test = np.reshape(x_test, (x_test.shape[0], img_rows, img_cols, 1))
    y_train = keras.utils.to_categorical(y_train, num_class)
    y_test = keras.utils.to_categorical(y_test, num_class)

    model = Sequential()

    model.add(Conv2D(64, (3, 3), activation="relu", input_shape=(img_rows, img_cols, 1)))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(num_class, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.25)
    score = model.evaluate(x_test, y_test)

    print("Test Loss:", score[0])
    print("Test Accuracy: ", score[1])


if __name__ == "__main__":
    cnn()
