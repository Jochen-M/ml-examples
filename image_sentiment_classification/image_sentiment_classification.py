# coding: utf8

import csv
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout

import utils as U


num_class = 7
img_rows = 48
img_cols = 48

batch_size = 128
epochs = 1


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

    print(x_all.shape)
    exit()

    x_all = x_all.astype("float32") / 255
    x_all = np.reshape(x_all, (x_all.shape[0], img_rows, img_cols, 1))
    y_all = keras.utils.to_categorical(y_all, num_class)

    x_train, y_train, x_test, y_test = U.split_testing_set(x_all, y_all)

    return x_train, y_train, x_test, y_test


def cnn(x_train, y_train, x_test, y_test):
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

    return model


def plot_saliency_map(model, x_test):
    conv2d_1_model = K.function([model.layers[0].input, K.learning_phase()], [model.layers[0].output])
    conv2d_1_output = conv2d_1_model([x_test, 0])[0]
    max_pooling2d_1_model = K.function([model.layers[0].input, K.learning_phase()], [model.layers[2].output])
    max_pooling2d_1_output = max_pooling2d_1_model([x_test, 0])[0]
    conv2d_2_model = K.function([model.layers[0].input, K.learning_phase()], [model.layers[3].output])
    conv2d_2_output = conv2d_2_model([x_test, 0])[0]
    max_pooling2d_2_model = K.function([model.layers[0].input, K.learning_phase()], [model.layers[5].output])
    max_pooling2d_2_output = max_pooling2d_2_model([x_test, 0])[0]

    for i in range(10):
        for _ in range(64):
            feature_map = conv2d_1_output[i, :, :, _]
            plt.subplot(8, 8, _ + 1)
            plt.imshow(feature_map, cmap='gray')
            plt.axis('off')

        plt.savefig(f"saliency_maps/conv2d_1_output_{i}.png")

        for _ in range(64):
            feature_map = max_pooling2d_1_output[i, :, :, _]
            plt.subplot(8, 8, _ + 1)
            plt.imshow(feature_map, cmap='gray')
            plt.axis('off')

        plt.savefig(f"saliency_maps/max_pooling2d_1_output_{i}.png")

        for _ in range(128):
            feature_map = conv2d_2_output[i, :, :, _]
            plt.subplot(8, 16, _ + 1)
            plt.imshow(feature_map, cmap='gray')
            plt.axis('off')

        plt.savefig(f"saliency_maps/conv2d_2_output_{i}.png")

        for _ in range(128):
            feature_map = max_pooling2d_2_output[i, :, :, _]
            plt.subplot(8, 16, _ + 1)
            plt.imshow(feature_map, cmap='gray')
            plt.axis('off')

        plt.savefig(f"saliency_maps/max_pooling2d_2_output_{i}.png")


def visualizing_filters():
    pass


def main():
    x_train, y_train, x_test, y_test = load_data()

    model = cnn(x_train, y_train, x_test, y_test)

    plot_saliency_map(model, x_test[:10])


if __name__ == "__main__":
    main()


# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d_1 (Conv2D)            (None, 46, 46, 64)        640
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 46, 46, 64)        0
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 23, 23, 64)        0
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 21, 21, 128)       73856
# _________________________________________________________________
# dropout_2 (Dropout)          (None, 21, 21, 128)       0
# _________________________________________________________________
# max_pooling2d_2 (MaxPooling2 (None, 7, 7, 128)         0
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 6272)              0
# _________________________________________________________________
# dense_1 (Dense)              (None, 1024)              6423552
# _________________________________________________________________
# dropout_3 (Dropout)          (None, 1024)              0
# _________________________________________________________________
# dense_2 (Dense)              (None, 256)               262400
# _________________________________________________________________
# dropout_4 (Dropout)          (None, 256)               0
# _________________________________________________________________
# dense_3 (Dense)              (None, 7)                 1799
# =================================================================
