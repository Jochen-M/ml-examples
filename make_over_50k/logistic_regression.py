# coding: utf8

import os
import argparse
import numpy as np
import pandas as pd
from math import floor

import utils as U


def load_data(train_data_path, train_label_path, test_data_path, test_label_path):
    x_all = pd.read_csv(train_data_path, sep=",", header=0)
    x_all = np.array(x_all.values)
    y_all = pd.read_csv(train_label_path, sep=",", header=0)
    y_all = np.array(y_all.values)
    x_test = pd.read_csv(test_data_path, sep=",", header=0)
    x_test = np.array(x_test.values)
    y_test = pd.read_csv(test_label_path, sep=",", header=0)
    y_test = np.array(y_test)[:, -1]
    return x_all, y_all, x_test, y_test


def evaluate(w, b, x, y):
    z = np.dot(x, w) + b
    y_ = U.sigmoid(z)
    y_ = np.around(y_)
    result = (y_ == np.squeeze(y))
    return float(result.sum()) / x.shape[0]


def train(x_all, y_all, save_dir):
    x_train, y_train, x_valid, y_valid = U.split_validation_set(x_all, y_all, 0.1)

    # initialize parameters, hyperparameters
    w = np.zeros((x_all.shape[1],))
    b = np.zeros((1,))
    lr = 0.01
    batch_size = 32
    step_num = floor(x_train.shape[0] / batch_size)
    epochs = 1000
    save_params_iter = 50

    # start training
    total_loss = 0.0
    for epoch in range(epochs):
        # validation and saving parameters
        if epoch % save_params_iter == 0 and epoch != 0:
            print(f"=====Saving Param at Epoch {epoch}=====")
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            np.savetxt(os.path.join(save_dir, "w"), w)
            np.savetxt(os.path.join(save_dir, "b"), b)
            print(f"Epoch avg loss = {total_loss / (save_params_iter * step_num)}")
            total_loss = 0.0
            acc = evaluate(w, b, x_valid, y_valid)
            print(f"Validation acc = {acc}")

        # random shuffle
        x_train, y_train = U.shuffle(x_train, y_train)
        # train with batch
        for step in range(step_num):
            X = x_train[step*batch_size: (step+1)*batch_size]
            Y = y_train[step*batch_size: (step+1)*batch_size]

            z = np.dot(X, w) + b
            y = U.sigmoid(z)

            cross_entropy = -1 * (np.dot(np.squeeze(Y), np.log(y)) +
                                  np.dot((1-np.squeeze(Y)), np.log(1-y)))
            total_loss += cross_entropy

            w_grad = np.mean(-1 * X * (np.squeeze(Y) - y).reshape((batch_size, 1)), axis=0)
            b_grad = np.mean(-1 * (np.squeeze(Y) - y))

            # SGD updating parameters
            w = w - lr * w_grad
            b = b - lr * b_grad

    return


def infer(x_test, y_test, save_dir, output_dir):
    # load parameters
    print(f"=====Loading Param from {save_dir}=====")
    w = np.loadtxt(os.path.join(save_dir, "w"))
    b = np.loadtxt(os.path.join(save_dir, "b"))

    # predict
    acc = evaluate(w, b, x_test, y_test)
    print(f"Testing acc = {acc}")

    z = np.dot(x_test, w) + b
    y_ = np.around(U.sigmoid(z))

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_path = os.path.join(output_dir, "logistic_prediction.csv")
    with open(output_path, "w") as f:
        f.write("id, label")
        for i, v in enumerate(y_):
            f.write("%d, %d\n" % (i+1, v))

    return


def main(opts):
    x_all, y_all, x_test, y_test = load_data(
        opts.train_data_path, opts.train_label_path,
        opts.test_data_path, opts.test_label_path)
    x_all, x_test = U.normalize(x_all, x_test)

    if opts.train:
        train(x_all, y_all, opts.save_dir)
    elif opts.infer:
        infer(x_test, y_test, opts.save_dir, opts.output_dir)
    else:
        print("Error: Argument --train or --infer not found")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Logistic Regression with Gradient Descent Method")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--train", action="store_true", default=False,
                       dest="train", help="Input --train to Train")
    group.add_argument("--infer", action="store_true", default=False,
                       dest="infer", help="Input --infer to Infer")
    group.add_argument("--train_data_path", type=str,
                       default="data/X_train", dest="train_data_path",
                       help="Path to training data")
    parser.add_argument("--train_label_path", type=str,
                        default="data/Y_train", dest="train_label_path",
                        help="Path to training data\"s label")
    group.add_argument("--test_data_path", type=str,
                       default="data/X_test", dest="test_data_path",
                       help="Path to testing data")
    group.add_argument("--test_label_path", type=str,
                       default="data/Y_test.csv", dest="test_label_path",
                       help="Path to testing data\"s label")
    parser.add_argument("--save_dir", type=str,
                        default="logistic_params/", dest="save_dir",
                        help="Path to save the model parameters")
    parser.add_argument("--output_dir", type=str,
                        default="logistic_output/", dest="output_dir",
                        help="Path to save the output")
    opts = parser.parse_args()
    main(opts)
