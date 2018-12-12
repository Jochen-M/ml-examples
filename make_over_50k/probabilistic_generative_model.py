# coding: utf8

import os
import argparse
import numpy as np
import pandas as pd

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


def evaluate(x, y, mu1, mu2, shared_sigma, N1, N2):
    sigma_inverse = np.linalg.inv(shared_sigma)
    w = np.dot((mu1 - mu2), sigma_inverse)
    b = - 0.5 * np.dot(np.dot([mu1], sigma_inverse), mu1)\
        + 0.5 * np.dot(np.dot([mu2], sigma_inverse), mu2)\
        + np.log(float(N1) / N2)
    z = np.dot(x, w) + b
    y_ = np.around(U.sigmoid(z))
    result = (y_ == np.squeeze(y))
    return float(result.sum()) / x.shape[0], y_


def train(x_all, y_all, save_dir):
    x_train, y_train, x_valid, y_valid = U.split_validation_set(x_all, y_all, 0.1)

    # Gaussian distribution parameters
    train_data_size = x_train.shape[0]
    num_fetures = x_train.shape[1]
    cnt1 = 0
    cnt2 = 0

    mu1 = np.zeros((num_fetures,))
    mu2 = np.zeros((num_fetures,))
    for i in range(train_data_size):
        if y_train[i] == 0:
            mu1 += x_train[i]
            cnt1 += 1
        else:
            mu2 += x_train[i]
            cnt2 += 1
    mu1 /= cnt1
    mu2 /= cnt2

    sigma1 = np.zeros((num_fetures, num_fetures))
    sigma2 = np.zeros((num_fetures, num_fetures))
    for i in range(train_data_size):
        if y_train[i] == 0:
            sigma1 += np.dot(np.transpose([x_train[i] - mu1]), [x_train[i] - mu1])
        else:
            sigma2 += np.dot(np.transpose([x_train[i] - mu2]), [x_train[i] - mu2])
    sigma1 /= cnt1
    sigma2 /= cnt2
    shared_sigma = float(cnt1) / train_data_size * sigma1 + float(cnt2) / train_data_size * sigma2

    N1 = cnt1
    N2 = cnt2

    print("=====Saving Param=====")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    param_dict = {"mu1": mu1, "mu2": mu2, "shared_sigma": shared_sigma, "N1": [N1], "N2": [N2]}
    for key in sorted(param_dict):
        print(f"Saving {key}")
        np.savetxt(os.path.join(save_dir, key), param_dict[key])

    print("=====Validating=====")
    acc, _ = evaluate(x_valid, y_valid, mu1, mu2, shared_sigma, N1, N2)
    print(f"Validation acc = {acc}")

    return


def infer(x_test, y_test, save_dir, output_dir):
    # load parameters
    print(f"=====Loading Param from {save_dir}")
    mu1 = np.loadtxt(os.path.join(save_dir, "mu1"))
    mu2 = np.loadtxt(os.path.join(save_dir, "mu2"))
    shared_sigma = np.loadtxt(os.path.join(save_dir, "shared_sigma"))
    N1 = np.loadtxt(os.path.join(save_dir, "N1"))
    N2 = np.loadtxt(os.path.join(save_dir, "N2"))

    # predict
    print("=====Testing=====")
    acc, y_ = evaluate(x_test, y_test, mu1, mu2, shared_sigma, N1, N2)
    print(f"Testing acc = {acc}")

    print(f"=====Write output to {output_dir}")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_path = os.path.join(output_dir, "prediction.csv")
    with open(output_path, "w") as f:
        f.write("id, label")
        for i, v in enumerate(y_):
            f.write("%d, %d\n" % (i + 1, v))


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
        description="Probabilistic Generative Model for Binary Classification")
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
                        default="generative_params/", dest="save_dir",
                        help="Path to save the model parameters")
    parser.add_argument("--output_dir", type=str,
                        default="generative_output/", dest="output_dir",
                        help="Path to save the output")
    opts = parser.parse_args()
    main(opts)
