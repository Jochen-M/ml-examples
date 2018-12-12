# coding: utf8
# 任务描述：
#   任务目标：预测某年某月某日某时的 PM 2.5 值；
#   预测根据：前九个小时的所有观测数据；
# 数据描述：
#   数据来源：豐原地区 2014 年全年气象观测数据
#       （包括AMB_TEMP、CO、PM 2.5 等，共 18 项，间隔为 1 小时）；
#   训练数据：每月前 20 天的观测数据；
#   测试数据：每月后 10 天的观测数据；


import os
import numpy as np
import matplotlib.pyplot as plt
import utils as U

# read data
x_train, y_train = U.load_train_data()
x_test, y_test = U.load_test_data()

if os.path.exists("model.npy"):
    w = np.load("model.npy")
else:
    # init weight & other hyperparams
    w = np.zeros(x_train.shape[1])
    lr = 1
    epochs = 10000

    # start training
    x_t = x_train.transpose()
    sum_grad = np.zeros(x_train.shape[1])
    for i in range(epochs):
        pre = np.dot(x_train, w)
        loss = pre - y_train
        grad = np.dot(x_t, loss)
        sum_grad += grad ** 2
        w = w - lr / np.sqrt(sum_grad) * grad
        cost = np.sqrt(np.mean(np.sum(loss ** 2)))
        print(f"Iteration: {i} | Cost: {cost}")

    np.save("model.npy", w)


# predict on training data
y_ = np.dot(x_train, w)
print(np.mean(np.abs(y_train - y_)))    # MAE

plt.figure(figsize=(12.0, 8.0))
plt.subplot(2, 1, 1)
plt.plot(range(240), y_train[:240], color="red")
plt.plot(range(240), y_[:240], color="green")
plt.title("Training Data")
plt.xlabel(r"$h$", fontsize=16)
plt.ylabel(r"$PM 2.5$", fontsize=16)


# predict on testing data
y_ = np.dot(x_test, w)
print(np.mean(np.abs(y_test - y_)))    # MAE

plt.subplot(2, 1, 2)
plt.plot(range(240), y_test, color="red")
plt.plot(range(240), y_, color="green")
plt.title("Testing Data")
plt.xlabel(r"$h$", fontsize=16)
plt.ylabel(r"$PM 2.5$", fontsize=16)
plt.show()


# ans = []
# for i in range(len(x_test)):
#     ans.append(["id_" + str(i)])
#     a = np.dot(w, x_test[i])
#     ans[i].append(a)
#
# filename = "predict.csv"
# text = open(filename, "w+")
# w = csv.writer(text, delimiter=",", lineterminator="\n")
# w.writerow(["id", "value"])
# for i in range(len(ans)):
#     w.writerow(ans[i])
# text.close()
