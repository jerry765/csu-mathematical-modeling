import sys
import csv
import numpy as np
import pandas as pd
import math


# 读取数据
def load_data(file_path):
    data = []
    for i in range(18):
        data.append([])

    n_row = 0
    with open(file_path, 'r', errors="ignore", encoding='big5') as text:
        row = csv.reader(text, delimiter=",")
        for r in row:
            if n_row > 0:
                for i in range(3, 27):
                    if r[i] != "NR":
                        data[(n_row - 1) % 18].append(float(r[i]))
                    else:
                        data[(n_row - 1) % 18].append(float(0))
            n_row = n_row + 1

    return data


# 数据预处理
def preprocess_data(data):
    x = []
    y = []

    for i in range(12):
        for j in range(471):
            x.append([])
            for t in range(18):
                for s in range(9):
                    x[471 * i + j].append(data[t][480 * i + j + s])
            y.append(data[9][480 * i + j + 9])

    x = np.array(x)
    y = np.array(y)
    x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

    return x, y


# 梯度下降训练模型
def train_model(x, y, learning_rate, num_iterations):
    w = np.zeros(len(x[0]))
    x_t = x.transpose()
    s_gra = np.zeros(len(x[0]))

    for i in range(num_iterations):
        hypo = np.dot(x, w)
        loss = hypo - y
        cost = np.sum(loss ** 2) / len(x)
        cost_a = math.sqrt(cost)
        gra = np.dot(x_t, loss)
        s_gra += gra ** 2
        ada = np.sqrt(s_gra)
        w = w - learning_rate * gra / ada

    return w


# 预测并保存结果
def predict_and_save(w, test_file):
    test_x = []
    n_row = 0
    with open(test_file, 'r', encoding='big5') as text:
        row = csv.reader(text, delimiter=',')
        for r in row:
            if n_row % 18 == 0:
                test_x.append([])
                for i in range(2, 11):
                    test_x[n_row // 18].append(float(r[i]))
            else:
                for i in range(2, 11):
                    if r[i] != "NR":
                        test_x[n_row // 18].append(float(r[i]))
                    else:
                        test_x[n_row // 18].append(0)
            n_row = n_row + 1

    test_x = np.array(test_x)
    test_x = np.concatenate((np.ones((test_x.shape[0], 1)), test_x), axis=1)

    ans = []
    result_file = './result/result.csv'
    with open(result_file, "w", encoding="gbk", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["数据编号", "PM2.5预测结果"])
        for i in range(len(test_x)):
            ans.append(["id_" + str(i)])
            a = np.dot(w, test_x[i])
            ans[i].append(a)
            csv_writer.writerow([str(i), a])
    f.close()

    print("共有预测结果%d条" % (len(ans)))


def main():
    train_file = './data/train.csv'
    test_file = './data/test.csv'
    learning_rate = 10
    num_iterations = 10

    data = load_data(train_file)
    x, y = preprocess_data(data)
    w = train_model(x, y, learning_rate, num_iterations)

    np.save('./model/model.npy', w)

    w = np.load('./model/model.npy')

    predict_and_save(w, test_file)


if __name__ == "__main__":
    main()
