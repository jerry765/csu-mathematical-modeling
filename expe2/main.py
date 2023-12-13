import csv
import numpy as np
import pandas as pd


def dataProcess_X(data):
    # income和sex列的值可以直接使用1位二进制码表示，不需要进行one-hot编码

    if "income" in data.columns:  # 删除sex和income列
        Data = data.drop(["income", "sex"], axis=1)
    else:
        Data = data.drop(["sex"], axis=1)

    # 离散属性列 col是column的简写 获取所有类型为Object的列
    listObjectData = [
        col for col in Data.columns if Data[col].dtypes == "object"]

    # 连续属性列
    listNonObjectData = [
        col for col in Data.columns if col not in listObjectData]
    ObjectData = Data[listObjectData]
    NonObjectData = Data[listNonObjectData]

    # 插入sex列，0代表male，1代表female astype(np.int)作用是把true和false转化为1和0
    NonObjectData.insert(0, "sex", (data["sex"] == " Female").astype(np.int64))
    # 2.one-hot编码
    ObjectData = pd.get_dummies(ObjectData)

    # 合并离散属性和连续属性
    Data = pd.concat([NonObjectData, ObjectData], axis=1)
    Data = Data.astype("int64")

    # 3.数据标准化
    Data = (Data - Data.mean()) / Data.std()
    return Data


def dataProcess_Y(data):
    # income属性，0代表小于等于50K，1代表大于50K
    return (data["income"] == " >50K").astype(np.int64)


def train(x_train, y_train):
    train_data_size = x_train.shape[0]
    test_data_size = y_train.shape[0]

    Mu1 = np.zeros((106,))  # 类别1均值
    Mu2 = np.zeros((106,))  # 类别2均值
    N1 = 0  # 类别1数量
    N2 = 0  # 类别2数量

    # 1.计算均值
    for i in range(train_data_size):
        if y_train[i] == 1:
            Mu1 += x_train[i]
            N1 += 1
        else:
            Mu2 += x_train[i]
            N2 += 1

    Mu1 /= N1
    Mu2 /= N2

    sigma1 = np.zeros((106, 106))  # 类别1方差
    sigma2 = np.zeros((106, 106))  # 类别2方差
    for i in range(train_data_size):
        if y_train[i] == 1:
            sigma1 += np.dot(np.transpose([x_train[i] - Mu1]), [x_train[i] - Mu1])
        else:
            sigma2 += np.dot(np.transpose([x_train[i] - Mu2]), [x_train[i] - Mu2])

    sigma1 /= N1
    sigma2 /= N2

    # 2.计算协方差
    Shared_sigma = (N1 / train_data_size) * sigma1 + (N2 / train_data_size) * sigma2

    return Mu1, Mu2, Shared_sigma, N1, N2


def cal(x_test, Mu1, Mu2, Shared_sigma, N1, N2):
    # 计算概率
    w = np.transpose(Mu1 - Mu2).dot(np.linalg.inv(Shared_sigma))
    b = -0.5 * np.transpose(Mu1).dot(np.linalg.inv(Shared_sigma)).dot(Mu1) + \
        0.5 * np.transpose(Mu2).dot(np.linalg.inv(Shared_sigma)).dot(Mu2) + \
        np.log(float(N1 / N2))
    arr = np.empty([x_test.shape[0], 1], dtype=float)
    for i in range(x_test.shape[0]):
        z = x_test[i, :].dot(w) + b
        z *= -1
        arr[i][0] = 1 / (1 + np.exp(z))
    return np.clip(arr, 1e-8, 1 - 1e-8)


def predict(x):
    ans = np.zeros([x.shape[0], 1], dtype=int)
    for i in range(len(x)):
        if x[i] > 0.5:
            ans[i] = 1
        else:
            ans[i] = 0

    return ans


if __name__ == "__main__":
    # 1.加载数据集
    trainData = pd.read_csv("./data/train.csv")
    testData = pd.read_csv("./data/test.csv")
    # 训练数据将107维降为106维，以适应测试数据
    X_train = dataProcess_X(trainData).drop(
        ['native_country_ Holand-Netherlands'], axis=1).values
    X_test = dataProcess_X(testData).values
    Y_train = dataProcess_Y(trainData).values
    # print(X_train.shape)
    # 计算概率所需的参数
    mu1, mu2, shared_sigma, n1, n2 = train(X_train, Y_train)
    result = cal(X_test, mu1, mu2, shared_sigma, n1, n2)
    answer = predict(result)
    print(answer[5:15])
