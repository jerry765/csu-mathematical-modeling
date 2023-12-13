import numpy as np
from keras import preprocessing

########## Begin ##########
# 导入包 Sequential, Flatten, Dense, Embedding
from keras import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Embedding
########## End ##########

import os
from keras.datasets import imdb

max_features = 10000
maxlen = 20
file_path = './data/imdb.npz'

# 加载数据
with np.load(file_path, allow_pickle=True) as f:
    X_train, y_train = f['x_train'], f['y_train']
    X_test, y_test = f['x_test'], f['y_test']

# 对数据进行预处理
X_train = preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = preprocessing.sequence.pad_sequences(X_test, maxlen=maxlen)
X_train = np.array([[min(word, max_features - 1) for word in sequence] for sequence in X_train])
X_test = np.array([[min(word, max_features - 1) for word in sequence] for sequence in X_test])

# 重塑数据形状为(samples, maxlen)的二维整数张量
X_train = preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = preprocessing.sequence.pad_sequences(X_test, maxlen=maxlen)

########## Begin ##########
# 定义一个序列模型
model = Sequential()

# 添加一个Embedding层，标记个数 10000，维度 8，输入长度是maxlen
model.add(Embedding(10000, 8, input_length=maxlen))

# 添加一个Flatten层
model.add(Flatten())

# 添加一个全连接层，输出维度是1，激活函数‘sigmoid’，作为分类器
model.add(Dense(1, activation='sigmoid'))

# 编译模型，优化器选取‘rmsprop’，损失函数选取‘binary_crossentropy’，评估方式是‘acc’
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

# 拟合模型，epoch选取 10，batch_size选取 32，validation_split为 0.2
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# 打印模型结构
########## End ##########
