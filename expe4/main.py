import numpy as np

# from keras.datasets import mnist
from keras.src.layers import Dense, Activation
from keras.src.utils import np_utils
from keras import models, Sequential
from keras import layers


# (X_train, y_train), (X_test, y_test) = mnist.load_data()
path = './data/mnist.npz'  # mnist数据集的文件路径
# --------------- Begin --------------- #
f = np.load(path)
X_train, y_train = f['x_train'], f['y_train']
X_test, y_test = f['x_test'], f['y_test']
f.close()
# --------------- End --------------- #

# 重塑数据集
# --------------- Begin --------------- #
X_train = X_train.reshape([60000, 784])
X_test = X_test.reshape([10000, 784])
X_train = X_train.astype(dtype='float32')
X_test = X_test.astype(dtype='float32')
# --------------- End --------------- #

# 归一化
# --------------- Begin --------------- #
X_train /= 255
X_test /= 255
# --------------- End --------------- #

# one-hot编码
# --------------- Begin --------------- #
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
# --------------- End --------------- #

model = Sequential()
model.add(Dense(10, input_shape=(784,)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
model.fit(X_train,y_train,epochs=200, batch_size=128, verbose=1, validation_split=0.2)

score = model.evaluate(X_test, y_test, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])


model2 = Sequential()
model2.add(Dense(128, input_shape=(784,)))
model2.add(Activation('softmax'))
model2.add(Dense(128))
model2.add(Activation('relu'))
model2.add(Dense(10))
model2.add(Activation('softmax'))

model2.summary()

model2.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
model2.fit(X_train, y_train, batch_size=128, epochs=20, verbose=1, validation_split=0.2)
model2.evaluate(X_test, y_test, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])