1.	import tensorflow as tf
import numpy as np
import math
import h5py
import keras
from keras.utils import np_utils
from keras import models

from keras.layers import InputLayer, Input, Reshape, MaxPooling2D, Conv2D, Dense, Flatten
from keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from keras import backend as K


# 载入数据
path = 'mnist.npz'
f = np.load(path)
x_train, y_train = f['x_train'],f['y_train']
x_test, y_test = f['x_test'],f['y_test']
f.close()

# 配置神经网络
img_size = 28
img_size_flat = 28 * 28
img_shape = (28, 28)
img_shape_full = (28, 28, 1)
num_classes = 10
num_channels = 1


# print(img_size)
# print(img_size_flat)
# print(img_shape)
# print(img_shape_full)
# print(num_classes)
# print(num_channels)

def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # 画图
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        # 显示真正的预测的类别
        if cls_pred is None:
            xlabel = "True:{0}".format(cls_true[i])
        else:
            xlabel = "True:{0}, Pred:{1}".format(cls_true[i], cls_pred[i])
        # 将类别作为x轴的标签
        ax.set_xlabel(xlabel)
        # 去除图中的刻度线
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def plot_example_errors(cls_pred, correct):
    incorrect = (correct == False)
    images = x_test[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = y_test[incorrect]
    plot_images(images=images[0:9], cls_true=cls_true[0:9], cls_pred=cls_pred[0:9])


model = models.Sequential()
model.add(InputLayer(input_shape=(img_size_flat,)))
model.add(Reshape(img_shape_full))
model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same', activation='relu', name='layer_conv1'))
model.add(MaxPooling2D(pool_size=2, strides=2))

model.add(Conv2D(kernel_size=5, strides=1, filters=36, padding='same', activation='relu', name='layer_conv2'))
model.add(MaxPooling2D(pool_size=2, strides=2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# 编译模型
model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
# 训练模型
x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000, 784)
#归一化
x_train = x_train / 255
x_test = x_test / 255
y_train = np_utils.to_categorical(y_train, 10)
model.fit(x_train, y_train, epochs=1, batch_size=128, validation_split=1 / 12, verbose=2)
# 评估与性能指标
y_test_f = np_utils.to_categorical(y_test, 10)
result = model.evaluate(x_test, y_test_f, verbose=1)
print('loss', result[0])
print('acc', result[1])
# 预测
predict = model.predict(x_test)
predict = np.argmax(predict, axis=1)
plot_images(x_test[0:9], y_test[0:9], predict[0:9])
# 错分类的图片
y_pred = model.predict(x_test)
cls_pred = np.argmax(y_pred, axis=1)
correct = (cls_pred == y_test)
plot_example_errors(cls_pred, correct=correct)

# 功能模型
inputs = Input(shape=(img_size_flat,))
net = inputs
net = Reshape(img_shape_full)(net)
net = Conv2D(kernel_size=5, strides=1, filters=16, padding='same', activation='relu', name='layer_conv1')(net)
net = MaxPooling2D(pool_size=2, strides=2)(net)

net = Conv2D(kernel_size=5, strides=1, filters=36, padding='same', activation='relu', name='layer_conv2')(net)
net = MaxPooling2D(pool_size=2, strides=2)(net)

net = Flatten()(net)
net = Dense(128, activation='relu')(net)
net = Dense(num_classes, activation='softmax')(net)
outputs = net

# 模型编译
model2 = models.Model(inputs=inputs, outputs=outputs)
model2.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# 训练
model2.fit(x_train, y_train, batch_size=128, epochs=1, validation_split=1 / 12, verbose=2)
y_test_h = np_utils.to_categorical(y_test, 10)
result = model2.evaluate(x_test, y_test_h, verbose=1)
print(model2.metrics_names[0], result[0])
print(model2.metrics_names[1], result[1])
# 预测
predict = model2.predict(x_test)
predict = np.argmax(predict, axis=1)
plot_images(x_test[0:9], y_test[0:9], predict[0:9])
# 错分类图片
y_pred = model2.predict(x_test)
cls_pred = np.argmax(y_pred, axis=1)
correct = (cls_pred == y_test)
plot_example_errors(cls_pred, correct=correct)

# 保存模型
path_model = "./model2.pkl"
model2.save(path_model)
#载入模型
del model2
model3 = models.load_model(path_model)

# 模型3预测
predict = model3.predict(x_test)
predict = np.argmax(predict, axis=1)
plot_images(x_test[0:9], y_test[0:9], predict[0:9])


# 卷积权重的辅助函数
def plot_conv_weights(weights, input_channel=0):
    w_min = np.min(weights)
    w_max = np.max(weights)
    num_filters = weights.shape[3]
    # 卷积核的平方根.
    num_grids = math.ceil(math.sqrt(num_filters))
    # 创建带有网格子图的图像.
    fig, axes = plt.subplots(num_grids, num_grids)
    #  绘制所有卷积核的权重
    for i, ax in enumerate(axes.flat):
        #  仅绘制有限的卷积核权重
        if i < num_filters:
            img = weights[:, :, input_channel, i]
            # 画图
            ax.imshow(img, vmin=w_min, vmax=w_max,interpolation='nearest', cmap='seismic')
        # 去除刻度线.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


# 得到层
model3.summary()
layer_input = model3.layers[0]
layer_conv1 = model3.layers[2]
layer_conv2 = model3.layers[4]

# 卷积权重
weights_conv1 = layer_conv1.get_weights()[0]
plot_conv_weights(weights=weights_conv1, input_channel=0)
weights_conv2 = layer_conv2.get_weights()[0]
plot_conv_weights(weights=weights_conv2, input_channel=0)


# 卷积层输出的帮助函数
def plot_conv_output(values):
    num_filters = values.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))
    # 创建带有网格子图的图像
    fig, axes = plt.subplots(num_grids, num_grids)
    # 画出所有卷积核的输出图像
    for i, ax in enumerate(axes.flat):
        # 仅画出有效卷积核图像
        if i < num_filters:
            # 获取第i个卷积核的输出图像
            img = values[0, :, :, i]
            # 画图e.
            ax.imshow(img, interpolation='nearest', cmap='binary')
        # 移除刻度线
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


# 输入图像
def plot_image(image):
    plt.imshow(image.reshape(img_shape), interpolation='nearest', cmap='binary')
    plt.show()


image1 = x_test[0]
plot_image(image1)

# 卷积层输出一
output_conv1 = K.function(inputs=[layer_input.input], outputs=[layer_conv1.output])
layer_output1 = output_conv1(np.array([image1]))[0]
print(layer_output1.shape)
plot_conv_output(values=layer_output1)

# 卷积层输出二
output_conv2 = models.Model(inputs=layer_input.input, outputs=layer_conv2.output)
layer_output2 = output_conv2.predict(np.array([image1]))
print(layer_output2.shape)
plot_conv_output(values=layer_output2)
