import numpy as np
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Embedding, SimpleRNN, LSTM
from keras.datasets import imdb
from keras.preprocessing import sequence
import matplotlib
import matplotlib.pyplot as plt
from keras.layers import LSTM


matplotlib.use('Agg')
max_features = 10000
maxlen = 500
print("Loading data...")
# Begin
"""
加载imdb数据，数据路径为'/data/workspace/myshixun/imdb.npz'
"""
file_path = './data/imdb.npz'
with np.load(file_path, allow_pickle=True) as f:
    X_train, y_train = f['x_train'], f['y_train']
    X_test, y_test = f['x_test'], f['y_test']

X_train = preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = preprocessing.sequence.pad_sequences(X_test, maxlen=maxlen)
X_train = np.array([[min(word, max_features - 1) for word in sequence] for sequence in X_train])
X_test = np.array([[min(word, max_features - 1) for word in sequence] for sequence in X_test])
# END

print(len(X_train), "train sequences")
print(len(X_train), "test sequences")

print("Pad sequences (sample x times)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

model = Sequential()
model.add(Embedding(max_features, 32))
# BEGIN
"""
LSTM层，参数为32
"""

model.add(LSTM(32))

# END
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('./data/plot1.png')
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('./data/plot2.png')
# plt.show()

