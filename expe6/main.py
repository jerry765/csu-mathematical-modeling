import numpy as np
import string
from keras.preprocessing.text import Tokenizer

"""
单词级的 one-hot 编码
"""


def word_one_hot(samples):
    # 构建数据中所有标记的索引，用一个字典来存储
    token_index = {}
    for sample in samples:
        # Begin
        # 利用 split 方法对样本进行分词.
        for word in sample.split():
            if word not in token_index:
                # 每一个单词指定唯一索引
                token_index[word] = len(token_index) + 1
        # End

    # 只考虑样本前 max_length 个单词
    max_length = 10

    # 结果返回给 results:
    results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
    for i, sample in enumerate(samples):
        for j, word in enumerate(sample.split()[:max_length]):
            index = token_index.get(word)
            # 唯一的元素为 1
            results[i, j, index] = 1

    # Begin
    # 查看索引字典和样本列表的第二个元素的第二个单词的编码情况
    # print(token_index)
    print(results[1, 1])
    # End


"""
字符级的 one-hot 编码
"""


def char_one_hot(samples):
    # 可以打印的 ASCII 字符
    characters = string.printable

    # Begin
    # 创建索引字典
    token_index = dict(zip(characters, range(1, len(characters) + 1)))
    # 只考虑样本前 max_length 个字符
    max_length = 50

    # 构建合适的 results 数组，将结果返回给 results:
    results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
    # End

    for i, sample in enumerate(samples):
        for j, character in enumerate(sample[:max_length]):
            index = token_index.get(character)
            results[i, j, index] = 1

    # Begin
    # 查看索引字典和样本列表的第三个元素的第三个字符的编码情况
    # print(token_index)
    print(results[2, 2])
    # End


# Begin
# 给 samples 列表新增一个元素 'a panda is sleeping'
samples = ['The cat sat on the mat.', 'The dog ate my homework.', 'a panda is sleeping']

# End

word_one_hot(samples)

char_one_hot(samples)

"""
用 Keras 实现单词级的 one-hot 编码
"""
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)

word_index = tokenizer.word_index
print(word_index)
