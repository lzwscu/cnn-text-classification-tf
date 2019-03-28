#/usr/bin/env python
#-*- coding:utf-8 -*-

# 导入使用到的库
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional
from keras.utils.np_utils import to_categorical
from keras import initializers
from keras import backend as K
from keras.engine.topology import Layer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(title, label, test_size=0.1, random_state=42)

# 对类别变量进行编码，共10类
y_labels = list(y_train.value_counts().index)
le = preprocessing.LabelEncoder()
le.fit(y_labels)
num_labels = len(y_labels)
y_train = to_categorical(y_train.map(lambda x: le.transform([x])[0]), num_labels)

y_test = to_categorical(y_test.map(lambda x: le.transform([x])[0]), num_labels)

# 分词，构建单词-id词典
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=" ")
tokenizer.fit_on_texts(title)
vocab = tokenizer.word_index

# 将每个词用词典中的数值代替
X_train_word_ids = tokenizer.texts_to_sequences(X_train)
X_test_word_ids = tokenizer.texts_to_sequences(X_test)

# One-hot
x_train = tokenizer.sequences_to_matrix(X_train_word_ids, mode='binary')
x_test = tokenizer.sequences_to_matrix(X_test_word_ids, mode='binary')

# 序列模式
x_train = pad_sequences(X_train_word_ids, maxlen=200)
x_test = pad_sequences(X_test_word_ids, maxlen=200)