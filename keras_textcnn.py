#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Input, concatenate
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.utils.np_utils import to_categorical
import os
import tarfile
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd

# 有些数据是含有html标签的，需要去除
import re


MAX_LEN = 200

def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)

def charge_age(age):
    age = int(float(age))
    if age <= 25:
        return 1
    elif 25 < age <= 35:
        return 2
    elif 35 < age <= 45:
        return 3
    elif 45 < age:
        return 4
    else:
        return 0




def read_files():
    df = pd.read_csv("../data/user_age.file", sep="\t", names=["device_id", "cateid", "age"], index_col= False, header = None)
    df["new_age"] = df["age"].apply(lambda x: charge_age(x))
    df = df[df["new_age"] != 0]
    df["cateid"] = df["cateid"].apply(lambda x: " ".join(x.strip().split(",")[:500]))
    x = df["cateid"].values.tolist()
    y = df["new_age"].values.tolist()
    le = preprocessing.LabelEncoder()
    labels_list = list(df["new_age"].value_counts().index)
    le.fit(labels_list)
    num_labels = len(labels_list)

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=10)
    y_train = to_categorical([le.transform([x])[0] for x in train_y], num_labels)
    y_test = to_categorical([le.transform([x])[0] for x in test_y], num_labels)
    train_y, test_y = y_train, y_test
    return train_x, test_x, train_y, test_y, num_labels


def preprocess(train_texts, train_labels, test_texts, test_labels):
    tokenizer = Tokenizer(num_words=2000)  # 建立一个2000个单词的字典
    tokenizer.fit_on_texts(train_texts)
    # 对每一句影评文字转换为数字列表，使用每个词的编号进行编号
    x_train_seq = tokenizer.texts_to_sequences(train_texts)
    x_test_seq = tokenizer.texts_to_sequences(test_texts)
    x_train = sequence.pad_sequences(x_train_seq, maxlen=MAX_LEN)
    x_test = sequence.pad_sequences(x_test_seq, maxlen=MAX_LEN)
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)
    return x_train, y_train, x_test, y_test


def text_cnn(maxlen=MAX_LEN, max_features=2000, embed_size=32, num_labels=2):
    # Inputs
    comment_seq = Input(shape=[maxlen], name='x_seq')

    # Embeddings layers
    emb_comment = Embedding(max_features, embed_size)(comment_seq)

    # conv layers
    convs = []
    filter_sizes = [2, 3, 4, 5]
    for fsz in filter_sizes:
        l_conv = Conv1D(filters=100, kernel_size=fsz, activation='relu')(emb_comment)
        l_pool = MaxPooling1D(maxlen-fsz+1)(l_conv)
        l_pool = Flatten()(l_pool)
        convs.append(l_pool)
    merge = concatenate(convs, axis=1)

    out = Dropout(0.5)(merge)
    output = Dense(32, activation='relu')(out)

    output = Dense(units=num_labels, activation='softmax')(output)

    model = Model([comment_seq], output)
    #     adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    return model


if __name__ == '__main__':

    train_texts, test_texts, train_labels, test_labels, num_labels = read_files()
    x_train, y_train, x_test, y_test = preprocess(train_texts, train_labels, test_texts, test_labels)
    model = text_cnn(num_labels=num_labels)
    batch_size = 64
    epochs = 5
    model.fit(x_train, y_train,
              validation_split=0.1,
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True)
    scores = model.evaluate(x_test, y_test)
    print('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))
