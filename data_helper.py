import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

tokenizer = None
MAX_WORDS = 100000
MAX_LENGTH = 1000


def load_cvs_data(filename):
    if not os.path.exists(filename):
        return None
    return pd.read_csv(filename, sep=',',
            header=None, skiprows=1, 
            names=['text', 'language'],
            dtype={'text': str, 'language': str})


def data_by_column(df, col_name, col_value):
    if not (col_name in df.columns):
        return None
    return df.loc[df[col_name] == col_value]


def construct_dataset(df, value_col, label_col, batch_size, test_split=0.15, seed=49):
    global tokenizer, MAX_LENGTH

    total = df[value_col].size
    texts = df[value_col].map(str).values
    labels = df[label_col].map(int).values
    
    print(texts[:10])
    print(labels[:10])

    # get size of longest sentence
    MAX_LENGTH = int(df[value_col].str.len().max())
    tokenizer = Tokenizer(num_words=MAX_WORDS)

    # convert text to sequence
    tokenizer.fit_on_texts(texts)
    sequences =  tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print("unique words : {}".format(len(word_index)))

    # pad all sentences to same length
    sequences = pad_sequences(sequences, maxlen=MAX_LENGTH+5)

    # convert labels to categorial
    labels = to_categorical(labels)

    train_size = total
    if test_split:
        test_size = int(total * 0.3)
        test_size -= (test_size % batch_size)
        train_size = total - test_size

    if batch_size > 0:
        train_size -= (train_size % batch_size)

    dataset = tf.data.Dataset.from_tensor_slices((sequences, labels))
    dataset = dataset.shuffle(total, seed=seed)

    train_data = dataset.take(train_size)

    if test_size > 0:
        test_data = dataset.skip(train_size).take(test_size)

    return train_data, test_data
