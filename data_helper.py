import os
import re
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences


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


def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def build_vocabulary(sentences):
    word_counts = Counter(itertools.chain(*sentences))
    vocab_inv = [x[0] for x in word_counts.most_common()]
    vocab_inv = list(sorted(vocab_inv))
    vocab = {x: i for i, x in enumerate(vocab_inv)}
    return [vocab, vocab_inv]


def pre_processing(df, val_col, label_col):
    texts = df[val_col].map(str).values
    labels = df[label_col].map(int).values
    
    # build vocabulary list
    texts = [clean_str(s) for s in texts]
    _, vocab_inv = build_vocabulary(texts)
    vocab_size = len(vocab_inv)

    # encode sentences as int per words
    text_codes = [one_hot(text, vocab_size) for text in texts]
    MAX_LENGTH = max(len(val) for val in text_codes)

    code_x = pad_sequences(text_codes, maxlen=MAX_LENGTH, padding='post', value=0.0)
    code_y = to_categorical(labels)

    return code_x, code_y, vocab_inv, MAX_LENGTH


def construct_dataset(x, y, batch_size, test_split=0, valid_split=0, seed=434, shuffle=False):

    # split data into train, validation and test
    total = len(x)
    test_size = 0
    valid_size = 0
    train_size = total
    if test_split > 0:
        test_size = int(total * test_split)
        if batch_size > 0:
            test_size -= (test_size % batch_size)
        train_size = train_size - test_size

    if valid_split > 0:
        valid_size = int(total * valid_split)
        if batch_size > 0:
            valid_size -= (valid_size % batch_size)
        train_size = train_size - valid_size

    if batch_size > 0:
        train_size -= (train_size % batch_size)

    # create tensorflow dataset
    def _expand_function(item, label):
        item = tf.expand_dims(item, 0)
        label = tf.expand_dims(label, 0)
        return item, label

    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        dataset = dataset.shuffle(total, seed=seed)
    dataset = dataset.map(_expand_function)

    test_data = None
    valid_data = None

    # split dataset variable into train, validation, test
    if test_size > 0:
        test_data = dataset.take(test_size)
        dataset = dataset.skip(test_size)
    
    if valid_split > 0:
        valid_data = dataset.take(valid_size)
        dataset = dataset.skip(valid_size)

    train_data = dataset.take(train_size)

    train_data = train_data.repeat()
    valid_data = valid_data.repeat()
    test_data = test_data.repeat()

    sizes = [train_size, test_size, valid_size]

    return train_data, test_data, valid_data, sizes

