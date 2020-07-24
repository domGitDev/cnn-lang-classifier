import os
import sys
import ast
import json
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.models import Sequential
from keras.preprocessing.text import tokenizer_from_json

import data_helper

ClASSES = ['Afrikaans', 'English', 'Nederlands']

if __name__ == '__main__':
    #tf.compat.v1.disable_eager_execution()

    parser = argparse.ArgumentParser(description='Train language classifier.')
    parser.add_argument('-f', '--filename', type=str, help='dataset file')
    parser.add_argument('--model_file', default='./logs/model.h5', 
                        type=str, help='path to trained model')
    parser.add_argument('--tokenizer_file', default='./logs/tokenizer.json', 
                        type=str, help='path to tokenizer used during training')
    parser.add_argument('--st_index', default=5, type=int,
                        help='start dataset selection range')
    parser.add_argument('--ed_index', default=20, type=int,
                        help='end dataset selection range')

    args = parser.parse_args()
    
    config = tf.compat.v1.ConfigProto(device_count = {'GPU': 1})
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    if not os.path.exists(args.filename):
        raise ValueError('File not found: {0}.'.format(args.filename))

    if not os.path.exists(args.model_file):
        raise ValueError('File not found: {0}.'.format(args.model_file))

    # load tokenizer object
    tokenizer = None
    with open(args.tokenizer_file) as f: 
        data = json.load(f) 
        tokenizer = tokenizer_from_json(data)
        print('Loaded tokenizer')
    
    model = load_model(args.model_file)
    model.summary()
    
    df_lang = data_helper.load_cvs_data(args.filename)

    # drop rows with null values
    df_lang = df_lang.dropna(how='any',axis=0) 

    # select dataset by index
    st = args.st_index
    ed = args.ed_index
    if ed == -1:
        ed = df_lang['text'].size
    
    texts = df_lang['text'].map(str).values
    np.random.shuffle(texts)
    tokenizer, encoded_texts, _, _ = data_helper.pre_processing(texts[st:ed], tokenizer)

    predictions = model.predict(encoded_texts)
    labels = np.argmax(predictions, axis=1)

    print('True Label\tPred Label\t\ttext')
    for text, label, index in zip(df_lang['text'][st:ed], df_lang['language'][st:ed], labels):
        print('{0:10s}\t{1:10s}\t\t{2}'.format(label, ClASSES[index], text))


