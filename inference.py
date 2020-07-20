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
    parser.add_argument('--model_file', default='./logs/model.hdf5', 
                        type=str, help='path to trained model')
    parser.add_argument('--tokenizer_file', default='./logs/tokenizer.json', 
                        type=str, help='path to tokenizer used during training')

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
    with open('tokenizer.json') as f: 
        data = json.load(f) 
        tokenizer = tokenizer_from_json(data)
    
    model = load_model(args.model_file)
    model.summary()
    
    df_lang = data_helper.load_cvs_data(args.filename)

    # drop rows with null values
    df_lang = df_lang.dropna(how='any',axis=0) 
    print(df_lang.head())

    texts = df_lang['text'].map(str).values
    #np.random.shuffle(texts)
    encoded_texts = data_helper.pre_processing_predict(texts, tokenizer)

    predictions = model.predict(encoded_texts[:5])
    clasess = np.argmax(predictions, axis=1)

    print(clasess)


