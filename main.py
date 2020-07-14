import os
import sys
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf

import data_helper


if __name__ == '__main__':
    #tf.compat.v1.disable_eager_execution()

    parser = argparse.ArgumentParser(description='Order book analysis.')
    parser.add_argument('-f', '--filename', type=str, help='dataset file')
    parser.add_argument('--batch_size', default=64, type=int)

    args = parser.parse_args()

    if not os.path.exists(args.filename):
        raise ValueError('File not found: {0}.'.format(args.filename))

    df_lang = data_helper.load_cvs_data(args.filename)
    print('TOTAL SAMPLES', df_lang.text.size)

    uniq_labels = np.unique(df_lang['language'].values).tolist()
    df_lang['lang_code'] = df_lang['language'].map(lambda x: uniq_labels.index(x))

    train_data, test_data = data_helper.construct_dataset(
                            df_lang, 'text', 'lang_code', args.batch_size)

    data_iter = train_data.take(1)
    for d, l in data_iter:
        print(d, l)
    

