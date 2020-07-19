import os
import sys
import ast
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import seaborn as sns
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf

import data_helper
from cnn_model import create_model


def plot_history(history):# Plot the loss and accuracy
    # Format the train history
    history_df = pd.DataFrame(history.history, columns=history.history.keys())

    # Plot the accuracy
    fig = plt.figure()
    fig.set_size_inches(18.5, 10)
    ax = plt.subplot(211)
    ax.plot(history_df["accuracy"], label="accuracy")
    ax.plot(history_df["val_accuracy"], label="val_accuracy")
    ax.legend()
    plt.title('Score during training.')
    plt.xlabel('Training step')
    plt.ylabel('Accuracy')
    plt.grid(b=True, which='major', axis='both')
    
    # Plot the loss
    ax = plt.subplot(212)
    ax.plot(history_df["loss"], label="loss")
    ax.plot(history_df["val_loss"], label="val_loss")
    ax.legend()
    plt.title('Loss during training.')
    plt.xlabel('Training step')
    plt.ylabel('Loss')
    plt.grid(b=True, which='major', axis='both')
    
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    #tf.compat.v1.disable_eager_execution()

    parser = argparse.ArgumentParser(description='Train language classifier.')
    parser.add_argument('-f', '--filename', type=str, help='dataset file')
    parser.add_argument('--num_class', default=3, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--test_perc', default=0.1, type=float)
    parser.add_argument('--valid_perc', default=0.3, type=float)
    parser.add_argument('--drop', default=0.1, type=float)
    parser.add_argument('--embed_dim', default=512, type=int)
    parser.add_argument('--num_filters', default=16, type=int)
    parser.add_argument('--filter_sizes', default='[3,4,5]', type=str,
                        help='convolution over 3, 4 and 5 words: [3,4,5]')
    parser.add_argument('--shuffle', default=True, type=bool)
    parser.add_argument('--seed', default=411, type=int)
    parser.add_argument('--log_dir', default='./logs', 
                        type=str, help='log directory path')

    args = parser.parse_args()
    filter_sizes = ast.literal_eval(args.filter_sizes)

    config = tf.compat.v1.ConfigProto(device_count = {'GPU': 1})
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    if not os.path.exists(args.filename):
        raise ValueError('File not found: {0}.'.format(args.filename))

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    df_lang = data_helper.load_cvs_data(args.filename)
    print(df_lang.head())
    print(df_lang[df_lang.isnull().any(axis=1)])

    # drop rows with null values
    df_lang = df_lang.dropna(how='any',axis=0) 
    print(df_lang.isnull().any().describe(include='all'))

    uniq_labels = np.unique(df_lang['language'].values).tolist()
    uniq_labels = list(sorted(uniq_labels))
    df_lang['lang_code'] = df_lang['language'].map(lambda x: uniq_labels.index(x))

    fig, axs = plt.subplots(1, 2, figsize = (18, 30))

    # plot imbalanced dataset
    g = sns.countplot(df_lang['lang_code'], ax=axs[0])
    g.set_title('Original')
    g.set_xticklabels(uniq_labels)

    # oversample dataset to balance each class
    max_size = df_lang['lang_code'].value_counts().max()
    lst = [df_lang]
    for class_index, group in df_lang.groupby('lang_code'):
        lst.append(group.sample(max_size-len(group), replace=True))
    df_lang = pd.concat(lst)

    # plot balanced dataset
    g = sns.countplot(df_lang['lang_code'], ax=axs[1])
    g.set_title('Balanced')
    g.set_xticklabels(uniq_labels)

    print([(i,label) for i,label in enumerate(uniq_labels)])
    print(df_lang['language'].value_counts())

    # display figure
    plt.show()
    plt.close(fig)

    # convert texts and labels into codes
    code_x, code_y, vocab_inv, max_length = data_helper.pre_processing(df_lang, 'text', 'lang_code')
    vocab_size = len(vocab_inv)

    # convert numpy array into tensorflow Dataset 
    # split dataset in train, validation and test
    results = data_helper.construct_dataset(code_x, code_y, args.batch_size, args.test_perc, 
                                            args.valid_perc, args.seed, args.shuffle)

    # unpack results 
    train_data, test_data, valid_data, shape = results

    # create iterator instance of datasets
    train_iter = iter(train_data)
    valid_iter = iter(valid_data)
    test_iter = iter(test_data)
    
    # compute steps per epoch
    train_steps = shape[0] // args.batch_size
    test_steps = shape[1] // args.batch_size
    valid_steps = shape[2] // args.batch_size

    # create instanciate of CNN models
    model = create_model(max_length, args.num_class, vocab_size, args.embed_dim, 
                                    args.num_filters, filter_sizes, args.drop, args.log_dir)

    model.summary()

    # train CNN model
    history = model.fit_generator(train_iter,
                    epochs=args.epochs, 
                    steps_per_epoch=train_steps,
                    validation_data=valid_iter,
                    validation_steps=valid_steps,
                    verbose=1)

    out_file = os.path.join(args.log_dir, 'model.hdf5')
    model.save(out_file)

    # log to console train and test accuracy
    train_iter = iter(train_data)
    loss, accuracy = model.evaluate_generator(valid_iter, steps=valid_steps, verbose=False)
    print('Validation Accuracy: {:.4f}'.format(accuracy))
    loss, accuracy = model.evaluate_generator(test_iter, steps=test_steps, verbose=False)
    print('Testing Accuracy:  {:.4f}'.format(accuracy))

    # plot train vs validation accuracy
    plot_history(history)
