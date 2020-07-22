import os
import sys
import ast
import json
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import seaborn as sns
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from keras.utils import to_categorical

import data_helper
from cnn_model import create_model, create_model2

ClASSES = ['afrikaans', 'english', 'nederlands']

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


def plot_test_prediction(test_iter):
    print('\nIter over test data')
    test_codex = []
    test_codey = []

    for x, y in test_iter:
        test_codex.append(x[0].numpy())
        test_codey.append(y.numpy()[0])

    print('Run test prediction...')
    predictions = model.predict(np.array(test_codex), verbose=1)
    pred_indexs = np.argmax(predictions, axis=1)
    true_indexs = np.argmax(test_codey, axis=1)

    true_labels = [ClASSES[i] for i in true_indexs]
    pred_labels = [ClASSES[i] for i in pred_indexs]

    test_codex = [[c for c in codes if c != 0] for codes in test_codex]
    test_df = pd.DataFrame({'language':true_labels, 'pred_language': pred_labels})

    # construct how many miss prediction
    gb = test_df.groupby('language')
    dfs = [(x, gb.get_group(x).reset_index()) for x in gb.groups]
    
    # plot correct and miss prediction per classes
    fig, axs = plt.subplots(1, len(dfs), figsize = (18, 30))
    colors = list(mcolors.TABLEAU_COLORS.values())

    for i, (lang, df) in enumerate(dfs):
        print(df.head())
        df['pred_language'].value_counts().plot.bar(ax=axs[i], color=colors[i])
        axs[i].set_title(lang)
        axs[i].set_xticks(range(3))
        i += 1

    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    #tf.compat.v1.disable_eager_execution()

    parser = argparse.ArgumentParser(description='Train language classifier.')
    parser.add_argument('-f', '--filename', type=str, help='dataset file')
    parser.add_argument('--num_class', default=3, type=int)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--test_perc', default=0.25, type=float)
    parser.add_argument('--valid_perc', default=0.35, type=float)
    parser.add_argument('--drop', default=0.1, type=float)
    parser.add_argument('--embed_dim', default=128, type=int)
    parser.add_argument('--num_filters', default=64, type=int)
    parser.add_argument('--filter_sizes', default='[2,3,4,5]', type=str,
                        help='convolution over 2, 3 and 5 words: [2,3,5]')
    parser.add_argument('--shuffle', default=True, type=bool)
    parser.add_argument('--seed', default=231, type=int)
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

    # remove sentence with duplicate labels
    groups = df_lang.groupby('text').size().reset_index(name='counts')
    duplicated = groups.loc[groups['counts'] > 1]
    for sentence in duplicated['text']:
        indexes = df_lang.loc[df_lang['text'] == sentence].index
        df_lang.drop(indexes, inplace = True)
    
    uniq_labels = np.unique(df_lang['language'].values).tolist()
    uniq_labels = list(sorted(uniq_labels))
    df_lang['lang_code'] = df_lang['language'].map(lambda x: ClASSES.index(str(x).lower()))
    print('Num Rows', df_lang['lang_code'].size)

    fig, axs = plt.subplots(1, 2, figsize = (18, 30))

    # plot imbalanced dataset
    g = sns.countplot(df_lang['lang_code'], ax=axs[0])
    g.set_title('Original')
    g.set_xticklabels(uniq_labels)

    # oversample dataset to balance each class
    max_size = df_lang['lang_code'].value_counts().max()
    lst = [df_lang]
    for class_index, group in df_lang.groupby('language'):
        lst.append(group.sample(max_size-len(group), random_state=1, replace=True))
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
    texts = df_lang['text'].map(str).values
    labels = df_lang['lang_code'].map(int).values

    results = data_helper.pre_processing(texts)
    tokenizer, code_x, vocab_size, max_length = results

    code_y = to_categorical(labels)
    
    # convert numpy array into tensorflow Dataset 
    # split dataset in train, validation and test
    results = data_helper.construct_dataset(
                        code_x, code_y, args.batch_size, args.test_perc, 
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

    #model = create_model2(max_length, args.num_class, vocab_size, args.embed_dim, 0.2)

    model.summary()
    plot_model(model, to_file=args.log_dir + '/cnn_lang.png', show_shapes=True)

    # train CNN model
    history = model.fit_generator(train_iter,
                    epochs=args.epochs,
                    steps_per_epoch=train_steps,
                    validation_data=valid_iter,
                    validation_steps=valid_steps,
                    shuffle=False,
                    verbose=1)

    # save trained model to file
    model_file = os.path.join(args.log_dir, 'model.h5')
    model.save(model_file)

    tokenizer_json = tokenizer.to_json()
    tokenizer_file = os.path.join(args.log_dir, 'tokenizer.json') 
    with open(tokenizer_file, 'w', encoding='utf-8') as f:  
          f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    # log to console train and test accuracy
    train_iter = iter(train_data)
    loss, accuracy = model.evaluate_generator(valid_iter, steps=valid_steps, verbose=False)
    print('Validation Accuracy: {:.4f}'.format(accuracy))
    loss, accuracy = model.evaluate_generator(test_iter, steps=test_steps, verbose=False)
    print('Testing Accuracy:  {:.4f}'.format(accuracy))

    # plot train vs validation accuracy
    plot_history(history)

    # output test prediction to csv file
    test_iter = iter(test_data)
    plot_test_prediction(test_iter)
