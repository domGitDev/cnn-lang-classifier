import os
import sys
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import argparse
import pandas as pd
import numpy as np
import tensorflow as tf

import data_helper
from cnn_model import create_model, create_model2


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

    parser = argparse.ArgumentParser(description='Order book analysis.')
    parser.add_argument('-f', '--filename', type=str, help='dataset file')
    parser.add_argument('--num_class', default=3, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=64, type=int)

    args = parser.parse_args()

    config = tf.compat.v1.ConfigProto(device_count = {'GPU': 1})
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    if not os.path.exists(args.filename):
        raise ValueError('File not found: {0}.'.format(args.filename))

    df_lang = data_helper.load_cvs_data(args.filename)
    print('TOTAL SAMPLES', df_lang.text.size)

    uniq_labels = np.unique(df_lang['language'].values).tolist()
    uniq_labels = list(sorted(uniq_labels))
    df_lang['lang_code'] = df_lang['language'].map(lambda x: uniq_labels.index(x))

    train_data, test_data, valid_data, shape, vocab_inv = data_helper.construct_dataset(
                                                                    df_lang, 'text', 'lang_code', 
                                                                    args.batch_size, 0.2, 0.3)

    train_iter = iter(train_data)
    valid_iter = iter(valid_data)
    test_iter = iter(test_data)

    drop = 0.1
    embed_dim = 20
    filter_sizes = [3,4,5]
    num_filters = 64

    max_length = shape[3]
    vocab_size = len(vocab_inv)
    num_class = df_lang['lang_code'].size
    
    
    train_steps = shape[0] // args.batch_size
    test_steps = shape[1] // args.batch_size
    valid_steps = shape[2] // args.batch_size

    model, checkpoint = create_model(max_length, args.num_class, vocab_size, 
                                    embed_dim, num_filters, filter_sizes, drop)

    #model, checkpoint = create_model2(max_length, args.num_class, vocab_size, 
    #                                embed_dim, num_filters=128)

    model.summary()
    history = model.fit_generator(train_iter,
                    epochs=args.epochs, 
                    steps_per_epoch=train_steps,
                    validation_data=valid_iter,
                    validation_steps=valid_steps,
                    #callbacks=[checkpoint],
                    verbose=1)

    #'''
    train_iter = iter(train_data)
    loss, accuracy = model.evaluate_generator(valid_iter, steps=valid_steps, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate_generator(test_iter, steps=test_steps, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    plot_history(history)
    #'''