import os
import sys
import tensorflow as tf
from keras.models import Model
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, Adamax, SGD
from keras.layers import Activation, BatchNormalization
from keras.layers import Embedding, Conv2D, MaxPool2D
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, LSTM
from keras.layers import Reshape, Flatten, Dropout, Concatenate, Lambda


def create_model(input_length, num_class, vocab_size, embed_dim, 
        num_filters=10, filter_sizes=(), drop=0.5, log_dir='./logs'):
    
    inputs = Input(shape=(input_length,), dtype='int32')
    embedding = Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=input_length)(inputs)
    reshape = Reshape((input_length, embed_dim, 1))(embedding)

    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embed_dim), padding='same')(reshape)
    activ_0 = Activation('relu')(conv_0)
    batch_0 = BatchNormalization()(activ_0)
    conv_0_1 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embed_dim))(batch_0)
    activ_0_1 = Activation('relu')(conv_0_1)
    maxp_0 = MaxPool2D(pool_size=(input_length-filter_sizes[0]+1, 1))(activ_0_1)

    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embed_dim), padding='same')(reshape)
    activ_1 = Activation('relu')(conv_1)
    batch_1 = BatchNormalization()(activ_1)
    conv_1_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embed_dim))(batch_1)
    activ_1_1 = Activation('relu')(conv_1_1)
    maxp_1 = MaxPool2D(pool_size=(input_length-filter_sizes[1]+1, 1))(activ_1_1)

    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embed_dim), padding='same')(reshape)
    activ_2 = Activation('relu')(conv_2)
    batch_2 = BatchNormalization()(activ_2)
    conv_2_1 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embed_dim))(batch_2)
    activ_2_1 = Activation('relu')(conv_2_1)
    maxp_2 = MaxPool2D(pool_size=(input_length-filter_sizes[2]+1, 1))(activ_2_1)

    concat_tensor = Concatenate(axis=1)([maxp_0, maxp_1, maxp_2])
    flatten = Flatten()(concat_tensor)
    #dense = Dense(units=64, activation='relu')(flatten)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=num_class, activation='sigmoid')(dropout)

    # this creates a model that includes
    model = Model(inputs=inputs, outputs=output)

    #opt = Adamax(learning_rate=1e-4, beta_1=0.9, beta_2=0.999)
    opt = SGD(lr=1e-4, decay=1e-8, momentum=0.9, nesterov=True)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def create_model2(input_length, num_class, vocab_size, embed_dim, num_filters=64, filter_size=5, drop=0.5):

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=input_length))
    model.add(Dropout(drop))
    model.add(Conv1D(num_filters, filter_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(embed_dim, dropout=drop, recurrent_dropout=drop))
    model.add(Dense(num_class, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',    metrics=['accuracy'])

    return model

