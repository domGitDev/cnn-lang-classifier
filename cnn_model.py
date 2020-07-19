import os
import sys
import tensorflow as tf
from keras.models import Model
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, Adamax, SGD
from keras.layers import Activation, BatchNormalization
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate, Lambda

from keras import backend as K
from keras.utils.generic_utils import get_custom_objects


class Swish(Activation):
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'


def swish(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'swish': Swish(swish)})


def create_model(input_length, num_class, vocab_size, embed_dim, 
        num_filters=10, filter_sizes=(), drop=0.5, log_dir='./logs'):
    
    inputs = Input(shape=(input_length,), dtype='int32')
    embedding = Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=input_length)(inputs)
    reshape = Reshape((input_length, embed_dim, 1))(embedding)

    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embed_dim), padding='same')(reshape)
    activ_0 = Activation('swish')(conv_0)
    batch_0 = BatchNormalization()(activ_0)
    conv_0_1 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embed_dim))(batch_0)
    activ_0_1 = Activation('swish')(conv_0_1)
    maxp_0 = MaxPool2D(pool_size=(input_length - filter_sizes[0]+1, 1))(activ_0_1)

    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embed_dim), padding='same')(reshape)
    activ_1 = Activation('swish')(conv_1)
    batch_1 = BatchNormalization()(activ_1)
    conv_1_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embed_dim))(batch_1)
    activ_1_1 = Activation('swish')(conv_1_1)
    maxp_1 = MaxPool2D(pool_size=(input_length - filter_sizes[1]+1, 1))(activ_1_1)

    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embed_dim), padding='same')(reshape)
    activ_2 = Activation('swish')(conv_2)
    batch_2 = BatchNormalization()(activ_2)
    conv_2_1 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embed_dim))(batch_2)
    activ_2_1 = Activation('swish')(conv_2_1)
    maxp_2 = MaxPool2D(pool_size=(input_length - filter_sizes[2]+1, 1))(activ_2_1)

    concat_tensor = Concatenate(axis=1)([maxp_0, maxp_1, maxp_2])
    flatten = Flatten()(concat_tensor)
    dense = Dense(units=32, activation='swish')(flatten)
    dropout = Dropout(drop)(dense)
    output = Dense(units=num_class, activation='softmax')(dropout)

    # this creates a model that includes
    model = Model(inputs=inputs, outputs=output)

    checkpoint = ModelCheckpoint(os.path.join(log_dir,
                    'weights.{epoch:03d}-{val_accuracy:.4f}.hdf5'), 
                    monitor='val_accuracy', verbose=1, 
                    save_best_only=True, mode='auto')

    #opt = Adamax(learning_rate=1e-3, beta_1=0.9, beta_2=0.999)
    opt = SGD(lr=1e-4, decay=1e-9, momentum=0.85, nesterov=False)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model, checkpoint

