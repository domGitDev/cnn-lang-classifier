from keras.models import Model
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, Adamax, SGD
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate, Lambda
import tensorflow as tf

def create_model(input_length, num_class, vocab_size, embed_dim, num_filters=10, filter_sizes=(), drop=0.5):
    inputs = Input(shape=(input_length,), dtype='int32')
    embedding = Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=input_length)(inputs)
    reshape = Reshape((input_length, embed_dim, 1))(embedding)

    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embed_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embed_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embed_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

    maxpool_0 = MaxPool2D(pool_size=(input_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(input_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(input_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=num_class, activation='softmax')(dropout)

    # this creates a model that includes
    model = Model(inputs=inputs, outputs=output)

    checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
    #opt = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    opt = Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model, checkpoint


def create_model2(input_length, num_class, vocab_size, embed_dim, num_filters=128):
    inputs = Input(shape=(input_length,), dtype='int32')
    embedding = Embedding(input_dim=vocab_size+1, output_dim=embed_dim, input_length=input_length)(inputs)

    l_cov1= Conv1D(num_filters, 5, activation='relu')(embedding)
    l_pool1 = MaxPooling1D(5)(l_cov1)
    
    l_cov2 = Conv1D(num_filters, 5, activation='relu')(l_pool1)
    l_pool2 = MaxPooling1D(5)(l_cov2)

    l_cov3 = Conv1D(num_filters, 5, activation='relu')(l_pool2)
    l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling

    l_flat = Flatten()(l_pool3)
    l_dense = Dense(num_filters, activation='relu')(l_flat)
    output = Dense(len(num_class), activation='softmax')(l_dense)

    model = Model(inputs=inputs, outputs=output)

    checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', 
                                monitor='val_accuracy', verbose=1, 
                                save_best_only=True, mode='auto')

    return model, checkpoint

