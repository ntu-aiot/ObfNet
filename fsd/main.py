from __future__ import print_function
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, Conv2D, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import os
import cv2
import click
import librosa

import tensorflow as tf
import numpy as np
tf.random.set_seed(1)
np.random.seed(1)


def wav2mfcc(file_path, max_pad_len=45):
    wave, sr = librosa.load(file_path, mono=True)
    #print(sr)#22050
    mfcc = librosa.feature.mfcc(wave)
    if mfcc.shape[1] <= 45:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc,
                      pad_width=((0, 0), (0, pad_width)),
                      mode='constant')
    else:
        mfcc = mfcc[:, 0:45]  #[0,45)

    return mfcc


def get_data():
    """get the wav data in and convert them to pictures with labels."""

    labels = []
    mfccs = []

    for f in os.listdir('dataset/fsd/'):
        if f.endswith('.wav'):
            # MFCC
            mfccs.append(wav2mfcc('dataset/fsd/' + f))

            # List of labels
            label = f.split('_')[0]
            labels.append(label)

    return np.asarray(mfccs), to_categorical(labels)


def get_inference_cnn(input_shape, num_classes):
    model = Sequential()
    model.add(
        Conv2D(32,
               kernel_size=(2, 2),
               activation='relu',
               input_shape=(20, 45, 1)))
    model.add(Conv2D(48, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 6), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dropout(0.1))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def get_inference_mlp(input_shape, num_classes):
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(800))
    model.add(Dropout(0.15))
    model.add(Activation('relu'))
    model.add(Dense(300))
    model.add(Dropout(0.15))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Dropout(0.1))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def get_obfmodel_cnn():
    model = Sequential()
    model.add(
        Conv2D(3,
               kernel_size=(2, 4),
               activation='relu',
               input_shape=(20, 45, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(5, kernel_size=(3, 6), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(900, W_regularizer=regularizers.l2(0.1)))
    model.add(Dropout(0.15))
    model.add(Activation('relu'))
    model.add(Reshape((20, 45, 1)))
    return model


def get_obfmodel_mlp():
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(200))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(900))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Reshape((20, 45, 1)))
    return model


@click.command()
@click.option(
    '--is-inf/--no-inf',
    '-T',
    default=True,
    help=
    "flag for whether training inference network, default set to False (trian ObfNet only)"
)
@click.option('--is-inf-cnn/--is-inf-mlp',
              '-IC/-IM',
              default=True,
              help="flag for training Convolutional Inference Network.")
@click.option('--is-obf-cnn/--is-obf-mlp',
              '-OC/-OM',
              default=False,
              help="flag for training Convolutional Obfuscation Network.")
@click.option('--inf-path', default="", help='path to inference model')
@click.option('--save-sound',
              '-S',
              default=False,
              help='flag for saving sound.')
def main(is_inf, is_inf_cnn, is_obf_cnn, inf_path, save_sound):

    os.makedirs("models/fsd/", exist_ok=True)

    batch_size = 64
    num_classes = 10
    epochs = 200

    mfccs, labels = get_data()

    dim_1 = mfccs.shape[1]
    dim_2 = mfccs.shape[2]
    channels = 1

    X = mfccs
    X = X.reshape((mfccs.shape[0], dim_1, dim_2, channels))
    y = labels

    input_shape = (dim_1, dim_2, channels)


    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.1,
                                                        random_state=1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    if is_inf:
        if is_inf_cnn:
            inference_model = get_inference_cnn(input_shape, num_classes)
            inf_path = 'models/fsd/inf-cnn.h5'
        else:
            inference_model = get_inference_mlp(input_shape, num_classes)
            inf_path = 'models/fsd/inf-mlp.h5'

        inference_model.compile(loss='categorical_crossentropy',
                                optimizer='adadelta',
                                metrics=['accuracy'])

        inference_model.fit(X_train,
                            y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            callbacks=[
                                ModelCheckpoint(inf_path,
                                                monitor='val_loss',
                                                save_best_only=True)
                            ],
                            verbose=1,
                            validation_split=0.1)
    else:
        if not inf_path:
            raise ValueError(
                'Please input path to inference network or train the inference model.'
            )
        inference_model = load_model(inf_path)

    print(inference_model.summary())
    print('Inference Result: ')
    print(inference_model.evaluate(X_test, y_test, verbose=0))

    ######################
    # training of ObfNet #
    ######################

    if is_obf_cnn:
        obfmodel = get_obfmodel_cnn()
    else:
        obfmodel = get_obfmodel_mlp()
    obfmodel.build((None, 20, 45, 1))

    inference_model.trainable = False
    for l in inference_model.layers:
        l.trainable = False

    combined_model = Model(inputs=obfmodel.input,
                           outputs=inference_model(obfmodel.output))
    combined_model.compile(optimizer='adadelta',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    print(combined_model.summary())
    com_path = 'models/fsd/combined-model.h5'

    combined_model.fit(x=X_train,
                       y=y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       callbacks=[
                           ModelCheckpoint(com_path,
                                           monitor='val_accuracy',
                                           verbose=1,
                                           save_best_only=True,
                                           mode='max')
                       ],
                       verbose=1,
                       validation_data=(X_test, y_test))

    ######################
    # testing of ObfNet #
    ######################

    score = combined_model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == "__main__":
    main()