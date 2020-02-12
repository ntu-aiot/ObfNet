from __future__ import print_function
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import cv2
import click

import tensorflow as tf
import numpy as np
tf.random.set_seed(1)
np.random.seed(1)


def get_inference_cnn(input_shape, num_classes):
    model = Sequential()
    model.add(
        Conv2D(32,
               kernel_size=(3, 3),
               activation='relu',
               input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def get_inference_mlp(input_shape, num_classes):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def get_obfmodel_mlp(input_shape, num_neuron=100):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(num_neuron, activation='relu'))
    model.add(Dense(28 * 28, activation='relu'))
    model.add(Reshape(input_shape))
    return model


def get_obfmodel_cnn(input_shape, num_neuron=32):
    model = Sequential()
    model.add(
        Conv2D(32,
               kernel_size=(3, 3),
               activation='relu',
               input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(num_neuron, activation='relu'))
    model.add(Dense(28 * 28, activation='relu'))
    model.add(Reshape(input_shape))
    return model


def get_mid_out(model, layer_num, data):
    get_output = K.function([model.layers[0].input],
                            [model.layers[layer_num].output])
    return get_output([data])[0]


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
@click.option('--num-neuron',
              '-n',
              default=100,
              help='# of neurons at ObfNet.')
@click.option('--inf-path', default="", help='path to inference model')
@click.option('--save-image',
              '-S',
              default=False,
              help='flag for saving images.')
def main(is_inf, is_inf_cnn, is_obf_cnn, num_neuron, inf_path, save_image):

    os.makedirs("models/mnist/", exist_ok=True)

    batch_size = 128
    num_classes = 10
    epochs = 14
    # epochs = 28

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # cv2.imwrite('imgs/0.jpg', x_test[0])
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    if is_inf:
        if is_inf_cnn:
            inference_model = get_inference_cnn(input_shape, num_classes)
            inf_path = 'models/mnist/inf-cnn.h5'
        else:
            inference_model = get_inference_mlp(input_shape, num_classes)
            inf_path = 'models/mnist/inf-mlp.h5'

        inference_model.compile(loss='categorical_crossentropy',
                                optimizer='adam',
                                metrics=['accuracy'])

        inference_model.fit(x_train,
                            y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            callbacks=[
                                ModelCheckpoint(inf_path,
                                                monitor='val_accuracy',
                                                verbose=1,
                                                save_best_only=True,
                                                mode='max')
                            ],
                            verbose=1,
                            validation_data=(x_test, y_test))
    else:
        if not inf_path:
            raise ValueError(
                'Please input path to inference network or train the inference model.'
            )
        inference_model = load_model(inf_path)

    print(inference_model.summary())
    print('Inference Result: ')
    print(inference_model.evaluate(x_test, y_test, verbose=0))

    ######################
    # training of ObfNet #
    ######################

    if is_obf_cnn:
        obfmodel = get_obfmodel_cnn(input_shape, num_neuron)
    else:
        obfmodel = get_obfmodel_mlp(input_shape, num_neuron)
        obfmodel.build((None, 28, 28, 1))

    inference_model.trainable = False
    for l in inference_model.layers:
        l.trainable = False

    combined_model = Model(inputs=obfmodel.input,
                           outputs=inference_model(obfmodel.output))
    combined_model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    print(combined_model.summary())
    com_path = 'models/mnist/combined-model.h5'

    combined_model.fit(x=x_train,
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
                       validation_data=(x_test, y_test))

    ######################
    # testing of ObfNet #
    ######################

    score = combined_model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    if save_image:
        os.makedirs("imgs/mnist/", exist_ok=True)
        ind = [3, 2, 1, 18, 4, 8, 11, 0, 61, 7]
        h1 = []
        for i in ind:
            h1.append(x_test[i] * 255)
        v1 = np.concatenate(np.array(h1), axis=1)
        cv2.imwrite('imgs/mnist/input.jpg', v1)
        midout = get_mid_out(combined_model, 6, x_test)
        h = []
        for i in ind:
            h.append(midout[i] * 255)
        v = np.concatenate(np.array(h), axis=1)
        cv2.imwrite('imgs/mnist/obfresult.jpg', v)


if __name__ == "__main__":
    main()
