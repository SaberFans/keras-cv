'''Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

import argparse
import h5py
from datasets.tiny_imagenet import *

parent_output_dir = 'hdf5'
output_path1 = 'hdf5/tiny-imagenet_train.h5'
output_path2 = 'hdf5/tiny-imagenet_val.h5'


def get_data(data_dir, hdf5):
    """This function loads in the data, either by loading images on the fly or by creating and
    loading from a hdf5 database.

    Args:
        data_dir: Root directory of the datasets.
        hdf5: Boolean. If true, (create and) load data from a hdf5 database.

    Returns:
        X: training images.
        Y: training labels.
        X_test: validation images.
        Y_test: validation labels."""

    # Get the filenames of the lists containing image paths and labels.
    train_file, val_file = build_dataset_index(data_dir)

    # Check if (creating and) loading from hdf5 database is desired.
    if hdf5:
        # Create folder to store datasets.
        if not os.path.exists(parent_output_dir):
            os.makedirs(parent_output_dir)
        # Check if hdf5 databases already exist and create them if not.
        if not os.path.exists(output_path1):
            from tflearn.data_utils import build_hdf5_image_dataset

            print("Creating hdf5 train datasets.")
            build_hdf5_image_dataset(train_file, image_shape=(256, 256), mode='file',
                                     output_path=output_path1, categorical_labels=True, normalize=True)

        if not os.path.exists(output_path2):
            from tflearn.data_utils import build_hdf5_image_dataset
            print("Creating hdf5 val datasets.")
            build_hdf5_image_dataset(val_file, image_shape=(256, 256), mode='file',
                                     output_path=output_path2, categorical_labels=True, normalize=True)

        # Load training data from hdf5 datasets.
        h5f = h5py.File(output_path1, 'r')
        X = h5f['X']
        Y = h5f['Y']

        # Load validation data.
        h5f = h5py.File(output_path2, 'r')
        X_test = h5f['X']
        Y_test = h5f['Y']

        # Load images directly from disk when they are required.
    else:
        from tflearn.data_utils import image_preloader
        X, Y = image_preloader(train_file, image_shape=(64, 64), mode='file', categorical_labels=True, normalize=True,
                               filter_channel=True)
        X_test, Y_test = image_preloader(val_file, image_shape=(64, 64), mode='file', categorical_labels=True,
                                         normalize=True, filter_channel=True)

    # Randomly shuffle the datasets.
    # X, Y = shuffle(X, Y)

    return X, Y, X_test, Y_test


def main(data_dir, hdf5):
    batch_size = 32
    num_classes = 200
    # epochs = 100
    epochs = 10
    data_augmentation = True
    num_predictions = 20
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'keras_cifar10_trained_model.h5'

    # The data, shuffled and split between train and test sets:

    x_train, y_train, x_test, y_test = get_data(data_dir, hdf5)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the datasets
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the datasets
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            workers=4)

    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])


if __name__ == '__main__':
    # Parse arguments and create output directories.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='data/tiny-imagenet-200',
                        help='Directory in which the input data is stored.')
    parser.add_argument('--hdf5',
                        help='Set if hdf5 database should be created.',
                        action='store_true')
    parser.add_argument('--name', type=str,
                        default='default',
                        help='Name of this training run. Will store results in output/[name]')
    args, unparsed = parser.parse_known_args()

    main(args.data_dir, args.hdf5)
