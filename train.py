'''Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
import pickle

import time

from keras.callbacks import TensorBoard
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D

import argparse
from datasets.tiny_imagenet import *

# Initial Settings
batch_size = 32
num_classes = 200
# epochs = 100
epochs = 50
sample_size = 100000
validation_sample_size = 10000
save_dir = os.path.join(os.getcwd(), 'saved_models')

img_width, img_height = 64, 64


def create_simple_model(input_shape):
    model = Sequential()
    # First convolution layer. 32 filters of size 3.
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    # First batch normalization layer
    model.add(BatchNormalization())
    # First Pooling layer. 64x64x32 -> 32x32x32
    model.add(MaxPooling2D((2, 2), 1, padding='same'))

    # Second convolution layer. 32 filters of size 3. Activation function ReLU. 32x32x32 -> 32x32x32
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    # Activation function ReLU. 32x32x32 -> 32x32x32
    # Second batch normalization layer
    model.add(BatchNormalization())

    # Second Pooling layer. 32x32x32 -> 16x16x32
    model.add(MaxPooling2D((2, 2), 1, padding='same'))

    # First fully connected layer. 16x16x32 -> 1x8192 -> 1x4096. ReLU activation.
    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))

    # Third batch normalization layer
    model.add(BatchNormalization())

    # Dropout layer for the first fully connected layer.
    model.add(Dropout(0.5))

    # Second fully connected layer. 32x32x32 -> 1x32768 -> 1x4096. ReLU activation.
    model.add(Dense(4096))
    model.add(Activation('relu'))

    # Forth batch normalization layer
    model.add(BatchNormalization())

    # Dropout layer for the second fully connected layer.
    model.add(Dropout(0.5))

    # Final fully connected layer. 1x4096 -> 1x200. Maps to class labels. Softmax activation to get probabilities.
    model.add(Dense(200))
    model.add(Activation('softmax'))

    return model


def create_lava_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200))
    model.add(Activation('softmax'))

    return model


def main(data_dir, model_name):
    model = create_simple_model(input_shape=(img_height, img_width, 3))
    # print model structure
    model.summary()

    # visualize
    from keras.utils import plot_model
    plot_model(model, to_file=model_name + '.png')

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
    adam = keras.optimizers.adam(lr=0.001)
    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    def top_5_accuracy(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=5)

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',  # use rmsprop optimizer
                  metrics=['accuracy', top_5_accuracy])

    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        data_dir + '/train',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
        data_dir + '/val/images',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    # train_datagen.fit(x_train)

    now = time.strftime("%c")
    run_name = model_name + now
    tensorbd = TensorBoard(log_dir='./logs/' + run_name, histogram_freq=0, batch_size=batch_size)
    # history = model.fit_generator(
    #     train_generator,
    #     steps_per_epoch=train_generator.n // train_generator.batch_size,
    #     epochs=epochs,
    #     validation_data=validation_generator,
    #     validation_steps=train_generator.n // train_generator.batch_size,
    #     workers=4,
    #     callbacks=[tensorbd])
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n // train_generator.batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=train_generator.n // train_generator.batch_size,
        workers=8)

    # persist the training data
    save(history, model_name)
    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    model_path = os.path.join(save_dir, model_name + '.h5')
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # Score trained model.
    score = model.evaluate_generator(
        validation_generator,
        steps=validation_sample_size / batch_size,
        workers=8)

    print(score)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def save(obj, name):
    try:
        filename = open(name + ".pickle", "wb")
        pickle.dump(obj, filename)
        filename.close()
        return (True)
    except:
        return (False)


if __name__ == '__main__':
    # Parse arguments and create output directories.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='data/tiny-imagenet-200',
                        help='Directory in which the input data is stored.')
    parser.add_argument('--name', type=str,
                        default='lavanet',
                        help='Name of this training run. Will store results in output/[name]')
    args, unparsed = parser.parse_known_args()

    main(args.data_dir, args.name)
