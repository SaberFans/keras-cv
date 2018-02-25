'''Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function

import json

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
save_dir = os.path.join(os.getcwd(), 'saved_models/miniCnn')

img_width, img_height = 64, 64


def create_miniCnn_model(input_shape, dropout=True):
    model = Sequential()
    # First convolution layer. 32 filters of size 3.
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'))

    # # First batch normalization layer, best practice is put after relu
    model.add(BatchNormalization())
    # First Pooling layer. 64x64x32 -> 32x32x32
    model.add(MaxPooling2D((2, 2), 2))

    # Second convolution layer
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    # # batch normalization layer, best practice is put after relu
    model.add(BatchNormalization())
    # Pooling layer. 64x64x32 -> 32x32x32
    model.add(MaxPooling2D((2, 2), 2))

    # Drop out layer
    if dropout:
        model.add(Dropout(0.25))

    # Third convolution layer. 64 filters of size 3. Activation function ReLU. 32x32x32 -> 32x32x32
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    # # batch normalization layer, best practice is put after relu
    model.add(BatchNormalization())
    # Pooling layer. 64x64x32 -> 32x32x32
    model.add(MaxPooling2D((2, 2), 1))

    # Forth convolution layer. 64 filters of size 3. Activation function ReLU. 32x32x32 -> 32x32x32
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

    # Second Batch Norm
    model.add(BatchNormalization())
    # Second Pooling layer. 32x32x32 -> 16x16x32
    model.add(MaxPooling2D((2, 2), 1))

    # First fully connected layer. 16x16x32 -> 1x8192 -> 1x4096. ReLU activation.
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))

    # Third batch normalization layer
    model.add(BatchNormalization())

    # Dropout layer for the first fully connected layer.
    if dropout:
        model.add(Dropout(0.5))

    # Final fully connected layer. 1x4096 -> 1x200. Maps to class labels. Softmax activation to get probabilities.
    model.add(Dense(200))
    model.add(Activation('softmax'))

    return model


def create_trivial_model(input_shape):
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


def get_datagen(data_aug, data_dir):
    train_datagen = ImageDataGenerator()
    val_datagen = ImageDataGenerator()
    if data_aug:
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            rotation_range=90)
        val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        data_dir + '/train',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
        data_dir + '/val',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')
    return train_generator, validation_generator


def train(data_dir, opti, model_name, data_aug=True, lossfunc='categorical_crossentropy', dropout=True):

    # by default train miniCnn
    model = create_miniCnn_model(input_shape=(img_height, img_width, 3), dropout=dropout)
    if 'trivial' in model_name:
        model = create_trivial_model(input_shape=(img_height, img_width, 3))
    # print model structure
    assert (model is not None), 'model_name is empty, define the one you want to run!'

    model.summary()

    # visualize
    from keras.utils import plot_model
    plot_model(model, to_file=model_name + '.png')

    # initiate RMSprop optimizer

    def top_5_accuracy(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=5)

    model.compile(loss=lossfunc,
                  optimizer=opti,
                  metrics=['accuracy', top_5_accuracy])

    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:

    (train_generator, validation_generator) = get_datagen(data_aug, data_dir)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    # train_generator.fit(x_train)

    now = time.strftime("%c")
    run_name = model_name + now
    tensorbd = TensorBoard(log_dir='./logs/' + run_name, histogram_freq=0, batch_size=batch_size)
    # record into local log
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n // train_generator.batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=train_generator.n // train_generator.batch_size,
        workers=4,
        callbacks=[tensorbd])

    # persist the model history
    save(history, os.path.join(save_dir, model_name))

    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    model_path = os.path.join(save_dir, model_name + '.h5')
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # Score trained model.
    score = model.evaluate_generator(
        validation_generator,
        steps=validation_generator.n // validation_generator.batch_size,
        workers=4)

    print(score)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    write2f(score, os.path.join(save_dir, model_name + '_eval_score'))


def save(obj, name):
    try:
        filename = open(name + ".pickle", "wb")
        pickle.dump(obj, filename)
        filename.close()
        return (True)
    except:
        return (False)

def write2f(obj, name):
    try:
        filename = open(name, "w")
        filename.write("loss: {}, accuracy: {}, top5_accuracy: {}".format(obj[0], obj[1],obj[2]))
        filename.close()
        return (True)
    except:
        return (False)

if __name__ == '__main__':
    # Parse arguments and create output directories.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='data/',
                        help='Directory in which the input data is stored.')
    parser.add_argument('--name', type=str,
                        default='miniCnn',
                        help='Name of this training run. Will store results in output/[name]')
    args, unparsed = parser.parse_known_args()
    print('----------Default MiniCnn config-------------')
    train(args.data_dir, model_name=args.name, opti='sgd')
    print('---------------------------------------------')

    # run scheduler
    rates = [0.001, 0.01, 0.1]
    optimzers = ['sgd', 'rmsprop', 'adam']
    # default opt
    optimizer = keras.optimizers.SGD(lr=rates[0], decay=1e-6, momentum=0.9, nesterov=True)

    # run through all the optimizers with different rates
    print('-----optimizer/learning rate picking process starting-------')
    for opt in optimzers:
        if opt is 'rmsprop':
            optimizer = keras.optimizers.rmsprop(lr=rate, decay=1e-6)
        elif opt is 'adam':
            optimizer = keras.optimizers.adam(lr=rate)
        for rate in rates:
            modelName = args.name + '_' + opt + '_' + str(rate)
            train(args.data_dir, opti=optimizer, model_name=modelName)
    print('----------------------------------------------')

    # run the augmentation check
    print('-----augmentation comp process starting-------')
    modelName = args.name + '_DataAug'
    train(args.data_dir, model_name= modelName, opti=optimizer, data_aug=False)

    modelName = args.name + '_noDataAug'
    train(args.data_dir, model_name=modelName, opti=optimizer, data_aug=False)
    print('----------------------------------------------')

    # run the dropout test
    print('-----drop out comp process starting-------')
    modelName = args.name + '_DropOut'
    train(args.data_dir, model_name= modelName, opti=optimizer)
    modelName = args.name + '_NoDropOut'
    train(args.data_dir, model_name=modelName, opti=optimizer, dropout=False)
    print('----------------------------------------------')
    # run different loss
    print('-----loss comp process starting-------')
    modelName = args.name + '_categorical_crossentropy'
    train(args.data_dir, model_name=modelName, lossfunc='categorical_crossentropy')
    modelName = args.name + '_sparse_categorical_crossentropy'
    train(args.data_dir, model_name=modelName, lossfunc = 'sparse_categorical_crossentropy')
    print('----------------------------------------------')