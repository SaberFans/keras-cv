'''Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function

import argparse
import time

import keras
from keras import applications, Input, Model
from keras.callbacks import TensorBoard
from keras.layers import Flatten, Dense, GlobalAveragePooling2D
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator

from datasets.tiny_imagenet import *

# Initial Settings
batch_size = 32
num_classes = 200
# epochs = 100
epochs = 25
sample_size = 100000
validation_sample_size = 10000
save_dir = os.path.join(os.getcwd(), 'saved_models')

img_width, img_height = 224, 224


def main(data_dir, model_name, pretrain=None):
    if pretrain == 'yes':
        pretrain = 'imagenet'

    xcep_model = applications.Xception(weights=pretrain, input_shape=(img_width, img_height, 3),
                                       include_top=False)
    x = xcep_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # add your top layer block to your base model
    model = Model(xcep_model.input, predictions)

    # this is the model we will fine-tune
    model.summary()

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    adam = keras.optimizers.adam(lr=0.001)
    sgd = keras.optimizers.sgd(lr=0.01, decay=0.004, momentum=.9)

    def top_5_accuracy(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=5)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy', top_5_accuracy])

    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)  # normalize the grb value
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        data_dir + '/train',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        data_dir + '/val',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    # train_datagen.fit(x_train)

    print(train_generator.n // train_generator.batch_size)
    print("---------")

    now = time.strftime("%c")
    run_name = model_name + now
    tensorbd = TensorBoard(log_dir='./logs/' + run_name, histogram_freq=0, batch_size=batch_size)
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n // train_generator.batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=train_generator.n // train_generator.batch_size,
        workers=4,
        callbacks=[tensorbd])
    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    model_path = os.path.join(save_dir, model_name + '.h5')
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # Score trained model.
    # score = mob_model.evaluate_generator(
    #     validation_generator,
    #     steps=validation_sample_size / batch_size,
    #     workers=4)
    #
    # print(score)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])


if __name__ == '__main__':
    # Parse arguments and create output directories.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='data/',
                        help='Directory in which the input data is stored.')
    parser.add_argument('--name', type=str,
                        default='xception',
                        help='Name of this training run. Will store results in output/[name]')
    parser.add_argument('--pretrain', type=str,
                        help='Name of this training run. Will store results in output/[name]')
    args, unparsed = parser.parse_known_args()

    main(args.data_dir, args.name, args.pretrain)
