'''Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras import applications

from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.layers import MaxPooling2D

import argparse
from datasets.tiny_imagenet import *

# Initial Settings
batch_size = 32
num_classes = 200
# epochs = 100
epochs = 10
sample_size = 100000
validation_sample_size = 10000
save_dir = os.path.join(os.getcwd(), 'saved_models')

img_width, img_height = 224, 224


def main(data_dir, model_name):
    # AlexNet with batch normalization in Keras
    # input image is 224x224

    res50_model = applications.ResNet50(weights=None, include_top=False, input_shape=(img_width, img_height, 3))

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    adam = keras.optimizers.adam(lr=0.001)

    def top_5_accuracy(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=5)

    # Let's train the model using RMSprop
    res50_model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy', top_5_accuracy])

    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    train_datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the datasets
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the datasets
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        rescale=1. / 255)  # normalize the grb value
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

    res50_model.fit_generator(
        train_generator,
        steps_per_epoch=sample_size // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=sample_size // batch_size,
        workers=4)

    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    model_path = os.path.join(save_dir, model_name + '.h5')
    res50_model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # Score trained model.
    score = res50_model.evaluate_generator(
        validation_generator,
        steps=validation_sample_size / batch_size,
        workers=4)

    print(score)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    # Parse arguments and create output directories.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='data/tiny-imagenet-200',
                        help='Directory in which the input data is stored.')
    parser.add_argument('--name', type=str,
                        default='alex',
                        help='Name of this training run. Will store results in output/[name]')
    args, unparsed = parser.parse_known_args()

    main(args.data_dir, args.name)
