'''Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function

import argparse
import os.path
import time

import keras
from keras import models
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.metrics import top_k_categorical_accuracy

from datasets.tiny_imagenet import *
# Initial Settings
from util.modelutil import create_miniCnn_model, create_trivial_model, get_datagen, write2f

batch_size = 32
num_classes = 200
# epochs = 100
epochs = 50
sample_size = 100000
validation_sample_size = 10000
save_dir = os.path.join(os.getcwd(), 'saved_models/miniCnn')

img_width, img_height = 128, 128

def train(data_dir, opti, model_name, data_aug=True, lossfunc='categorical_crossentropy', dropout=True):

    # by default train miniCnn
    model = create_miniCnn_model(input_shape=(img_height, img_width, 3), dropout=dropout)
    if 'trivial' in model_name:
        model = create_trivial_model(input_shape=(img_height, img_width, 3))
    if 'miniCnn' in model_name:
        model = create_miniCnn_model(input_shape=(img_height, img_width, 3))
    # print model structure
    model.summary()

    # custom top 5 accuracy
    def top_5_accuracy(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=5)
    custom_metric = top_5_accuracy
    assert (model is not None), 'model_name is empty, define the one you want to run!'
    if os.path.isfile(save_dir.join(model_name)+'.h5'):
        print('---------loading existing pre-trained model---------')
        model = models.load_model(os.path.join(save_dir, model_name)+'.h5',
                                  custom_objects={'top_5_accuracy': custom_metric})

    # visualize
    from keras.utils import plot_model
    plot_model(model, to_file=model_name + '.png')
    # compile model
    model.compile(loss=lossfunc,
                  optimizer=opti,
                  metrics=['accuracy', top_5_accuracy])

    (train_generator, validation_generator) = get_datagen(data_aug, data_dir, img_width = img_width, img_height=img_height)

    now = time.strftime("%c")
    run_name = model_name + now
    # tensorboard
    tensorbd = TensorBoard(log_dir='./logs/' + run_name, histogram_freq=0, batch_size=batch_size)
    # best checkpoint
    filepath = save_dir.join(model_name+"weights.best.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # early stop
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5,
                              verbose=1, mode='auto')
    # record into local log
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n // train_generator.batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=train_generator.n // train_generator.batch_size,
        workers=4,
        callbacks=[tensorbd, checkpoint, earlystop])
    #
    # # persist the model history
    # save(history, os.path.join(save_dir, model_name))

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

    # write evaluation score to file
    write2f(score, save_dir.join(model_name + '_eval_score'))


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
