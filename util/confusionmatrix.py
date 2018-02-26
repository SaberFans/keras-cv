'''Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function

import itertools
from collections import OrderedDict

import matplotlib.pyplot as plt
from keras import applications, models
from keras.layers import np
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
save_dir = os.path.join(os.getcwd(), '../saved_models/miniCnn')

def confu_matrix_gen(data_dir, img_width, img_height, model=None):
    if model ==None:
        model = applications.ResNet50(weights=None, input_shape=(img_width, img_height, 3), classes=2)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    validation_generator = test_datagen.flow_from_directory(
        data_dir + '/val',
        target_size=(img_width, img_height),
        batch_size=1,
        class_mode='categorical',
        shuffle=False)

    preds = model.predict_generator(validation_generator, verbose=0)
    print('predictions:', np.argmax(preds, axis=1))
    #
    # print('Predicted:', decode_predictions(preds))
    # validation_generator.reset()
    # print (validation_generator.class_indices)

    y_real = validation_generator.classes

    print('real class:', y_real)
    class_ind = validation_generator.class_indices
    print('class ind:', class_ind)

    class_ind = {'a':1, 'a2':2, 'a0': 3, 'b': 4}
    # orderedKeys = OrderedDict(class_ind)
    # print('Predicted:', decode_predictions(preds, top=3)[0])


    values = class_ind.values()
    print('sorted values:', values)



    #
    # cnf_matrix = confusion_matrix(y_real, preds)
    # print(cnf_matrix)
    #
    # plot_confusion_matrix(cnf_matrix, classes=keys)
    # np.set_printoptions(precision=2)

    # # Plot non-normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=labels,
    #                       title='Confusion matrix, without normalization')

    # plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# load saved best checkpoint model
# custom top 5 accuracy
def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)
custom_metric = top_5_accuracy
test_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = test_datagen.flow_from_directory(
        '../data/val',
        target_size=(128, 128),
        batch_size=1,
        class_mode='categorical',
        shuffle=False)

print('---------loading existing pre-trained model---------')
model = models.load_model(os.path.join(save_dir, 'miniCnn.h5'), custom_objects={'top_5_accuracy':custom_metric})
# Score trained model.
score = model.evaluate_generator(
    validation_generator,
    steps=validation_generator.n // validation_generator.batch_size,
    workers=4)

print(score)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# confu_matrix_gen('../data', 197, 197)
