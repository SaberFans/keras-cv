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
from sklearn.metrics import confusion_matrix

from datasets.tiny_imagenet import *

# Initial Settings
batch_size = 32
num_classes = 200
# epochs = 100
epochs = 25
sample_size = 100000
validation_sample_size = 10000
save_dir = os.path.join(os.getcwd(), '../saved_models/miniCnn')

img_width = 128
img_height = 128

def confu_matrix_gen(data_dir, validation_generator, model=None):
    # if model ==None:
    #     model = applications.ResNet50(weights=None, input_shape=(img_width, img_height, 3), classes=2)
    #
    y_real = validation_generator.classes
    y_preds = model.predict_generator(validation_generator, verbose=0)
    # # print('Predicted:', decode_predictions(preds))
    y_preds = np.argmax(y_preds, axis=1)

    # # validation_generator.reset()
    # # print (validation_generator.class_indices)

    print('real class:', y_real)
    print('predicted class:', y_preds)
    # class_ind = validation_generator.class_indices
    # print('class ind:', class_ind)
    #
    # # orderedKeys = OrderedDict(class_ind)
    # # print('Predicted:', decode_predictions(preds, top=3)[0])
    #
    # values = class_ind.values()
    # print('sorted values:', values)

    classes = 20
    # y_preds = [0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7]
    # y_real =  [0,1,3,3,4,5,6,7,0,1,3,3,4,5,6,7]

    y_preds_m = []
    y_real_m= []
    y_index = 0
    for i, j in zip(y_preds, y_real):
        if i<classes and j<classes:
            y_preds_m.append(i)
            y_real_m.append(j)


    cnf_matrix = confusion_matrix(y_real_m, y_preds_m)
    print('----------confusion matrix--------')
    print(cnf_matrix)

    # labels = np.arange(classes)
    # plot_confusion_matrix(cnf_matrix, classes=labels)
    # np.set_printoptions(precision=2)
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
print('---------loading existing pre-trained model---------')
model = models.load_model(os.path.join(save_dir, 'miniCnn.h5'), custom_objects={'top_5_accuracy':custom_metric})


test_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = test_datagen.flow_from_directory(
        '../data/val',
        target_size=(img_height, img_width),
        batch_size=1,
        class_mode='categorical',
        shuffle=False)
confu_matrix_gen('../data', validation_generator)
