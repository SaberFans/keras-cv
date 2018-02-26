import keras
import pickle
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization, Activation
from keras.preprocessing.image import ImageDataGenerator

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

def get_datagen(data_aug, data_dir, img_width, img_height, batch_size):
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

def create_miniCnn_model(input_shape, dropout=True):
    model = Sequential()
    # First convolution layer. 32 filters of size 3.
    model.add(Conv2D(96, kernel_size= (3, 3), strides=2, input_shape=input_shape))
    # (64-3)/2 +1 = 51*51*93
    model.add(keras.layers.LeakyReLU())
    # First Pooling layer. 128*128*32 -> 32x32x32
    model.add(MaxPooling2D((3, 3), 2))
    # (51-3)/2 + 1 = 20
    # # First batch normalization layer, best practice is put after relu
    model.add(BatchNormalization())
    # 20*20*96

    # Second convolution layer
    model.add(Conv2D(256, (5, 5), strides=1, padding='same'))
    # (20-5+2)/1 + 1 = 17*17*256
    model.add(keras.layers.LeakyReLU())

    model.add(MaxPooling2D((3, 3), 2))
    # (17-3)/2 + 1 = 8

    # # batch normalization layer, best practice is put after relu
    model.add(BatchNormalization())

    # Drop out layer
    if dropout:
        model.add(Dropout(0.1))

    # Third convolution layer. 64 filters of size 3. Activation function ReLU. 32x32x32 -> 32x32x32
    model.add(Conv2D(384, (3, 3)))
    model.add(keras.layers.LeakyReLU())
    model.add(MaxPooling2D((3, 3), 2, padding='same'))
    # (8-3+2)/2 + 1 = 4

    # # batch normalization layer, best practice is put after relu
    model.add(BatchNormalization())
    # 4*4*256

    # First fully connected layer. 16x16x32 -> 1x8192 -> 1x4096. ReLU activation.
    model.add(Flatten())
    model.add(Dense(2048))
    model.add(keras.layers.LeakyReLU())

    # Third batch normalization layer
    model.add(BatchNormalization())

    # Dropout layer for the first fully connected layer.
    if dropout:
        model.add(Dropout(0.5))

    # Final fully connected layer. 1x4096 -> 1x200. Maps to class labels. Softmax activation to get probabilities.
    model.add(Dense(200))
    model.add(Activation('softmax'))

    return model
def create_miniCnn2_model(input_shape, dropout=True):

    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), input_shape=input_shape))
    model.add(keras.layers.LeakyReLU())

    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(keras.layers.LeakyReLU())

    model.add(MaxPooling2D((3, 3), 2, padding='same'))

    model.add(Conv2D(128, (3, 3)))
    model.add(keras.layers.LeakyReLU())

    model.add(Conv2D(128, (3, 3)))
    model.add(keras.layers.LeakyReLU())

    model.add(MaxPooling2D((3, 3), 2, padding='same'))
    if dropout:
        model.add(Dropout(0.1))

    # First fully connected layer.
    model.add(Flatten())
    model.add(Dense(2048))
    model.add(keras.layers.LeakyReLU())

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
    model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same'))
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