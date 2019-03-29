import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

tf.logging.set_verbosity(tf.logging.DEBUG)

raw_data = pd.read_csv('./codebase/adrel/dataset/train.csv')
raw_data.head()

seed = 27
train, validate = train_test_split(raw_data, test_size=0.1, random_state=seed, stratify=raw_data['label'])

x_train = train.values[:, 1:]
type(x_train)
x_train.shape
# label column
y_train = train.values[:, 0]

x_validate = validate.values[:, 1:]
# label column
y_validate = validate.values[:, 0]

batch_size = 512
num_classes = 10
epochs = 32

# input image dimensions
img_rows, img_cols = 28, 28

# BW images have only 1 channel as RGB's have 3
# 'channels_first': (samples, channels, rows, cols)
if k.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_validate = x_validate.reshape(x_validate.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
# TF default - 'channels_last': (samples, rows, cols, channels)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_validate = x_validate.reshape(x_validate.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# finding a non-zero number in the train dataset
# to test astype() method effect
# for i in range(len(x_train)):
#     for j in range(len(x_train[i])):
#         for k in range(len(x_train[i][j])):
#             if x_train[i, j, k, 0] > 0:
#                 print('({}, {}, {}) -'.format(i, j, k), end=' ')
type(x_train[1148][7][17][0])
x_train[1148][7][17][0]

# type casting
x_train = x_train.astype('float32')
x_validate = x_validate.astype('float32')

# normalization
x_train /= 255
x_validate /= 255

x_train.shape
print(x_train.shape[0], 'number of train samples')
print(x_validate.shape[0], 'number of validation samples')

# multi-class vector to binary class matrices
# one-vs-all (OvA, OvR)
# equivalent to sklearn.preprocessing.LabelBinarizer()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_validate = keras.utils.to_categorical(y_validate, num_classes)

# data generation feature of Keras
datagen = ImageDataGenerator(
    width_shift_range=0.075,
    height_shift_range=0.075,
    rotation_range=12,
    shear_range=0.075,
    zoom_range=0.05,
    fill_mode='constant',
    cval=0
)

datagen.fit(x_train)

# building the model
model = Sequential()

# convolutional module 1
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# convolutional module 2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# terminal module
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))

# classification layer with softmax activation function
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001)

model.fit_generator(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    epochs=epochs,
    steps_per_epoch=x_train.shape[0] / 32,
    verbose=1,
    validation_data=(x_validate, y_validate),
    callbacks=[reduce_lr])

score = model.evaluate(x_validate, y_validate, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])



