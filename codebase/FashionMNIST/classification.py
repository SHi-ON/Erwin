# TF and tf.keras
import tensorflow as tf
from tensorflow import keras

# import keras

import numpy as np
import matplotlib.pyplot as plt

# if you import Keras from tf.keras it would be different
print(keras.__version__)

fashion_mnist = keras.datasets.fashion_mnist
type(fashion_mnist)

# x_train, y_train and x_test, y_test
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

type(train_images)
train_images.shape
len(train_images)
train_labels.shape
len(train_labels)

test_images.shape
test_labels.shape

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure()
plt.imshow(train_images[54320], cmap='cividis')
plt.colorbar()
plt.show()

train_images[54320][10]

# Normalization
train_images = train_images / 255.0
test_images = test_images / 255.0

# plot image size times 100 in pixels
plt.figure(figsize=(8, 6))
for i in range(25):
    plt.subplot(5, 7, i + 1)
    # no ticks
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i+4])
    plt.xlabel(class_names[train_labels[i]])
    plt.colorbar()
plt.show()

# keras.Sequential() aliases keras.models.Sequential()
model = keras.models.Sequential()
# transforms the format of the images from a 2d-array (of 28 by 28 pixels), to a 1d-array of 28 * 28 = 784 pixels.
# Think of this layer as unstacking rows of pixels in the image and lining them up.
# This layer has no parameters to learn; it only re-formats the data.
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

# metrics to monitor the training and testing steps
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# start to fitting the model
model.fit(train_images, train_labels, epochs=10)

# It turns out, the accuracy on the test dataset is a little less than the accuracy on the training dataset.
# This gap between training accuracy and test accuracy is an example of over-fitting.
# Over-fitting is when a machine learning model performs worse on new data than on their training data.
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_loss, test_acc)

prediction = model.predict(test_images)
print(prediction[0])
prediction[0]

np.max(prediction[1000])
np.argmax(prediction[0])
class_names[9]
test_labels[0]


def plot_image(i, prediction_array, true_label, img):
    prediction_array, true_label, img = prediction_array[i], true_label[i], img[i]
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap='binary')

    predicted_label = np.argmax(prediction_array)
    if predicted_label != true_label:
        color = 'red'
    else:
        color = 'blue'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], np.max(prediction_array) * 100, class_names[true_label]), color=color)


def plot_value_array(i, prediction_array, true_label):
    prediction_array, true_label = prediction_array[i], true_label[i]
    plt.xticks([])
    plt.yticks([])

    barplot = plt.bar(range(10), prediction_array, color='grey')
    plt.ylim([0, 1])
    predicted_label = np.argmax(prediction_array)

    # wise coloring. if the same only blue color, if not two diff colors
    barplot[predicted_label].set_color('red')
    barplot[true_label].set_color('blue')


i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, prediction, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, prediction, test_labels)
plt.show()

num_rows = 9
num_cols = 10
num_images = num_rows * num_cols
# good landscape sizing
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2*i+1)
    plot_image(i, prediction, test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2*i+2)
    plot_value_array(i, prediction, test_labels)
plt.show()


