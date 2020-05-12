import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

#load handwritten digits dataset
#xtrain/xtest are the images, ytrain/ytest are the image labels
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

image_index = 68
print(y_train[image_index])#show image label
plt.imshow(x_train[image_index], cmap ='Greys')#display image
plt.show()
#reshape the data to 4d
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
print(x_train.shape)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28,28,1)
#ensure the values are floats
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#normalize data (scale between 0-1)
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(8, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
#model.add(Dense(128, activation=tf.nn.relu))
#model.add(Dropout(0.2))
model.add(Dense(128,activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=x_train[:1000],y=y_train[:1000], epochs=7)
model.evaluate(x_test[:1000], y_test[:1000])
