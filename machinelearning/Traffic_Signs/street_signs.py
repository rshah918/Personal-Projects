import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation
import pickle
from numpy import genfromtxt

pickle_off = open("data0.pickle","rb")
data = pickle.load(pickle_off)


x_train = data['x_train'].transpose(0, 2, 3, 1)
y_train = data['y_train']
x_test = data['x_test'].transpose(0, 2, 3, 1)
y_test = data['y_test']
label_names = genfromtxt('label_names.csv', delimiter=',', dtype = None, encoding='utf8')
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(14, kernel_size=(3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(28, kernel_size=(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(500, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(43,activation=tf.nn.softmax))
print("------------------------------------------------------------------------------------------------------------------------------------ ")
print("------------------------------------------------------------------------------------------------------------------------------------ ")
print(''' _____                       _       _   _                   _   _   _                      _   _   _      _                      _
/  __ \                     | |     | | (_)                 | | | \ | |                    | | | \ | |    | |                    | |
| /  \/ ___  _ ____   _____ | |_   _| |_ _  ___  _ __   __ _| | |  \| | ___ _   _ _ __ __ _| | |  \| | ___| |___      _____  _ __| | __
| |    / _ \| '_ \ \ / / _ \| | | | | __| |/ _ \| '_ \ / _` | | | . ` |/ _ \ | | | '__/ _` | | | . ` |/ _ \ __\ \ /\ / / _ \| '__| |/ /
| \__/\ (_) | | | \ V / (_) | | |_| | |_| | (_) | | | | (_| | | | |\  |  __/ |_| | | | (_| | | | |\  |  __/ |_ \ V  V / (_) | |  |   <
 \____/\___/|_| |_|\_/ \___/|_|\__,_|\__|_|\___/|_| |_|\__,_|_| \_| \_/\___|\__,_|_|  \__,_|_| \_| \_/\___|\__| \_/\_/ \___/|_|  |_|\_\

''')
print("------------------------------------------------------------------------------------------------------------------------------------ ")
print("------------------------------------------------------------------------------------------------------------------------------------ ")

#train model
print("------------------------------------------------------------------------------------------------------------------------------------ ")
print("------------------------------------------------------------------------------------------------------------------------------------ ")
print("""\n\t\tINITIALIZING NETWORK TRAINING\n""")
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=x_train[:20000],y=y_train[:20000], epochs=5)

print('\n\t\t------ Testing the CNN ------')
#test model
image_index = np.random.randint(0, 10000)
print(image_index)
pred = model.predict(x_train[image_index:image_index+10000])
count = 0
correct = 0
for prediction in pred:
    predicted_label_index = np.argmax(prediction)
    predicted_label = label_names[predicted_label_index][1]
    if(count < 5):
        print("\t\t--------------------------------- \n")
        print("\t\tActual Image Label: ", label_names[y_train[count + image_index]][1])
        print("\t\tNeural Network Prediction: ", label_names[predicted_label_index][1])
        test_image = x_train[count + image_index]
        plt.imshow(test_image)
        plt.show()
    if(label_names[y_train[count + image_index]][1] == label_names[predicted_label_index][1]):
        correct += 1
    count +=1
print("Test accuracy: ", (correct/count) * 100, "%")
