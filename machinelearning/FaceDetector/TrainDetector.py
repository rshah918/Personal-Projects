import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
import ImagePyramid
import SlidingWindow
import csv
from numpy.testing import assert_allclose
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalMaxPooling2D, BatchNormalization, Activation
from PIL import Image
from PIL import ImageDraw
from collections import namedtuple
import timeit
import glob
import random
from sklearn.utils import shuffle
#DATASET: UTKFace dataset

def createModel():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), input_shape=(None, None,1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('sigmoid'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(3,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('sigmoid'))
    model.add(BatchNormalization())

    model.add(Conv2D(128, kernel_size=(3,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('sigmoid'))
    model.add(BatchNormalization())

    model.add(GlobalMaxPooling2D())
    model.add(Dense(1,activation=tf.nn.sigmoid))
    opt = keras.optimizers.Adam(learning_rate=0.008)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])
    return model

def addSaltNoise(image,prob):
    '''
    Add salt noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def addSaltandPepperNoise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def addPepperNoise(image,prob):
    '''
    Add pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            else:
                output[i][j] = image[i][j]
    return output

def addGaussianNoise(image):
    row,col= image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = image + gauss
    noisy = np.rot90(noisy)
    return noisy

def loadData():
    #create csv of image labels
    with open('imageLabels.csv', 'w', newline = '') as datafile:
        writer = csv.writer(datafile)
        images = []
        labels = []
        num_positive = 0
        num_negative = 0
        for filepath in glob.iglob('input_dataset/*.jpg'):

            #load images and convert to grayscale
            img = Image.open(filepath)
            img = img.resize((200,200))
            img = img.convert('L')
            #determine image label based on filename
            if "neg" in filepath or "UMD" in filepath:
                #add SALT AND PEPPER noise to 8X negative samples
                labels.append([0])
                images.append(np.array(img))

                noisyImage = np.rot90(np.array(img))
                labels.append([0])
                images.append(np.array(noisyImage))

                noisyImage = addSaltandPepperNoise(np.array(img), 0.60)
                labels.append([0])
                images.append(np.array(noisyImage))

                noisyImage = np.rot90(np.array(noisyImage))
                labels.append([0])
                images.append(np.array(noisyImage))

                noisyImage = np.array(addSaltNoise(np.array(img), 0.6))
                labels.append([0])
                images.append(np.array(noisyImage))

                noisyImage = np.rot90(np.array(noisyImage))
                labels.append([0])
                images.append(np.array(noisyImage))

                noisyImage = addPepperNoise(np.array(img), 0.6)
                labels.append([0])
                images.append(np.array(noisyImage))

                noisyImage = np.rot90(np.array(noisyImage))
                labels.append([0])
                images.append(np.array(noisyImage))

                num_negative += 8
            else:
                #limit number of positive samples to ensure dataset balance
                if num_positive < 16000:
                    labels.append([1])
                    images.append(np.array(img))
                    num_positive+=1
        #shuffle data
        images, labels = shuffle(images,labels)
        #write labels to file
        for label in labels:
            writer.writerow(list(label))
        print("Number of positive samples: " + str(num_positive))
        print("Number of negative samples: " + str(num_negative))
        return images

def loadTestData():
    #load images as list of np arrays
    dataset = np.load('images.npy', allow_pickle=True)
    images = []
    for i in range(397):
        img = Image.fromarray(dataset[i][0], 'RGB')
        img = img.convert('L')
        images.append(np.array(img))
    return images

def getLabel(imageIndex):
    #return 1 if index is of a face,
    #return 0 if not a face
    with open('imageLabels.csv', 'r', newline = '') as labels:
        reader = csv.reader(labels)
        imageLabel = list(reader)[imageIndex][0]
        imageLabel = int(imageLabel)
        return imageLabel

def shrinkImage(image, newHeight=250):
    if image.shape[0] > 250:
        #shrink images so that image height = 250 while maintaining aspect ratio
        #input and output are numpy arrays
        im = Image.fromarray(image)
        img1 = ImageDraw.Draw(im)
        height = (newHeight)
        hpercent = (height/float(image.shape[0]))
        width = int((float(image.shape[1])*float(hpercent)))
        size = (width,height)
        im.thumbnail(size)
        image = np.array(im)
        return image
    #if image is small enough then dont resize
    else:
        return image

def train(model, images, batchSize=1):
    imageIndex = 0
    X = []
    Y = []
    for image in images:
        image = np.divide(image, 255.0)
        #add image label to batch
        label = getLabel(imageIndex)
        Y.append(label)
        #add image to batch
        X.append(image)
        imageIndex+=1


    #reshape image batch
    X = np.array(X)
    X = X.reshape(len(images), image.shape[0], image.shape[1], 1)
    #train
    model.fit(x=X,y=Y, batch_size=batchSize,epochs=10)
    model.save_weights('my_model_weights.h5')

    print("Training Complete!")

def displayPredictions(image, predictions):
    if len(predictions) == 0:
        return
    img = Image.fromarray(image, 'L')
    img1 = ImageDraw.Draw(img)

    for x,y,x2,y2 in getPredictionCoordinates(image, predictions):
        #convert to pixel coordinates wrt original image height and width
        img1.rectangle([x, y, x2, y2])
    img.show()

def getPredictionCoordinates(image, predictions):
    imageWidth = image.shape[1]
    imageHeight = image.shape[0]
    for prediction in predictions:
        #extract window coordinates and metadata
        windowCoordinates = prediction[0]
        scaledImageWidth = prediction[1]
        scaledImageHeight = prediction[2]
        #convert pixel coordinates to a % of image height/width
        windowCoordinates = [windowCoordinates[0]/scaledImageWidth,
        windowCoordinates[1]/scaledImageWidth,
        windowCoordinates[2]/scaledImageHeight,
        windowCoordinates[3]/scaledImageHeight]
        #convert to pixel coordinates wrt original image height and width
        yield [windowCoordinates[0]*imageWidth, windowCoordinates[2]*imageHeight, windowCoordinates[1]*imageWidth, windowCoordinates[3]*imageHeight]

def predict(model, image, windowSize = 200, displayThreshold=0.8, imagePyramid=True):
    if imagePyramid == False:
        X = []
        image2 = image
        image = np.array(image)

        X.append(np.array(image))
        X = np.array(X)
        X = X.reshape(1,image.shape[0],image.shape[1],1)
        probability = model.predict(X)
        print(probability[0])
        #filter predictions
        img = Image.fromarray(image2)
        #img.show()
        return probability[0]

    if imagePyramid == True:
        #image = shrinkImage(image)
        predictions = []
        imagePyramid = ImagePyramid.GetImagePyramid(image, windowSize=windowSize, scaleFactor=1.1)
        for pyramidLevel in imagePyramid:
            #get scaled images height and weight for later use
            scaledImageWidth = pyramidLevel.shape[1]
            scaledImageHeight = pyramidLevel.shape[0]
            for imageWindows in SlidingWindow.SlidingWindow(pyramidLevel, windowSize, stepsize=40):
                for windowCoordinates in imageWindows:
                        X = []
                        #get the window thats within the window coordinates
                        window = pyramidLevel[windowCoordinates[2]:windowCoordinates[3],windowCoordinates[0]:windowCoordinates[1]]
                        #verify that the window is the correct shape
                        if(window.shape != (windowSize,windowSize)):
                            continue;
                        #format input as np array (1,windowSize,windowSize, num_channels=1)
                        window = np.array(window)

                        X.append(window)
                        X = np.array(X)
                        X = X.reshape(1,windowSize,windowSize,1)
                        #get prediction
                        window_output = model.predict(X)
                        print(window_output[0])
                        #filter predictions
                        if(window_output[0] > displayThreshold):
                            predictions.append([windowCoordinates, scaledImageWidth, scaledImageHeight])
        return predictions

#load
model = createModel()
#model.load_weights('FaceDetectorWeights.h5')
images = loadData()
#train
train(model,images, batchSize=32)
#test
images = loadTestData()
for image in images[0:10]:

    scaleInvariant = True
    predictions = predict(model,image, windowSize=50, displayThreshold=0.995, imagePyramid=scaleInvariant)
    if scaleInvariant == True:
        displayPredictions(image, predictions)
    else:
        img = Image.fromarray(image)
        img.show()
