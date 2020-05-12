import tensorflow as tf
import numpy as np
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


def createModel():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5,5), input_shape=(None, None, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(3,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(128, kernel_size=(3,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(GlobalMaxPooling2D())
    model.add(Dense(1,activation=tf.nn.sigmoid))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def loadData():

    #load images
    dataset = np.load('images.npy', allow_pickle=True)
    img = Image.fromarray(dataset[3][0], 'RGB')
    images = []
    for i in range(397):
        images.append(dataset[i][0])
    return images

def getLabel(windowCoordinates, imageIndex, imageWidth, imageHeight, scaledImage):
    with open('bounding_boxes.csv', 'r', newline = '') as labels:
        img = Image.fromarray(scaledImage, 'RGB')
        img1 = ImageDraw.Draw(img)

        #query dataset
        reader = csv.reader(labels)
        imageLabel = list(reader)[imageIndex]
        #get all the bounding boxes for this image
        boundingBoxCollection = []
        for i in range(2, len(imageLabel)-4):
            boundingBox = imageLabel[i:i+4]
            #convert to pixel coordinates
            boundingBox[0] = float(boundingBox[0]) * imageWidth
            boundingBox[1] = float(boundingBox[1]) * imageHeight
            boundingBox[2] = float(boundingBox[2]) * imageWidth
            boundingBox[3] = float(boundingBox[3]) * imageHeight
            boundingBoxCollection.append(boundingBox)#add to collection
        #calculate overlap between window and all bounding boxes
        for box in boundingBoxCollection:

            #define rectangle object
            Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
            #initialize rectangles
            bounding_box = Rectangle(box[0], box[1], box[2], box[3])
            bounding_box_area = ((bounding_box.xmax-bounding_box.xmin) * (bounding_box.ymax-bounding_box.ymin))
            window = Rectangle(windowCoordinates[0], windowCoordinates[2], windowCoordinates[1], windowCoordinates[3])
            window_area = ((window.xmax-window.xmin) * (window.ymax-window.ymin))

            img1.rectangle([bounding_box.xmin, bounding_box.ymin, bounding_box.xmax, bounding_box.ymax], fill="red")
            img1.rectangle([window.xmin, window.ymin, window.xmax, window.ymax])

            #calculate overlap
            dx = min(bounding_box.xmax, window.xmax) - max(bounding_box.xmin, window.xmin)
            dy = min(bounding_box.ymax, window.ymax) - max(bounding_box.ymin, window.ymin)
            if (dx>=0) and (dy>=0):
                intersection = dx*dy
                    #if window covers 80% of a bounding box and window is smaller than 140% of BB area
                if intersection >= (0.8*bounding_box_area) and window_area <= (1.5*bounding_box_area):
                    return 0.99
        img.show()
        return 0.00

def shrinkImage(image):
    #shrink images so that image height = 250 while maintaining aspect ratio
    #input and output are numpy arrays
    im = Image.fromarray(image)
    img1 = ImageDraw.Draw(im)
    height = (250)
    hpercent = (height/float(image.shape[0]))
    width = int((float(image.shape[1])*float(hpercent)))
    size = (width,height)
    im.thumbnail(size)
    print(image.shape)
    image = np.array(im)

    print(im.size)
    return image

def train(model, images, batchSize=1):
    imageIndex = 0
    batchX = []
    batchY = []
    images_in_batch = 0
    for image in images:
        image = shrinkImage(image)

        tic = timeit.default_timer()
        #generate image pyramid for each image
        windowSize = 40
        imagePyramid = ImagePyramid.GetImagePyramid(image, windowSize=windowSize, scaleFactor=1.1)
        #iterate over sliding window
        for pyramidLevel in imagePyramid:
            for imageWindows in SlidingWindow.SlidingWindow(pyramidLevel, windowSize, stepsize=5):
                for windowCoordinates in imageWindows:
                    #get window from windowCoordinates
                        tic2 = timeit.default_timer()
                        tic3 = timeit.default_timer()
                        #window from scaled image
                        window = pyramidLevel[windowCoordinates[2]:windowCoordinates[3],windowCoordinates[0]:windowCoordinates[1]]
                        #make sure image window has valid window shape
                        if(window.shape != (windowSize,windowSize,3)):
                            continue;
                        #add image to batch
                        batchX.append(np.array(window))
                        #return 1 if face is in window
                        scaledImageHeight = pyramidLevel.shape[0]
                        scaledImageWidth = pyramidLevel.shape[1]
                        label = getLabel(windowCoordinates, imageIndex, scaledImageWidth, scaledImageHeight, pyramidLevel)
                        #add label to batch
                        batchY.append(label)
                        images_in_batch +=1
                        #train!!!!
                        if images_in_batch == batchSize:
                            tok3 = timeit.default_timer()

                            print("Processing time loading batch: " + str((tok3-tic3)/(batchSize)) + " seconds")
                            model.fit(x=np.asarray(batchX),y=batchY, batch_size=batchSize,epochs=10)
                            model.save('my_model.h5')
                            batchX = []
                            batchY = []
                            images_in_batch = 0
                            print("Training Neural Network on Image " + str(imageIndex))
                            tok2 = timeit.default_timer()
                            print("Processing time per image window: " + str((tok2-tic2)/batchSize) + " seconds")



        tok = timeit.default_timer()
        print(str(imageIndex+1) + " Images complete, " + str(397-(imageIndex+1)) + " Images remaining")
        print("Processing time per image: " + str(tok-tic) + " seconds")
        exit(0)
        imageIndex+=1

#CURRENTLY TAKES 23 MINUTE PER IMAGE: ~152.18 HOURS == 6.34 DAYS
# TODO:verify getLabel, write a forward pass function to save windows with faces, Non-max suppression
model = createModel()
#model = load_model('my_model.h5')
images = loadData()
train(model,images, batchSize=32)
print("Training Complete!")
#model.evaluate(np.expand_dims(images[0], axis=0), [0.99])
