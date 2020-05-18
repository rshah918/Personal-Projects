#use builtin webcam for realtime face detection
import cv2
import numpy as np
import tensorflow as tf
import keras
import ImagePyramid
import SlidingWindow
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalMaxPooling2D, BatchNormalization, Activation
from PIL import Image
from PIL import ImageDraw
from collections import namedtuple

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

def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

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

def predict(model, image, windowSize = 60, displayThreshold=0.8):
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
                    #filter predictions
                    if(window_output[0] > displayThreshold):
                        predictions.append([windowCoordinates, scaledImageWidth, scaledImageHeight])
    return predictions

def runDetector(windowSize=60,displayThreshold=0.998):
    cap = cv2.VideoCapture(0)
    #load model
    model = createModel()
    model.load_weights('FaceDetectorWeights.h5')

    while(True):
        #resize and greyscale each frame
        ret, frame = cap.read()
        frame = cv2.resize(frame, (391,220), interpolation=cv2.INTER_LANCZOS4)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_arry = np.array(gray)
        #predict
        predictions = predict(model, frame_arry, windowSize=windowSize,displayThreshold=displayThreshold)
        #get bounding boxes
        boxes = []
        faces = getPredictionCoordinates(frame_arry, predictions)
        for(x,y,x2,y2) in faces:
            boxes.append([x,y,x2,y2])
        #non-max suppression
        suppressed_boxes = non_max_suppression_fast(np.array(boxes), 0.01)
        for (x,y,x2,y2) in suppressed_boxes:
            x,y,x2,y2 = int(x),int(y),int(x2),int(y2)
            img = cv2.rectangle(frame, (x,y), (x2, y2), (255,0,0), 2)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): #kill with q key
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

runDetector(windowSize=60,displayThreshold=0.999)
