import numpy as np
from PIL import Image
from PIL import ImageOps
import imutils
'''
Input: 3D/2D Numpy array representing image
Generates an Image Pyramid
Output: 1D list of Numpy images, each progressively smaller by a factor of scaleFactor
'''
def printImagePyramid(imagePyramid, pyramidHeight):
    #print image pyramid
    for i in range(pyramidHeight):
        image = Image.fromarray(imagePyramid[i], 'RGB')
        image.show()

def validateParameters(imageArray, scaleFactor):
    if(scaleFactor <= 1):
        print("ERROR: scaleFactor must be > 1")
        exit(0)
    return

def GetImagePyramid(imageArray, scaleFactor=1.1, windowSize=50):
    validateParameters(imageArray, scaleFactor)
    imagePyramid = []
	# yield the original image
    imagePyramid.append(imageArray)
	# keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(imageArray.shape[1] / scaleFactor)
        imageArray = imutils.resize(imageArray, width=w)
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if imageArray.shape[0] < windowSize or imageArray.shape[1] < windowSize:
        	break
		# yield the next image in the pyramid
        imagePyramid.append(imageArray)
    return imagePyramid
