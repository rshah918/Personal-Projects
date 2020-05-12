import numpy as np
from PIL import Image
from PIL import ImageOps
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

def GetImagePyramid(imageArray, scaleFactor=1.1, windowSize=50):
    validateParameters(imageArray, scaleFactor)
    pyramidHeight = int(min(imageArray.shape[1], imageArray.shape[0])/windowSize)-1
    imagePyramid = [] #contains a list of numpy images, largest to smallest
    width = imageArray.shape[1]
    height = imageArray.shape[0]
    #add original image to pyramid
    imagePyramid.append(imageArray)
    #generate each level of the pyramid
    for pyramidLevel in range(0, pyramidHeight):
        image = Image.fromarray(imageArray, 'RGB')
        #shrink image, use bicubic interpolation
        scaledWidth = int(width - (width*(scaleFactor-1)))
        scaledHeight = int(height - (height*(scaleFactor-1)))
        #ensure image isnt smaller than window
        if(min(scaledWidth,scaledHeight) < windowSize):
            break;
        shrunkImage = ImageOps.fit(image, (scaledWidth, scaledHeight))
        #convert back to numpy array
        shrunkImageArray = np.array(shrunkImage)
        imagePyramid.append(shrunkImageArray) # add shrunk image to pyramid
        #update values for next iteration
        imageArray = shrunkImageArray
        width = scaledWidth
        height = scaledHeight
    #printImagePyramid(imagePyramid, pyramidHeight)
    return imagePyramid
