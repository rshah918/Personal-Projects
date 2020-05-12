import ImagePyramid
import numpy as np
from PIL import Image

def SlidingWindow(image, windowSize=25, stepsize=5):
    #gets sliding windows from image
    windows = []
    imageWidth = image.shape[1]
    imageHeight = image.shape[0]
    if imageWidth < windowSize or imageHeight < windowSize:
        return windows;
    for y in range(0,imageHeight-windowSize, stepsize):
        for x in range(0,imageWidth-windowSize, stepsize):
            windows.append([x,x+windowSize,y,y+windowSize])
    yield windows
