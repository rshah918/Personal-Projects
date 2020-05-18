import ImagePyramid
import numpy as np
from PIL import Image


def SlidingWindow(image, windowSize=25, stepsize=5):
    # slide a window across the image
    windows = []
    for y in range(0, image.shape[0], stepsize):
        for x in range(0, image.shape[1], stepsize):
            windows.append([x,x+windowSize,y,y+windowSize])
    # yield the current window
    yield windows
