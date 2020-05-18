import json
import codecs
import requests
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from io import BytesIO
import numpy as np
import csv
from PIL import Image
from PIL import ImageDraw
#save dataset
def extractData():
    # get links and stuff from json
    jsonData = []
    JSONPATH = "../test/input_dataset/face_detection.json"
    with codecs.open(JSONPATH, 'rU', 'utf-8') as js:
        for line in js:
            jsonData.append(json.loads(line))

    print(f"{len(jsonData)} image found!")

    print("Sample row:")
    jsonData[0]
    images = []
    for data in tqdm(jsonData):
        response = requests.get(data['content'])
        img = np.asarray(Image.open(BytesIO(response.content)))
        images.append([img, data["annotation"]])
    np.save('images.npy', images)

#get labels into a csv
def bounding_box_CSV():
    '''This program extracts bounding boxes into a csv, and optionally displays all the images with their Bounding Boxes
    '''
    with open('bounding_boxes.csv', 'w', newline = '') as datafile:
        writer = csv.writer(datafile)
        data = np.load('images.npy', allow_pickle=True) #load dataset

        showImage = True; #set to True if you want to display an image and Bounding Boxes
        for i in range(len(data)):
            imageIndex = i
            metaDataIndex = 1 #index 0 is image, index 1 is metadata
            imageMetaData = data[imageIndex][metaDataIndex] #load metadata
            if showImage == True:
                img_nparray = (data[i][0])#load image as a numpy array
                img = Image.fromarray(img_nparray, 'RGB')
                img1 = ImageDraw.Draw(img)
                #extract bounding box coordinates, image height and width
            faces = []
            img_width = imageMetaData[0]['imageWidth']
            img_height = imageMetaData[0]['imageHeight']
            faces.append(img_width)
            faces.append(img_height)
            for face in imageMetaData:
                bounding_boxes = face['points']
                x1 = bounding_boxes[0]['x']
                y1 = bounding_boxes[0]['y']
                x2 = bounding_boxes[1]['x']
                y2 = bounding_boxes[1]['y']
                faces = faces + [x1,y1,x2,y2]
                #draw bounding box
                if showImage == True:
                    img1.rectangle([x1*img_width, y1*img_height, x2*img_width, y2*img_height])
            writer.writerow(faces)
            if showImage == True:
                img.show()
                showImage = False;



        print("Bounding box coordinates saved, 'please open bounding_boxes.csv'")


extractData()
bounding_box_CSV()
