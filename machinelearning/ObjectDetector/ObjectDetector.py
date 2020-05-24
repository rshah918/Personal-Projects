from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
import cv2
import numpy as np
import mxnet as mx
import gluoncv as gcv
import time
#YOLOv3 Object detection on your webcam for real-time detection
#Detects 20 classes: aeroplane, bicycle, boat, bottle, bus, car, cat, chair,
#cow, dining table, dog, horse, motorbike, person, potted plant, sheep, train, TV

def runDetector(fast=True):
    cap = cv2.VideoCapture(0)
    # Load the model
    if fast:
        net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_voc', pretrained=True)
    else:
        net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)
    # Compile the model for increased speed
    net.hybridize()
    while(True):
        ret, frame = cap.read()
        # Image pre-processing
        frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
        rgb_nd, frame = gcv.data.transforms.presets.ssd.transform_test(frame, short=350, max_size=700)
        # Run frame through network
        class_IDs, scores, bounding_boxes = net(rgb_nd)
        # Display the result
        img = gcv.utils.viz.cv_plot_bbox(frame, bounding_boxes[0], scores[0], class_IDs[0], thresh=0.5, class_names=net.classes)
        gcv.utils.viz.cv_plot_image(img)
        #kill loop with the q key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everythings done, release the capture
    cap.release()
    cv2.destroyAllWindows()

runDetector(fast=True)
