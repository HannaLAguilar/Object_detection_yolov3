"""
Main script to run YOLOv3 in an image in the images folder
Date: January 2022
Author: Hanna L. Aguilar
"""

import cv2
import imageio
import matplotlib.pyplot as plt

from tools import find_objects
from definitions import YOLO_DATA_PATH, IMAGE_PATH

IMAGE_NAME = 'cat_dog.jpeg'
WIDTH = 320

# Classes name
with open(YOLO_DATA_PATH / 'coco.names.txt', 'rt') as fp:
    classes = fp.read().rsplit('\n')

# YOLO
with open(YOLO_DATA_PATH / 'yolov3.cfg', 'rb') as fp:
    yolo_config = fp.read()

with open(YOLO_DATA_PATH / 'yolov3.weights', 'rb') as fp:
    yolo_weights = fp.read()

yolo_net = cv2.dnn.readNetFromDarknet(yolo_config, yolo_weights)
yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

if __name__ == '__main__':
    # For an image
    img = imageio.imread(IMAGE_PATH / IMAGE_NAME)
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (WIDTH, WIDTH),
                                 [0, 0, 0], 1, crop=False)
    yolo_net.setInput(blob)
    layer_names = yolo_net.getLayerNames()
    output_layer_names = yolo_net.getUnconnectedOutLayersNames()
    outputs = yolo_net.forward(output_layer_names)
    img_box = find_objects(img,
                           classes,
                           outputs)
    plt.figure(), plt.imshow(img_box), plt.show()
