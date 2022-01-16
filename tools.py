"""
Useful tools for run YOLOv3 in an image or video
Date: January 2022
Author: Hanna L. Aguilar
"""
from typing import Union, List
import copy
import matplotlib.colors as mcolors
import cv2
import numpy as np

COLORS = list(mcolors.TABLEAU_COLORS)
COLORS = [mcolors.to_rgb(color) for color in COLORS]
COLORS = np.array(COLORS) * 255


def find_objects(img: np.ndarray,
                 classes_name: Union[List, str],
                 outputs: np.ndarray,
                 confidence_threshold: float = 0.7,
                 nms_threshold: float = 0.3) -> np.ndarray:
    """
    Find objects in an image using YOLOv3 and return the image with the drawn objets

    Args:
        img: image in RGB
        classes_name: coco names list
        outputs: 3 outputs from the YOLOv3 net
        confidence_threshold: minimum confidence for decide the predominant class in an image
        nms_threshold: threshold use for eliminating overlapping boxes. Lower values means more strict threshold

    Returns:
        image with boxes, classes and percentages of prediction of the detected objects

    """
    img_boxes = copy.deepcopy(img)
    height, width, _ = img.shape
    # Bounding box list for every object in the image
    bbox = []
    class_ids = []
    confidences = []
    for output in outputs:
        for detection in output:
            # Get de scores for class prediction
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence_prediction = scores[class_id]
            if confidence_prediction > confidence_threshold:
                # Geometry parameters in order to draw a rectangle in the object
                w_pixels, h_pixels = int(
                    detection[2] * width), int(detection[3] * height)
                x_box = int(detection[0] * width - w_pixels / 2)
                y_box = int(detection[1] * height - h_pixels / 2)
                bbox.append([x_box, y_box, w_pixels, h_pixels])
                class_ids.append(class_id)
                confidences.append(confidence_prediction)

    # Select which of boxes to keep by given the index
    indices = cv2.dnn.NMSBoxes(bboxes=bbox,
                               scores=confidences,
                               score_threshold=confidence_threshold,
                               nms_threshold=nms_threshold)

    for i, idx in enumerate(indices):
        box = bbox[idx]
        x_box, y_box, w_box, h_box = box
        # Draw boxes and text
        cv2.rectangle(img_boxes,
                      (x_box, y_box),
                      (x_box + w_box, y_box + h_box),
                      COLORS[i], 2)
        cv2.putText(img_boxes,
                    text=f'{classes_name[class_ids[idx]].upper()}:{int(confidences[idx]*100)}%',
                    org=(x_box, y_box-10),
                    fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1,
                    color=COLORS[i], thickness=2)

    return img_boxes
