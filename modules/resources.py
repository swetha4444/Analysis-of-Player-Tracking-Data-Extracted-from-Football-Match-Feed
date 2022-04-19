import cv2
from sklearn.cluster import KMeans
import numpy as np
from modules.jerseycolor import *


pallete = {'b': (0, 0, 128),
        'g': (0, 128, 0),
        'r': (255, 0, 0),
        'c': (0, 192, 192),
        'm': (192, 0, 192),
        'y': (192, 192, 0),
        'k': (0, 0, 0),
        'w': (255, 255, 255)}

color_for_labels = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in color_for_labels]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        
        try:
            color = getClusters(img[y1:y2, x1:x2])
            #color = closest_color(list_of_colors, cluster)[0]
            #color = (int(color[0]), int(color[1]), int(color[2]))
        except:
            pass
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img

def transform_matrix(matrix, p, vid_shape, gt_shape):
    p = (p[0]*1280/vid_shape[1], p[1]*720/vid_shape[0])
    px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
    py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
    p_after = (int(px*gt_shape[1]/115) , int(py*gt_shape[0]/74))
    return p_after