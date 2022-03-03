from modules.yolo import YOLO
from modules.yolo import *

import torch
import os
import cv2
import numpy as np
import sys

cap = cv2.VideoCapture("input3.mp4")
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
detector = YOLO("models/yolov5s.pt",0.5, 0.3)

while(cap.isOpened()):
    bbox = []
    ret,frame = cap.read()
    if ret:
        main_frame = frame.copy()
        yoloOutput = detector.detect(frame)
        frame = draw_boxes(frame, yoloOutput, identities=None, offset=(0, 0))
    
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & ord('q') == 0xFF:
            break
    else:
        break
    
    
        
cap.release()
cv2.destroyAllWindows()