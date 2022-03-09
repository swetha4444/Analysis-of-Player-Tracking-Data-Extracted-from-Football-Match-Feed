from modules.yolo import YOLO
from modules.yolo import *
from modules.deepsort import *
from yolov5.utils.plots import plot_one_box
from modules.perspectiveTransform import *

import torch
import os
import cv2
import numpy as np
import sys

cap = cv2.VideoCapture("input3.mp4")
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
detector = YOLO("models/yolov5s.pt",0.5, 0.3)
deep_sort = DEEPSORT("deep_sort_pytorch/configs/deep_sort.yaml")

while(cap.isOpened()):
    bbox = []
    ret,frame = cap.read()
    if ret:
        main_frame = frame.copy()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yoloOutput = detector.detect(frame)
        print(getHomogrpahyMatrix('black.jpg',image))
        
        if yoloOutput:
                deep_sort.detection_to_deepsort(yoloOutput, frame)

                for i, obj in enumerate(yoloOutput):
                    xyxy = [obj['bbox'][0][0], obj['bbox'][0][1], obj['bbox'][1][0], obj['bbox'][1][1]]
                    x_center = (xyxy[0] + xyxy[2])/2 
                    y_center = xyxy[3]
                    
                    if obj['label']=='ball':
                        plot_one_box(xyxy,frame,(102,0,102),label='ball')
        else:
           deep_sort.deepsort.increment_ages()
    
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & ord('q') == 0xFF:
            break
    else:
        break
    
   
cap.release()
cv2.destroyAllWindows()