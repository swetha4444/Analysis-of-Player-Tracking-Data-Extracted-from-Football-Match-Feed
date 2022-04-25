from modules.yolo import YOLO
from modules.yolo import *
from modules.deepsort import *
from modules.perspective_transform import Perspective_Transform
from yolov5.utils.plots import plot_one_box
from helper import *
import cv2
import numpy as np
from modules.resources import *
from modules.jerseycolor import *
import matplotlib.pyplot as plt
import sys

playerTrackingData =[]
ballTrackingData =[]
frame_num = 0
cap = cv2.VideoCapture("./static/input3.mp4")
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
detector = YOLO("models/yolov5s.pt",0.5, 0.3)
deep_sort = DEEPSORT("deep_sort_pytorch/configs/deep_sort.yaml")
perspective_transform = Perspective_Transform()

bg_ratio = int(np.ceil(w/(3*115)))
gt_img = cv2.imread('./static/world_cup_template.png')
print("FRAME SIZE: ",(115*bg_ratio, 74*bg_ratio))
gt_img = cv2.resize(gt_img,(115*bg_ratio, 74*bg_ratio))
gt_h, gt_w, _ = gt_img.shape

frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
result = cv2.VideoWriter('./static/output.mp4', 
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         20, size)
frame_num = 0

while(cap.isOpened()):    
    bbox = []
    ret,frame = cap.read()
    bg_img = gt_img.copy()
    ids = None
    
    if ret:
        main_frame = frame.copy() 
        yoloOutput = detector.detect(frame)
        
        if frame_num % 5 ==0:
            M, warped_image = perspective_transform.homography_matrix(main_frame)
    
        if yoloOutput:
            deepOutput = deep_sort.detection_to_deepsort(yoloOutput, frame)
            frame_data,bg_img = getPlayerCoordinates(deepOutput,M,bg_img,(h, w), (gt_h, gt_w),frame_num,frame)
            playerTrackingData.extend(frame_data)
            for i, obj in enumerate(yoloOutput):
                xyxy = [obj['bbox'][0][0], obj['bbox'][0][1], obj['bbox'][1][0], obj['bbox'][1][1]]
                x_center = (xyxy[0] + xyxy[2])/2 
                y_center = xyxy[3]
                
                if obj['label']  == 'player':
                    coords = transform_matrix(M, (x_center, y_center), (h, w), (gt_h, gt_w))
                    try:
                        color = getClusters(main_frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]])
                        cv2.circle(bg_img, coords, 5, color, -1)
                    except:
                      pass
                
                elif obj['label']  =='ball':
                    coords = transform_matrix(M, (x_center, y_center), (h, w), (gt_h, gt_w))
                    cv2.circle(bg_img, coords, 3, (102,0,102), -1)
                    plot_one_box(xyxy,frame,(102,0,102),label='ball')
                    ballTrackingData.append([frame_num,list(coords)])         
                        
        else:
           deep_sort.deepsort.increment_ages()
    
        frame[frame.shape[0]-bg_img.shape[0]:, frame.shape[1]-bg_img.shape[1]:] = bg_img
        cv2.imshow('frame',frame)
        frame = cv2.resize(frame, size)
        result.write(frame)
        frame_num += 1
    else:
        break
    
    sys.stdout.write(
            "\r[Input Video : input3.mp4] [%d/%d Frames Processed]"
            % (
                frame_num,
                frame_count,
            )
        )
    
    if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

formatData(playerTrackingData,ballTrackingData)   
cap.release()
cv2.destroyAllWindows()