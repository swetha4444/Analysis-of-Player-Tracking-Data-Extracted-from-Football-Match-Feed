from modules.yolo import YOLO
from modules.yolo import *
from modules.deepsort import *
from modules.perspective_transform import Perspective_Transform
from yolov5.utils.plots import plot_one_box
import cv2
import numpy as np
from modules.resources import *
from modules.jerseycolor import *
import matplotlib.pyplot as plt
import sys

frame_num = 0
cap = cv2.VideoCapture("./static/input3.mp4")
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
detector = YOLO("models/yolov5s.pt",0.5, 0.3)
deep_sort = DEEPSORT("deep_sort_pytorch/configs/deep_sort.yaml")
perspective_transform = Perspective_Transform()

bg_ratio = int(np.ceil(w/(3*115)))
gt_img = cv2.imread('./static/black.jpg')
print((115*bg_ratio, 74*bg_ratio))
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
    
    if ret:
        main_frame = frame.copy() 
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yoloOutput = detector.detect(frame)
        
        if frame_num % 5 ==0:
            #M,gt_h, gt_w = getHomogrpahyMatrix('./static/world_cup_template.png',frame)
            M, warped_image = perspective_transform.homography_matrix(main_frame)
            #visualise_homography(frame,'./static/black.jpg',M)
        
        
        if yoloOutput:
                deep_sort.detection_to_deepsort(yoloOutput, frame)

                for i, obj in enumerate(yoloOutput):
                    xyxy = [obj['bbox'][0][0], obj['bbox'][0][1], obj['bbox'][1][0], obj['bbox'][1][1]]
                    x_center = (xyxy[0] + xyxy[2])/2 
                    y_center = xyxy[1]
                    if obj['label'] == 'player':
                        try:
                            color = detect_color(main_frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]])
                        except:
                          pass
                        coords = transform_matrix(M, (x_center, y_center), (h, w), (gt_h, gt_w))
                        cv2.circle(bg_img, coords, 5, (255,0,0), -1)
                        #cv2.putText(bg_img, str(i), coords, cv2.FONT_HERSHEY_PLAIN, 0.2, [255, 255, 255], 1)
                    
                    elif obj['label']=='ball':
                        coords = transform_matrix(M, (x_center, y_center), (h, w), (gt_h, gt_w))
                        cv2.circle(bg_img, coords, 5, (0,0,255), -1)
                        plot_one_box(xyxy,frame,(102,0,102),label='ball')
                        
        else:
           deep_sort.deepsort.increment_ages()
    
        frame[frame.shape[0]-bg_img.shape[0]:, frame.shape[1]-bg_img.shape[1]:] = bg_img
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
    
cap.release()
cv2.destroyAllWindows()