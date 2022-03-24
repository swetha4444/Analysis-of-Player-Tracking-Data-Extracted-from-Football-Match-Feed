from modules.yolo import YOLO
from modules.yolo import *
from modules.deepsort import *
from yolov5.utils.plots import plot_one_box
from modules.perspectiveTransform import *
import cv2
import numpy as np
from modules.resources import *


frame_num = 0
cap = cv2.VideoCapture("input3.mp4")
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("Frame vid size: ",w,h)
detector = YOLO("models/yolov5s.pt",0.5, 0.3)
deep_sort = DEEPSORT("deep_sort_pytorch/configs/deep_sort.yaml")

gt_img = cv2.imread('world_cup_template.png')
gt_img= cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
gt_img = cv2.resize(gt_img, (1280,720))/255.

while(cap.isOpened()):
    bbox = []
    ret,frame = cap.read()
    bg_img = gt_img.copy()
    
    if ret:
        main_frame = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yoloOutput = detector.detect(frame)
        
        if frame_num % 5 ==0:
            M,gt_h, gt_w = getHomogrpahyMatrix('world_cup_template.png',frame)
            #M = np.linalg.inv(M)
            print(gt_h,gt_w)
        
        if yoloOutput:
                deep_sort.detection_to_deepsort(yoloOutput, frame)

                for i, obj in enumerate(yoloOutput):
                    if obj['label'] == 'player':
                        xyxy = [obj['bbox'][0][0], obj['bbox'][0][1], obj['bbox'][1][0], obj['bbox'][1][1]]
                        x_center = (xyxy[0] + xyxy[2])/2 
                        y_center = xyxy[3]
                        coords = transform_matrix(M, (x_center, y_center), (h, w), (gt_h, gt_w))
                        cv2.circle(bg_img, coords, 5, (255,0,0), -1)
                        #cv2.putText(bg_img, str(i), coords, cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 1)
                    
                    elif obj['label']=='ball':
                        coords = transform_matrix(M, (x_center, y_center), (h, w), (gt_h, gt_w))
                        cv2.circle(bg_img, coords, 5, (0,0,255), -1)
                        plot_one_box(xyxy,frame,(102,0,102),label='ball')
        else:
           deep_sort.deepsort.increment_ages()
    
        #frame[frame.shape[0]-bg_img.shape[0]:, frame.shape[1]-bg_img.shape[1]:] = bg_img
        #cv2.resize(bg_img, (1024,1024))
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & ord('q') == 0xFF:
            break
    else:
        break
    
   
cap.release()
cv2.destroyAllWindows()