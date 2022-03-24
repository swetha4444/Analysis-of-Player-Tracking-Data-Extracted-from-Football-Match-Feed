import cv2
import pandas as plt

cap = cv2.VideoCapture("input3.mp4")
gt_img = cv2.imread('world_cup_template.png')
gt_img= cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
gt_img = cv2.resize(gt_img, (1280,720))/255.
bg_img = gt_img.copy()

cv2.circle(bg_img, (366, 825), 5, (0,0,255), -1)
while(cap.isOpened()):
    cv2.imshow('frame',bg_img)