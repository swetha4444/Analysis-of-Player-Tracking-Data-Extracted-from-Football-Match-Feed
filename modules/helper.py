from modules.resources import *
from modules.yolo import YOLO
from modules.yolo import *
from modules.deepsort import *
from modules.perspective_transform import Perspective_Transform
from yolov5.utils.plots import plot_one_box
from modules.helper import *
import cv2
import numpy as np
from modules.resources import *
from modules.jerseycolor import *
import matplotlib.pyplot as plt
import sys
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import nums_from_string

def pipeline(inputPath,outputPath):
    playerTrackingData =[]
    ballTrackingData =[]
    frame_num = 0
    cap = cv2.VideoCapture(inputPath)
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
    result = cv2.VideoWriter(outputPath, 
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            20, size)
    while(cap.isOpened()):    
        ret,frame = cap.read()
        bg_img = gt_img.copy()
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
    cap.release()
    cv2.destroyAllWindows()
    return playerTrackingData,ballTrackingData


def getPlayerCoordinates(outputs,M,template,frame_size,template_size,frame_count,frame):
    frame_data = []
    for obj in outputs:
        xyxy = [obj[0], obj[1], obj[2], obj[3]]
        id = obj[4]
        x_center = (xyxy[0] + xyxy[2])/2 
        y_center = xyxy[3]
        coords = transform_matrix(M, (x_center, y_center),frame_size,template_size)
        try:
            color = getClusters(frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]])
            cv2.putText(template, str(id), coords, cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
            frame_data.append([frame_count,list(coords),id,color])
        except:
            pass
    return frame_data,template

def findPlayerTeam(playerTrackingData):
    data = []
    for x in playerTrackingData['Team']:
        data.append(nums_from_string.get_nums(x))
    km = KMeans(n_clusters=2).fit(data)
    return km.labels_

def Insert_row(row_number, df, row_value):
    start_upper = 0
    end_upper = row_number
    start_lower = row_number
    end_lower = df.shape[0]
    upper_half = [*range(start_upper, end_upper, 1)]
    lower_half = [*range(start_lower, end_lower, 1)]
    lower_half = [x.__add__(1) for x in lower_half]
    index_ = upper_half + lower_half
    df.index = index_
    df.loc[row_number] = row_value
    df = df.sort_index()
    return df

def ballFormat(data):
    data[['x','y']] = data['(x,y)'].str.split(',',expand=True)
    data['x'] = data['x'].str.replace('[','')
    data['y'] = data['y'].str.replace(']','')
    j=0
    data1 = data
    while j<(len(data)):
        j+=1
        for i in range(0,len(data1)):
            if i!=data1['Frame'][i]:
                df2 = {'Frame':i}
                data1 = Insert_row(i, data1, df2)
    data1.drop(labels=['Unnamed: 0','(x,y)'], axis=1, inplace=True)    
    return data1

def createsubListBall(inpt):
    rv = []
    curr = [inpt[0]]
    for x in inpt[1:]:
        if x - 1 != curr[-1]:
            rv.append(curr)
            curr = [x]
        else:
            curr.append(x)
    rv.append(curr)
    return rv

def insertBallValues(balldf):
    indices = balldf[balldf['x'].isnull()].index.tolist()
    subLists = createsubListBall(indices)
    valuesX = []
    valuesY = []
    for sl in subLists:
        vslX,vslY = [],[]
        initialI = sl[0]-1
        finalI = sl[-1]+1
        Xin = int((float(balldf['x'].iloc[finalI]) - float(balldf['x'].iloc[initialI]))/len(sl))
        Yin = int((float(balldf['y'].iloc[finalI]) - float(balldf['y'].iloc[initialI]))/len(sl))
        vslX.append(int(balldf['x'].iloc[initialI])+Xin)
        vslY.append(int(balldf['y'].iloc[initialI])+Yin)
        for index in sl[:-1]:
            vslX.append(int(vslX[-1])+Xin)
            vslY.append(int(vslY[-1])+Yin)
        valuesX.append(vslX)
        valuesY.append(vslY)
    
    for i in range(len(subLists)):
        for j in range(len(subLists[i])):
            balldf['x'].iloc[subLists[i][j]] = valuesX[i][j]
            balldf['y'].iloc[subLists[i][j]] = valuesY[i][j]
    return balldf


def formatData(playerTrackingData,ballTrackingData):
    playerdf = pd.DataFrame(playerTrackingData,columns=["Frame","(x,y)","Player ID","Team"])
    teams = findPlayerTeam(playerTrackingData)
    playerdf['Teams'] = teams
    balldf = pd.DataFrame(ballTrackingData,columns=["Frame","(x,y)"])
    balldf = ballFormat(balldf)
    balldf = insertBallValues(balldf)
    playerdf.to_csv('./static/PlayerTrackingData.csv')
    balldf.to_csv('./static/BallTrackingData.csv')


