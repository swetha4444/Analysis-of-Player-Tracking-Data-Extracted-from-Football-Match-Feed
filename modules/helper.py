from modules.resources import *
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import nums_from_string

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

def formatData(playerTrackingData,ballTrackingData):
    playerdf = pd.DataFrame(playerTrackingData,columns=["Frame","(x,y)","Player ID","Team"])
    teams = findPlayerTeam(playerTrackingData)
    playerdf['Teams'] = teams
    balldf = pd.DataFrame(ballTrackingData,columns=["Frame","(x,y)"])
    playerdf.to_csv('./static/PlayerTrackingData.csv')
    balldf.to_csv('./static/BallTrackingData.csv')


