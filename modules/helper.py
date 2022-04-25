from modules.resources import *
import pandas as pd

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
            cv2.putText(template, str(id), coords, cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0], 1)
            frame_data.append([frame_count,list(coords),id,color])
        except:
            pass
    return frame_data,template

def formatData(playerTrackingData,ballTrackingData):
    playerdf = pd.DataFrame(playerTrackingData,columns=["Frame","(x,y)","Player ID","Team"])
    balldf = pd.DataFrame(ballTrackingData,columns=["Frame","(x,y)"])
    playerdf.to_csv('./static/PlayerTrackingData.csv')
    balldf.to_csv('./static/BallTrackingData.csv')