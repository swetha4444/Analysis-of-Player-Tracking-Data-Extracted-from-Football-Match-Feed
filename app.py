# from modules.helper import *

# inputPath = "./static/input3.mp4"
# outputPath = "./static/output.mp4"

# playerTrackingData,ballTrackingData = pipeline(inputPath,outputPath)
# formatData(playerTrackingData,ballTrackingData)

import streamlit as st
import cv2 as cv

video_file = st.file_uploader('video', type = ['mp4'])
print(video_file)
# cap = cv.VideoCapture(video_file)