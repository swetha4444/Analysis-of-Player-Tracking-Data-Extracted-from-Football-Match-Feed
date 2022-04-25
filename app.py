from modules.helper import *

inputPath = "./static/input3.mp4"
outputPath = "./static/output.mp4"

playerTrackingData,ballTrackingData = pipeline(inputPath,outputPath)
formatData(playerTrackingData,ballTrackingData)