
# import the necessary packages
import numpy as np
import argparse
import cv2
import os
from datetime import datetime
import time
import pandas as pd
import schedule


#initializing variables
threshold = 0.3
confidence1 = 0.5
count = 0
l = 0

gpu = input("y to use gpu: ")
b=0


#accessing the yolo model
labelsPath = "yolov3.txt"
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
color = (0, 255, 200) 

#weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
#configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
weightsPath = "yolov3.weights"
configPath = "yolov3.cfg"


net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

if gpu == "y":

	print(" setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

W = None
H = None
vidpath = input("video: ")
print("accessing video stream...")
vs = cv2.VideoCapture(vidpath)
#vs =  cv2.VideoCapture(0)


def rerouting():
	cv2.putText(frame,"Re-routing",(50,150),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),3,cv2.LINE_AA)

def alarmTriggered():
	cv2.putText(frame,"Alarm Triggered",(50,150),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),3,cv2.LINE_AA)

def trafficClear():
		cv2.putText(frame,"traffic Clear",(50,150),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),3,cv2.LINE_AA)
	


while True:

	schedule.run_pending()

	(grabbed, frame1) = vs.read()
	frame = cv2.resize(frame1,(858,480),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
	if not grabbed:
		break

	if W is None or H is None:
		(H, W) = frame.shape[:2]

	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)


	boxes = []
	confidences = []
	classIDs = []
	
	for output in layerOutputs:

		for detection in output:

			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if confidence > confidence1  and LABELS[classID]== "car" or LABELS[classID]== "motorcycle" or LABELS[classID]== "bus" or LABELS[classID]== "person":
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")


				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)


	idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)


	if len(idxs) > 0:
		center = []

		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			cen = [int(x + w / 2), int(y + h / 2)]
			center.append(cen)

			#crowd strength
			b = len(center)

			cv2.rectangle(frame, (x, y), (x + (int(w*1)), (y) + (int(h*1))), color, 2)

			text = LABELS[classIDs[i]]
			cv2.putText(frame, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        		
		cv2.putText(frame, 'No of vehicles: '+str(b), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1.5, (200, 100, 0), 3, cv2.LINE_AA) 

		if b > 20:
			alarmTriggered()
				
		else:
			trafficClear()                   

	if 1 > 0:

		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		if key == ord("q"):
			break
		#print('without sd: '+str(gws))
		#print('maximum crowd strength '+str(b))
	