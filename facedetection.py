#Face detection using Haar Cascades on webcam input.
#Execute this file to check if OpenCV and webcam have been configured properly.  

import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam = cv2.VideoCapture(0);
print("Press q to exit.")
while(True):
	ret,img = cam.read();

	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2 , minNeighbors=5)

	for(x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

	cv2.imshow("Face",img);

	if(cv2.waitKey(1) == ord('q')):
		break;

cam.release()
cv2.destroyAllWindows()
