#Detect face and display face ID
#Make sure to run trainer.py before executing this script

import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0);
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("LBPHfacemodel.yml")
id = 0

while(True):
	ret,img = cam.read();

	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces = faceDetect.detectMultiScale(gray,1.3,5);
	
	for(x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
		id,conf = recognizer.predict(gray[y:y+h,x:x+w])
		
		if(conf<50):
			cv2.putText( img , str(id), (x,y+h), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255));

	cv2.imshow("Face",img);

	if(cv2.waitKey(1) == ord('q')):
		break;

cam.release()
cv2.destroyAllWindows()
