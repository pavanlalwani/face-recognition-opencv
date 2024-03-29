#Train a Local Binary Patterns (LBPH) model
#from the labelled images in /dataset 
#Save the model to a file.
import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'dataset'

def processDataset(path):
	imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
	print(imagePaths)
	faceData = []
	IDs = []
	for imagePath in imagePaths:
		if(imagePath == 'dataset/.gitkeep'):
			break
		faceimage = Image.open(imagePath).convert('L');
		faceimageNParray = np.array(faceimage,'uint8')
		ID = int(os.path.split(imagePath)[-1].split('.')[1])
		faceData.append(faceimageNParray)
		print(ID)
		IDs.append(ID)
		cv2.imshow("training",faceimageNParray)
		cv2.waitKey(10)
	return IDs,faceData

IDs,faces=processDataset(path)
recognizer.train(faces,np.array(IDs))
recognizer.save('LBPHfacemodel.yml')
cv2.destroyAllWindows()