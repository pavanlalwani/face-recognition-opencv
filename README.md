## Face Recognition in OpenCV using LBPH classifiers

Local Binary Patterns (LBP) is a visual descriptor that can be used to detect visual features in images. LBP combined with histograms can be used to represent face images as a simple vector.
 
This project uses the LBPHFaceRecognizer class in OpenCV to create a model and classify faces accordingly.

## Setup

Make sure you have configured your webcam properly. 
After cloning, execute the following to install required packages and test your environment:

```bash
pip install -r requirements.txt
python facedetection.py
```

## Usage
Execute the files in the following order:

```python
python newfacedata.py
python trainer.py
python facerecognition.py
```
newfacedata.py captures the detected face from the webcam input and saves the images to /dataset. Make sure to use a unique IDs every time you enroll a new face. 

Run trainer.py each time to generate the LBPH model after you capture new face data to /dataset

