import numpy as np
import cv2
from imutils.video import FPS

class Detector:
    def __init__(self):
        self.facemodel=cv2.dnn.readNetFromCaffe("/home/smn06/Downloads/ax/opencv_face_detection/res10_300x300_ssd_iter_140000.protext",
        caffeModel="/home/smn06/Downloads/ax/opencv_face_detection/res10_300x300_ssd_iter_140000.caffemodel")

    def processImage(self, imgName):
        self.img=cv2.imread(imgName)
        (self.height,self.width)=self.img.shape[:2]

        self.processFrame()
        cv2.imshow("Output",self.img)
        cv2.waitKey(0)

    def processFrame(self):
        blob=cv2.dnn.blobFromImage(self.img,1.0,(300,300),(104.0,177.0,123.0),
        swapRB=False, crop=False)

        self.faceModel.setInput(blob)

        prediction=self.faceModel.forward()

        for i in range(0,prediction.shape[2]):
            if prediction[0,0,i,2] > 0.5:
                bbox=prediction[0,0,i,3:7] * np.array([self.width,self.height,self.width,self.height])
                (xmin,ymin,xmax,ymax)= bbox.astype("int")
                cv2.rectangle(self.img,(xmin,ymin),(xmax,ymax),(0,0,255),2)
