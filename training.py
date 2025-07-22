import time
import tensorflow as tf
#from tensorflow.keras.models import load_model
print(tf.__version__)
print(tf.keras.__version__)
#print("TensorFlow version:", tf.__version__)
import cv2
from cvzone.HandTrackingModule import HandDetector
#from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
#classifier = Classifier('/Users/kedarbulusu/PycharmProjects/PythonProject/Model/keras_model.h5', '/Users/kedarbulusu/PycharmProjects/PythonProject/Model/labels.txt')


offset = 20
imgSize = 300
folder = '/Users/kedarbulusu/PycharmProjects/SignLanguageDetection/data/B'
labels = ['A', 'B', 'C']
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']
        imgWhite = np.ones((imgSize,imgSize,3), np.uint8)*255
        imgCrop = img[y-offset:y+h+offset, x- offset:x+w+offset]

        #imgCropShape = imgCrop.shape


        aspectRatio = h/w
        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
            #prediction, index = classifier.getPrediction(img)
            #print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        #cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', img)
    # key = cv2.waitKey(1)
    cv2.waitKey(1)