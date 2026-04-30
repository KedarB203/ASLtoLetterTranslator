import time
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras.models import load_model
#print(tf.__version__)
print(tf.__version__)             # Should print 2.15 or newer
#print(tf.keras.__version__)       # Should not throw any error

import cv2
from cvzone.HandTrackingModule import HandDetector
#from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from tensorflow.keras.layers import DepthwiseConv2D

def depthwise_compat(*args, **kwargs):
    # Remove 'groups' if it exists (legacy models)
    kwargs.pop('groups', None)
    return DepthwiseConv2D(*args, **kwargs)

class Classifier:
    """
    Classifier class that handles image classification using a pre-trained Keras model.
    """

    def __init__(self, modelPath, labelsPath):
        self.model_path = modelPath
        np.set_printoptions(suppress=True)  # Disable scientific notation for clarity

        # Load the Keras model
        self.model = tf.keras.models.load_model(
            self.model_path,
            custom_objects={'DepthwiseConv2D': depthwise_compat}
        )

        # Create a NumPy array with the right shape to feed into the Keras model
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        self.labels_path = labelsPath

        # If a labels file is provided, read and store the labels
        if self.labels_path:
            label_file = open(self.labels_path, "r")
            self.list_labels = [line.strip() for line in label_file]
            label_file.close()
        else:
            print("No Labels Found")

    def getPrediction(self, img, draw=True, pos=(50, 50), scale=2, color=(0, 255, 0)):
        """
        Classifies the image and optionally draws the result on the image.

        :param img: image to classify
        :param draw: whether to draw the prediction on the image
        :param pos: position where to draw the text
        :param scale: font scale
        :param color: text color
        :return: list of predictions, index of the most likely prediction
        """
        # Resize and normalize the image
        imgS = cv2.resize(img, (224, 224))
        image_array = np.asarray(imgS)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the data array
        self.data[0] = normalized_image_array

        # Run inference
        prediction = self.model.predict(self.data)
        indexVal = np.argmax(prediction)

        # Draw the prediction text on the image if specified
        if draw and self.labels_path:
            cv2.putText(img, str(self.list_labels[indexVal]), pos, cv2.FONT_HERSHEY_COMPLEX, scale, color, 2)

        return list(prediction[0]), indexVal




cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier('/Users/kedarbulusu/personalFiles/ASLtoText/Model/keras_model.h5', '/Users/kedarbulusu/personalFiles/ASLtoText/Model/labels.txt')


offset = 20
imgSize = 300
folder = '/Users/kedarbulusu/PycharmProjects/SignLanguageDetection/data/B'
labels = ['A', 'B', 'C']
counter = 0

while True:
    success, img = cap.read()
    imgOutput = img.copy()
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
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)


        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            #print(prediction, index)

        cv2.putText(imgOutput, labels[index], (x,y-20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
        cv2.rectangle(imgOutput, (x, y), (x + w, y + h), (255, 0, 255), 2)

        #cv2.imshow('ImageCrop', imgCrop)
        #cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)
    cv2.waitKey(1)
