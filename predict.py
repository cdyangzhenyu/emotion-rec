#!/usr/bin/python
# -*- coding: UTF-8 -*-
from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
import datetime

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised","neutral"]

#EMOTIONS = ["生气", "反感", "害怕", "开心", "悲伤", "惊讶", "平静"]

feelings_faces = []
for index, emotion in enumerate(EMOTIONS):
    feelings_faces.append(imutils.resize(cv2.imread('emojis/' + emotion + '.png', -1),height=60,width=60))

# starting video streaming
#cv2.namedWindow('test')

imagepath = "./test_pic/happy.jpg"

image = cv2.imread(imagepath)

image = imutils.resize(image,width=300)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)

print faces

canvas = np.zeros((250, 300, 3), dtype="uint8")
frameClone = image.copy()
if len(faces) > 0:
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
        # the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        starttime = datetime.datetime.now()      
        preds = emotion_classifier.predict(roi)[0]
        endtime = datetime.datetime.now()
        delta = (endtime - starttime).microseconds/1000.0

        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
        print preds

for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
        # construct the label text
        text = "{}: {:.2f}%".format(emotion, prob * 100)
        emoji_face = feelings_faces[np.argmax(preds)]
        w = int(prob * 300)
        cv2.rectangle(canvas, (7, (i * 35) + 5),
                     (w, (i * 35) + 35), (0, 0, 255), -1)
        cv2.putText(canvas, text, (10, (i * 35) + 23),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                     (255, 255, 255), 2)
        cv2.putText(frameClone, label, (fX, fY - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                     (0, 0, 255), 2)

for c in range(0, 3):
        frameClone[10:70, 240:300, c] = emoji_face[:, :, c] * \
        (emoji_face[:, :, 3] / 255.0) + frameClone[10:70,
        240:300, c] * (1.0 - emoji_face[:, :, 3] / 255.0)

cv2.imshow("The result box. cost time: %s ms" % delta, frameClone)
print "The emotion is %s, and prob is %s" % (label, emotion_probability)
print "The processing cost time: %s ms" % delta
cv2.imshow("Probabilities", canvas)
cv2.waitKey(0)