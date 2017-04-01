# -*- coding: utf-8 -*-
"""
object detection module

using OpenCV to detect the single species of fish in each image

Competition allows use of external data, ImageNet data will be used for object
detection purpose for identify Region of Interest (ROI) containing fish.

mechanics of Haar Cascade algorithm:
http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html

guide to train a Haar Cascade:
http://docs.opencv.org/trunk/dc/d88/tutorial_traincascade.html
"""

import os
import cv2
import random
import subprocess

from app.main import FETCH, CV_TRAIN
from app.pipeline import generate_data_skeleton
from app.cv.fetchsamples import (generate_sample_skeleton, batch_retrieve,
                                 retrieve_image)
from app.settings import (HAARCASCADE, CV_SAMPLE_PATH, SYNSET_ID_POS,
                          SYNSET_ID_NEG, BASE_URL, IMAGE_PATH)


if FETCH:
    sample_pos = generate_sample_skeleton(SYNSET_ID_POS, 5e3, BASE_URL)
    sample_neg = generate_sample_skeleton(SYNSET_ID_NEG, 5e3, BASE_URL)

    batch_retrieve(func=retrieve_image,
                   iterable=sample_neg,
                   path=CV_SAMPLE_PATH + 'neg')
    batch_retrieve(func=retrieve_image,
                   iterable=sample_pos,
                   path=CV_SAMPLE_PATH + 'pos')

if CV_TRAIN:
    subprocess.call('sampletrain.sh', shell=True)

# load trained Haar cascade classifier
cascade = cv2.CascadeClassifier(HAARCASCADE + 'cascade.xml')

file_array, _ = generate_data_skeleton(
                os.path.join(os.path.realpath('.'), IMAGE_PATH) + 'train/ALB')

img = cv2.imread(random.choice(file_array))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

fish = cascade.detectMultiScale(gray, 3, 10)

for (x, y, w, h) in fish:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
