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


import cv2
from app.main import FETCH
from app.cv.fetchsamples import (generate_sample_skeleton, batch_retrieve,
                                 retrieve_image)
from app.settings import (HAARCASCADE, CV_SAMPLE_PATH, SYNSET_ID_POS,
                          SYNSET_ID_NEG, BASE_URL)


# load trained Haar cascade classifier
# fish_cascade = cv2.CascadeClassifier('app/assets/haarcascade_fish.xml')

if FETCH:
    sample_pos = generate_sample_skeleton(SYNSET_ID_POS, 5e3, BASE_URL)
    sample_neg = generate_sample_skeleton(SYNSET_ID_NEG, 5e3, BASE_URL)

    batch_retrieve(func=retrieve_image,
                   iterable=sample_neg,
                   path=CV_SAMPLE_PATH + 'neg')
    batch_retrieve(func=retrieve_image,
                   iterable=sample_pos,
                   path=CV_SAMPLE_PATH + 'pos')

# subprocess.call('scripts/sampletrain.sh', shell=True)
