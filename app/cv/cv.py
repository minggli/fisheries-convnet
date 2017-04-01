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
import subprocess
from app.settings import HAARCASCADE

# load trained Haar cascade classifier
# fish_cascade = cv2.CascadeClassifier('app/assets/haarcascade_fish.xml')


# subprocess.call('scripts/sampletrain.sh', shell=True)
