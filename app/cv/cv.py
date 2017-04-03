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
import json
import subprocess

from app.main import FETCH, CV_TRAIN, CV_DETECT
from app.pipeline import generate_data_skeleton
from app.cv.fetchsamples import (generate_sample_skeleton, batch_retrieve,
                                 retrieve_image)
from app.cv.serializer import serialize_json
from app.settings import (HAARCASCADE, CV_SAMPLE_PATH, SYNSET_ID_POS,
                          SYNSET_ID_NEG, BASE_URL, IMAGE_PATH, BOUNDINGBOX)


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
    subprocess.call(os.path.dirnames(os.path.realpath(__file__)) +
                    '/test.sh', shell=True)

if CV_DETECT:
    # load trained Haar cascade classifier
    cascade = cv2.CascadeClassifier(HAARCASCADE + 'cascade.xml')
    file_array, _ = generate_data_skeleton(root_dir=IMAGE_PATH + 'test_stg1')
    output = list()
    for path_to_image in file_array:
        original_img = cv2.imread(path_to_image, -1)
        grayscale = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        fish = cascade.detectMultiScale(grayscale)
        filename = path_to_image.split('/')[-1]
        img_json = serialize_json(filename, fish)
        output.append(img_json)
    with open(BOUNDINGBOX + 'test.json', 'w') as f:
        json.dump(output, f, sort_keys=True, indent=4, ensure_ascii=False)
