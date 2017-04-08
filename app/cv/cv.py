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

from multiprocessing import Pool
from .fetchsamples import (generate_sample_skeleton, batch_retrieve,
                           retrieve_image)

from ..main import FETCH, CV_TRAIN, CV_DETECT
from ..localizer import Localizer
from ..pipeline import generate_data_skeleton
from ..serializer import serialize_json
from ..settings import (HAARCASCADE, SYNSET_ID_POS, CV_IM_SAMPLE_PATH,
                        SYNSET_ID_NEG, BASE_URL, IMAGE_PATH, BOUNDINGBOX,
                        HAARPARAMS)


if FETCH:
    sample_pos = generate_sample_skeleton(SYNSET_ID_POS, 5e3, BASE_URL)
    sample_neg = generate_sample_skeleton(SYNSET_ID_NEG, 5e3, BASE_URL)

    batch_retrieve(func=retrieve_image,
                   iterable=sample_neg,
                   path=CV_IM_SAMPLE_PATH + 'neg')
    batch_retrieve(func=retrieve_image,
                   iterable=sample_pos,
                   path=CV_IM_SAMPLE_PATH + 'pos')

if CV_TRAIN:
    from . import description
    subprocess.call(os.path.dirname(os.path.realpath(__file__)) +
                    '/sampletrain.sh')

if CV_DETECT:
    # apply trained Haar Cascade classifier on test set.

    cascade = cv2.CascadeClassifier(HAARCASCADE + 'cascade.xml')
    file_array = generate_data_skeleton(IMAGE_PATH)[0]

    def detectobject(path_to_image, params=HAARPARAMS, haarcascadeclf=cascade):
        original_img = cv2.imread(path_to_image, -1)
        grayscale = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        fish_detector = haarcascadeclf.detectMultiScale(
                                        grayscale,
                                        minNeighbors=params['minNeighbors'],
                                        minSize=params['minSize'],
                                        maxSize=params['maxSize'])
        filename = os.path.split(path_to_image)[1]
        img_json = serialize_json(filename, fish_detector)
        print('processed {}'.format(path_to_image))
        return img_json

    output = list()
    for path_to_image in file_array:
        output.append(detectobject(path_to_image))

    with open(BOUNDINGBOX + 'test.json', 'w') as f:
        json.dump(list(filter(None, output)),
                  f,
                  sort_keys=True,
                  indent=4,
                  ensure_ascii=False)

    with Pool(4) as p:
        p.map(Localizer.localize, file_array)
