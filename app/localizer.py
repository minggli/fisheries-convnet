# -*- coding: utf-8 -*-
"""
localizer

localize bounding boxes and pad rest of image with zeros (255, 255, 255)
"""
import os
import cv2
import numpy as np

from app.cv.serializer import deserialize_json
from app.settings import CV_SAMPLE_PATH, BOUNDINGBOX

test_image = CV_SAMPLE_PATH + 'pos/img_00003.jpg'


class Localizer(object):

    def __init__(self, path_to_image):
        self.image = cv2.imread(path_to_image, -1)
        self.fname = os.path.split(path_to_image)[1]
        self.bboxes = \
            deserialize_json(BOUNDINGBOX)[self.fname]['annotations']

    @property
    def factory(self):
        """yield bounding boxes"""
        for bbox in self.bboxes:
            x = int(bbox['x'])
            y = int(bbox['y'])
            height = int(bbox['height'])
            width = int(bbox['width'])
            yield x, x + width, y, y + height

    def new_image(self):
        background = np.zeros(shape=self.image.shape)
        # highlight image with (1, 1, 1) on background of zeros
        for x, x_end, y, y_end in self.factory:
            background[x: x_end, y: y_end] = [1, 1, 1]

        # mirrir original image's bounding boxes into new
        self.output_image = np.mutiply(self.image, background)

    def show(self):
        cv2.imshow("Display window", self.output_image)
        cv2.waitKey(0)


# # image read as it is in as BGR
# image = cv2.imread(test_image, -1)
# b = image[2: 10, 3: 11, :]
# print(b)
# c = np.zeros(shape=(8, 8, 3))
# c[3, 3] = (1, 1, 1)
# d = np.multiply(b, c)
# print(d)
