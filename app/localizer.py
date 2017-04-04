# -*- coding: utf-8 -*-
"""
localizer

localize bounding boxes and pad rest of image with zeros (255, 255, 255)
"""
import os
import cv2

from app.cv.serializer import deserialize_json
from app.settings import CV_SAMPLE_PATH, BOUNDINGBOX

test_image = CV_SAMPLE_PATH + 'pos/img_00003.jpg'

bb_json = deserialize_json(BOUNDINGBOX)
bboxes = bb_json[os.path.split(test_image)[1]]['annotations']


class Localizer(object):

    def __init__(self, path_to_image, bboxes):
        self.image = image
        self.fname = os.path.split(image)[1]

    def factory(self):
        """yield bounding boxes"""
        for bbox in self.bboxes:
            x = int(bbox['x'])
            y = int(bbox['y'])
            height = int(bbox['height'])
            width = int(bbox['width'])
            yield (x, y), (x + width, y + height)


# image read as it is in as BGR
image = cv2.imread(test_image, -1)
print(image[2, 3])
