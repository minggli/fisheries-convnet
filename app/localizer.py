# -*- coding: utf-8 -*-
"""
localizer class

localize bounding boxes and pad rest of image with zeros (255, 255, 255)
"""
import os
import cv2
import numpy as np

from .cv.serializer import deserialize_json
from .settings import BOUNDINGBOX


class Localizer:

    def __init__(self, path_to_image):
        # cv2 loads image in BGR channel order
        self.path = path_to_image
        self.image = cv2.imread(path_to_image, -1)
        self.fname = os.path.split(path_to_image)[1]
        self.bboxes = None
        self.output_image = None

        self._set_bboxes()

    def _set_bboxes(self):
        try:
            self.bboxes = \
                deserialize_json(BOUNDINGBOX)[self.fname]['annotations']
        except KeyError:
            self.bboxes = None

    @property
    def _coordinates_factory(self):
        """yield bounding boxes"""
        for bbox in self.bboxes:
            x = int(bbox['x'])
            y = int(bbox['y'])
            height = int(bbox['height'])
            width = int(bbox['width'])
            yield x, x + width, y, y + height

    def declutter(self):
        filter_layer = np.zeros(shape=self.image.shape)
        # highlight image with (1, 1, 1) on background of zeros
        if self.bboxes:
            for x, x_end, y, y_end in self._coordinates_factory:
                filter_layer[y: y_end, x: x_end, :] = (1., 1., 1.)
            # elementwise multiplication of filter layer and original image
            self.output_image = cv2.convertScaleAbs(self.image * filter_layer)
        elif not self.bboxes:
            self.output_image = self.image
        return self

    def show(self):
        cv2.imshow("output", self.output_image)
        cv2.waitKey(0)

    def write(self):
        if self.bboxes:
            print('writing {}'.format(self.path))
            cv2.imwrite(self.path, self.output_image)
        elif not self.bboxes:
            print('bypassing {}'.format(self.path))

    @classmethod
    def localize(cls, path_to_image):
        cls(path_to_image).declutter().write()
