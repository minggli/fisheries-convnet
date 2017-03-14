#! ./venv/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from PIL import Image
from pipeline import pipe
from settings import IMAGE_PATH, IMAGE_SHAPE

test_image_folder = IMAGE_PATH + 'trial'

image_batch, label_batch = pipe(test_image_folder)

if __name__ == '__main__':

    with tf.Session() as sess:
        initializer = tf.global_variables_initializer()
        sess.run(initializer)

        print('Hello, World')
