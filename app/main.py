#! ./venv/bin/python3 -m app.main
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from app.pipeline import datapipe
from app.settings import IMAGE_PATH

from PIL import Image

if __name__ == '__main__':

    with tf.Session().as_default():
        test_image_folder = IMAGE_PATH + 'train/'
        image_batch, label_batch, _, _ = \
            datapipe(test_image_folder, test_size=.1)
        queue_initailizer = tf.train.start_queue_runners()
        label_batch = label_batch.eval()
        image = image_batch.eval()
        print(label_batch)
        print('Hello, World')
