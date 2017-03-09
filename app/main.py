#!/venv/bin python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from .pipeline import image_resize_op
from .settings import IMAGE_PATH, IMAGE_SHAPE

test_image = IMAGE_PATH + 'train/ALB/img_00003.jpg'
print(test_image)

with tf.Session() as sess:
    image_resize_op(test_image, IMAGE_SHAPE)
    initializer = tf.global_variables_initializer()
    sess.run(initializer)
