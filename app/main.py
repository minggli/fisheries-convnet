#! ./venv/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from PIL import Image

from pipeline import (folder_traverse, generate_data_skeleton, make_queue,
    decode_transform, batch_generator
)
from settings import IMAGE_PATH, IMAGE_SHAPE

test_image_folder = IMAGE_PATH + 'trial'

fs = folder_traverse(test_image_folder)
images_paths_array, label_array = generate_data_skeleton(file_structure=fs)

queue = make_queue(images_paths_array, label_array)
resized_image_queue, label_queue = decode_transform(input_queue=queue)

image_batch, label_batch = batch_generator(resized_image_queue, label_queue)


if __name__ == '__main__':
    print('Hello, World')
# with tf.Graph() as g:
#     with tf.Session() as sess:
#
#         initializer = tf.global_variables_initializer()
#         sess.run(initializer)
#         Image.fromarray(resized_image_queue.eval()).show()
