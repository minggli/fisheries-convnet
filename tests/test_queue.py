# -*- coding: utf-8 -*-

import tensorflow as tf

from app.pipeline import generate_data_skeleton, data_pipe
from app.settings import IMAGE_PATH
from app.engine.controller import multi_threading

sess = tf.Session()

train_file_array, train_label_array, valid_file_array, valid_label_array = \
        generate_data_skeleton(root_dir=IMAGE_PATH + 'train', valid_size=.2)
valid_image_batch, valid_label_batch = \
        data_pipe(valid_file_array, valid_label_array, num_epochs=1, shuffle=False)

init_op = tf.group(
            tf.local_variables_initializer(), tf.global_variables_initializer())
sess.run(init_op)

@multi_threading
def test_image():
    whole_valid_images = list()
    for _ in range(10):
        try:
            valid_image = sess.run(valid_image_batch)
            # print(valid_image[155])
            whole_valid_images.append(valid_image)
        except tf.errors.OutOfRangeError as e:
            flattened = [piece for blk in whole_valid_images for piece in blk]
            break

    print(flattened[-1])

with sess:
    test_image()
