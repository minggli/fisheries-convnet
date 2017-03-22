# -*- coding: utf-8 -*-

import tensorflow as tf
from PIL import Image

from app.pipeline import generate_data_skeleton, data_pipe
from app.settings import IMAGE_PATH
from app.controllers import multi_threading



sess = tf.Session()

train_file_array, train_label_array, valid_file_array, valid_label_array = \
        generate_data_skeleton(root_dir=IMAGE_PATH + 'train', valid_size=.2)
valid_image_batch, valid_label_batch = \
        data_pipe(valid_file_array, valid_label_array, num_epochs=1, shuffle=False)

test_file_array, _ = \
        generate_data_skeleton(root_dir=IMAGE_PATH + 'test_stg1', valid_size=None)
# there is no shuffling or more than 1 epoch of test set, only through once.
test_image_batch, _ = \
        data_pipe(test_file_array, _, num_epochs=1, shuffle=False)

init_op = tf.group(
            tf.local_variables_initializer(), tf.global_variables_initializer())

sess.run(init_op)

@multi_threading
def test_queue():
    whole_test_images = list()

    for _ in range(10):
        try:
            test_image = sess.run(test_image_batch)
            whole_test_images.append(test_image)
        except tf.errors.OutOfRangeError as e:
            flattened = [piece for blk in whole_test_images for piece in blk]
            break
    return flattened

with sess:
    total = test_queue()
    n = int(input('choose a image to test'))
    Image.fromarray(total[n]).show()
