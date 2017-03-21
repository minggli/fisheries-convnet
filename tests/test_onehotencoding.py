# -*- coding: utf-8 -*-

import tensorflow as tf

from app.pipeline import generate_data_skeleton, data_pipe
from app.settings import IMAGE_PATH
from app.controllers import multi_threading

sess = tf.Session()

train_file_array, train_label_array, valid_file_array, valid_label_array = \
        generate_data_skeleton(root_dir=IMAGE_PATH + 'train', valid_size=.2)
valid_image_batch, valid_label_batch = \
        data_pipe(valid_file_array, valid_label_array, num_epochs=1, shuffle=False)
train_image_batch, train_label_batch = \
        data_pipe(train_file_array, train_label_array, num_epochs=1, shuffle=False)

init_op = tf.group(
            tf.local_variables_initializer(), tf.global_variables_initializer())

sess.run(init_op)

@multi_threading
def test_image():
    whole_train_labels = list()
    for _ in range(10000):
        try:
            train_label = sess.run(train_label_batch)
            whole_train_labels.append(train_label)
        except tf.errors.OutOfRangeError as e:
            flattened = [piece for blk in whole_train_labels for piece in blk]
            break

    output = {i: flattened[k] for k, i in enumerate(train_file_array)}
    for key in output:
        print('{0}: {1}'.format(key, output[key]))

with sess:
    test_image()
