#! ./venv/bin/python3 -m app.engine.core
# -*- coding: utf-8 -*-

import operator
import functools
import numpy as np
import tensorflow as tf

from app.settings import IMAGE_PATH, IMAGE_SHAPE, MODEL_PATH
from app.models.cnn import ConvolutionalNeuralNet
from app.pipeline import data_pipe, generate_data_skeleton

from .controller import generate_validation_set, train

m = functools.reduce(operator.mul, IMAGE_SHAPE[:2], 1)
d = 3
n = 8

g = tf.Graph()
sess = tf.Session()
# initiate a new graph, abandoning previous graph.

cnn = ConvolutionalNeuralNet(shape=(None, d, m))

x, _y, keep_prob = cnn.x, cnn._y, cnn.keep_prob
# (72, 128, 3)
conv_layer_1 = cnn.add_conv_layer(x, [[5, 5, 3, 12], [12]], func='sigmoid')
# (72, 128, 3 * 12)
conv_layer_2 = cnn.add_conv_layer(conv_layer_1, [[5, 5, 12, 24], [24]], func='relu')
# (72, 128, 3 * 24)
max_pool_1 = cnn.add_pooling_layer(conv_layer_2)
# (36, 64, 3 * 24)
conv_layer_3 = cnn.add_conv_layer(max_pool_1, [[5, 5, 24, 48], [48]], func='sigmoid')
# (36, 64, 3 * 48)
conv_layer_4 = cnn.add_conv_layer(conv_layer_3, [[5, 5, 48, 48], [48]], func='relu')
# (36, 64, 3 * 48)
max_pool_2 = cnn.add_pooling_layer(conv_layer_4)
# (18, 32, 3 * 48)
conv_layer_5 = cnn.add_conv_layer(max_pool_2, [[5, 5, 48, 96], [96]], func='sigmoid')
# (18, 32, 3 * 96)
conv_layer_6 = cnn.add_conv_layer(conv_layer_5, [[5, 5, 96, 96], [96]], func='relu')
# (18, 32, 3 * 96)
max_pool_3 = cnn.add_pooling_layer(conv_layer_6)
# (9, 16, 3 * 96)
fully_connected_layer_1 = cnn.add_dense_layer(
                            max_pool_3,
                            [[9 * 16 * 96, 1024], [1024], [-1, 9 * 16 * 96]],
                            func='relu'
                            )
# drop_out_layer_1 = cnn.add_drop_out_layer(fully_connected_layer_2)
read_out = cnn.add_read_out_layer(fully_connected_layer_1, [[1024, 8], [8]])

# train
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=_y, logits=read_out)
loss = tf.reduce_mean(cross_entropy)
train_step = tf.train.AdamOptimizer().minimize(loss)

# eval
correct_prediction = tf.equal(tf.argmax(read_out, 1), tf.argmax(_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# prepare data feed
train_file_array, train_label_array, valid_file_array, valid_label_array = \
        generate_data_skeleton(root_dir=IMAGE_PATH + 'train', valid_size=.15)

train_image_batch, train_label_batch = \
        data_pipe(train_file_array, train_label_array, num_epochs=None)
valid_image_batch, valid_label_batch = \
        data_pipe(valid_file_array, valid_label_array, num_epochs=1)

test_file_array, _ = \
        generate_data_skeleton(root_dir=IMAGE_PATH + 'test_stg1', valid_size=None)
test_image_batch, _ = \
        data_pipe(test_file_array, _, num_epochs=1, shuffle=False)

initializer = tf.group(
                    tf.local_variables_initializer(),
                    tf.global_variables_initializer()
                )

with sess:
    sess.run(initializer)
    whole_valid_images, whole_valid_labels = \
            generate_validation_set(sess, valid_image_batch, valid_label_batch)
    train(sess, x, _y, train_image_batch, train_label_batch, whole_valid_images,
                whole_valid_labels, train_step, accuracy, loss)
    # save_sess(sess, path=MODEL_PATH)
