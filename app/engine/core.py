# -*- coding: utf-8 -*-

import operator
import functools
import numpy as np
import tensorflow as tf

from app.main import EVAL
from app.models.cnn import ConvolutionalNeuralNet
from app.settings import IMAGE_PATH, IMAGE_SHAPE, MODEL_PATH, MAX_STEPS
from app.pipeline import data_pipe, generate_data_skeleton

from .controller import generate_validation_set, train, save_session, predict, \
                        submit, restore_session

sess = tf.Session()
cnn = ConvolutionalNeuralNet(shape=(None, IMAGE_SHAPE[2],
                        functools.reduce(operator.mul, IMAGE_SHAPE[:2], 1)))

x, _y = cnn.x, cnn._y
keep_prob = tf.placeholder(tf.float32)
# (72, 128, 3)
conv_layer_1 = cnn.add_conv_layer(x, [[7, 7, 3, 24], [24]], func='relu')
# (72, 128, 12)
conv_layer_2 = cnn.add_conv_layer(conv_layer_1, [[7, 7, 24, 24], [24]], func='relu')
# (72, 128, 24)
max_pool_1 = cnn.add_pooling_layer(conv_layer_2)
# (36, 64, 24)
conv_layer_3 = cnn.add_conv_layer(max_pool_1, [[5, 5, 24, 48], [48]], func='relu')
# (36, 64, 48)
conv_layer_4 = cnn.add_conv_layer(conv_layer_3, [[5, 5, 48, 48], [48]], func='relu')
# (36, 64, 48)
max_pool_2 = cnn.add_pooling_layer(conv_layer_4)
# (18, 32, 48)
# conv_layer_5 = cnn.add_conv_layer(max_pool_2, [[3, 3, 48, 96], [96]], func='relu')
# # (18, 32, 96)
# conv_layer_6 = cnn.add_conv_layer(conv_layer_5, [[3, 3, 96, 96], [96]], func='relu')
# # (18, 32, 96)
# max_pool_3 = cnn.add_pooling_layer(conv_layer_6)
# (9, 16, 96)
fully_connected_layer_1 = cnn.add_dense_layer(
                            max_pool_2,
                            [[18 * 32 * 48, 2048], [2048], [-1, 18 * 32 * 48]],
                            func='relu'
                            )
# (1, 4096)
# fully_connected_layer_2 = cnn.add_dense_layer(
#                             max_pool_2,
#                             [[4096, 2048], [2048], [-1, 4096]],
#                             func='relu'
#                             )
# (1, 2048)
drop_out_layer_1 = cnn.add_drop_out_layer(fully_connected_layer_1, keep_prob)
# (1, 2048)
logits = cnn.add_read_out_layer(drop_out_layer_1, [[2048, 8], [8]])

# train
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=_y, logits=logits)
loss = tf.reduce_mean(cross_entropy)
train_step = tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(loss)

# eval
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# prepare data feed
train_file_array, train_label_array, valid_file_array, valid_label_array = \
        generate_data_skeleton(root_dir=IMAGE_PATH + 'train', valid_size=.1)
train_image_batch, train_label_batch = \
        data_pipe(train_file_array, train_label_array, num_epochs=None)
valid_image_batch, valid_label_batch = \
        data_pipe(valid_file_array, valid_label_array, num_epochs=1)

test_file_array, _ = \
        generate_data_skeleton(root_dir=IMAGE_PATH + 'test_stg1', valid_size=None)
test_image_batch, _ = \
        data_pipe(test_file_array, _, num_epochs=1, shuffle=False)

init_op = tf.group(
            tf.local_variables_initializer(), tf.global_variables_initializer())

if not EVAL:
    with sess:
        sess.run(init_op)

        train(MAX_STEPS, sess, x, _y, keep_prob, train_image_batch,
        train_label_batch, valid_image_batch, valid_label_batch, train_step,
        accuracy, loss)

        save_session(sess, path=MODEL_PATH)
elif EVAL:
    sess.run(init_op)
    with sess:
        restore_session(sess, MODEL_PATH)
        probs = predict(sess, x, keep_prob, logits, test_image_batch)
        submit(probs, IMAGE_PATH)
