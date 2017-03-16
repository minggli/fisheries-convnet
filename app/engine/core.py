#! ./venv/bin/python3 -m app.main
# -*- coding: utf-8 -*-

import operator
import functools
import tensorflow as tf

from app.models.cnn import ConvolutionalNeuralNet
from app.settings import IMAGE_PATH, IMAGE_SHAPE
from app.pipeline import datapipe

m = functools.reduce(operator.mul, IMAGE_SHAPE[:2], 1)
d = 3
n = 8

g = tf.Graph()
sess = tf.Session()

cnn = ConvolutionalNeuralNet(shape=(None, d, m))

x_placeholder, y_placeholder, keep_prob = cnn.x, cnn._y, cnn.keep_prob
# (72, 128, 3)
conv_layer_1 = cnn.add_conv_layer(x_placeholder, [[5, 5, 3, 12], [12]], func='sigmoid')
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
                            [[9 * 16 * 288, 2048], [2048], [-1, 9 * 16 * 288]],
                            func='sigmoid'
                            )
fully_connected_layer_2 = cnn.add_dense_layer(
                            fully_connected_layer_1,
                            [[9 * 16 * 288, 1024], [1024], [-1, 9 * 16 * 288]],
                            func='relu'
                            )

drop_out_layer_1 = cnn.add_drop_out_layer(fully_connected_layer_2)

read_out = cnn.add_read_out_layer(drop_out_layer_1, [[1024, 8], [8]])

# train
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_placeholder, logits=read_out)
loss = tf.reduce_mean(cross_entropy)
train_step = tf.train.AdamOptimizer().minimize(loss)

# eval
correct_prediction = tf.equal(tf.argmax(read_out, 1), tf.argmax(y_placeholder, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

initializer = tf.global_variables_initializer()
