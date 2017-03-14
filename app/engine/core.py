#! ./venv/bin/python3 -m app.engine.core
# -*- coding: utf-8 -*-


import tensorflow as tf
from app.models.cnn import ConvolutionalNeuralNet as cnn

g = tf.Graph()
sess = tf.Session()

#
# default = {
#         'hidden_layer_1': [[5, 5, d, 32], [32]],
#         'hidden_layer_2': [[5, 5, 32, 64], [64]],
#         'dense_conn_1': [[2 * 2 * 64, 1024], [1024], [-1, 2 * 2 * 64]],
#         'read_out': [[1024, n], [n]],
#         'alpha': 5e-5,
#         'test_size': .20,
#         'batch_size': 200,
#         'num_epochs': 5000,
#         'drop_out': [.4, .5]
#     }


cnn = cnn(shape=(None, 3, 8))
x = None
conv_layer_1 = cnn.add_conv_layer(x, [[5, 5, 3, 64], [64]], func='relu')
conv_layer_2 = cnn.add_conv_layer()
