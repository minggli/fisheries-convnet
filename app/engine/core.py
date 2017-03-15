#! ./venv/bin/python3 -m app.engine.core
# -*- coding: utf-8 -*-


import tensorflow as tf
from app.models.cnn import ConvolutionalNeuralNet
from app.settings import IMAGE_PATH
from app.pipeline import pipe

g = tf.Graph()
sess = tf.Session()

test_image_folder = IMAGE_PATH + 'trial'

image_batch, label_batch = pipe(test_image_folder)


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


cnn = ConvolutionalNeuralNet(shape=(None, 3, 8))

IMAGE_SHAPE = (192, 192 , 3)

x, _y = cnn.x, cnn._y
# (192, 192, 3)
conv_layer_1 = cnn.add_conv_layer(x, [[5, 5, 3, 12], [12]], func='sigmoid')
# (192, 192, 3 * 12)
conv_layer_2 = cnn.add_conv_layer(conv_layer_1, [[5, 5, 12, 24], [24]], func='relu')
# (192, 192, 3 * 24)
max_pool_1 = con.add_pooling_layer(conv_layer_2)
# (96, 96, 3 * 24)
conv_layer_3 = cnn.add_conv_layer(max_pool_1, [[5, 5, 24, 48], [48]], func='sigmoid')
# (96, 96, 3 * 48)
conv_layer_4 = cnn.add_conv_layer(conv_layer_3, [[5, 5, 48, 48], [48]], func='relu')
# (96, 96, 3 * 48)
max_pool_2 = con.add_pooling_layer(conv_layer_4)
# (48, 48, 3 * 48)
conv_layer_5 = cnn.add_conv_layer(max_pool_2, [[5, 5, 48, 96], [96]], func='sigmoid')
# (48, 48, 3 * 96)
conv_layer_6 = cnn.add_conv_layer(conv_layer_5, [[5, 5, 96, 96], [96]], func='relu')
# (48, 48, 3 * 96)
max_pool_3 = con.add_pooling_layer(conv_layer_6)
# (24, 24, 3 * 96)
fully_connected_layer_1 = conn.add_dense_layer(
                            max_pool_3,
                            [[24 * 24 * 288, 2048], [2048], [-1, 24 * 24 * 288]],
                            func='sigmoid'
                            )
fully_connected_layer_2 = conn.add_dense_layer(
                            fully_connected_layer_1,
                            [[24 * 24 * 288, 1024], [1024], [-1, 24 * 24 * 288]],
                            func='relu'
                            )
drop_out_layer_1 = conn.add_drop_out_layer(fully_connected_layer_2)
logits = con.add_read_out_layer(drop_out_layer_1, [[1024, 8], [8]])

# train
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y_)
loss = tf.reduce_mean(cross_entropy)
train_step = tf.train.AdamOptimizer().minimize(loss)

# eval
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# miscellaneous
initializer = tf.global_variables_initializer()
