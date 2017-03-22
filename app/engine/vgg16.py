# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf

from app.main import EVAL
from app.models.cnn import ConvolutionalNeuralNet
from app.settings import IMAGE_PATH, IMAGE_SHAPE, MODEL_PATH, MAX_STEPS, ALPHA
from app.pipeline import data_pipe, generate_data_skeleton
from app.controllers import train, save_session, predict, submit, restore_session

sess = tf.Session()
cnn = ConvolutionalNeuralNet(shape=IMAGE_SHAPE)

x, _y = cnn.x, cnn._y
# keep prob seems to behave differet from normal variables
keep_prob = tf.placeholder(tf.float32)
# (90, 160, 3)
conv_layer_1 = cnn.add_conv_layer(x, [[3, 3, 3, 12], [12]], func='relu')

conv_layer_2 = cnn.add_conv_layer(conv_layer_1, [[3, 3, 12, 12], [12]], func='relu')

max_pool_1 = cnn.add_pooling_layer(conv_layer_2)
# (45, 80, *)
conv_layer_3 = cnn.add_conv_layer(max_pool_1, [[3, 3, 12, 24], [24]], func='relu')

conv_layer_4 = cnn.add_conv_layer(conv_layer_3, [[3, 3, 24, 24], [24]], func='relu')

max_pool_2 = cnn.add_pooling_layer(conv_layer_4)
# (23, 40, *)
conv_layer_5 = cnn.add_conv_layer(max_pool_2, [[3, 3, 24, 48], [48]], func='relu')

conv_layer_6 = cnn.add_conv_layer(conv_layer_5, [[3, 3, 48, 48], [48]], func='relu')

conv_layer_7 = cnn.add_conv_layer(conv_layer_6, [[3, 3, 48, 48], [48]], func='relu')

max_pool_3 = cnn.add_pooling_layer(conv_layer_7)
# (12, 20, *)
conv_layer_8 = cnn.add_conv_layer(max_pool_3, [[3, 3, 48, 96], [96]], func='relu')

conv_layer_9 = cnn.add_conv_layer(conv_layer_8, [[3, 3, 96, 96], [96]], func='relu')

conv_layer_10 = cnn.add_conv_layer(conv_layer_9, [[3, 3, 96, 96], [96]], func='relu')

max_pool_4 = cnn.add_pooling_layer(conv_layer_10)
# (6, 10, *)
conv_layer_11 = cnn.add_conv_layer(max_pool_4, [[3, 3, 96, 96], [96]], func='relu')

conv_layer_12 = cnn.add_conv_layer(conv_layer_11, [[3, 3, 96, 96], [96]], func='relu')

conv_layer_13 = cnn.add_conv_layer(conv_layer_12, [[3, 3, 96, 96], [96]], func='relu')

max_pool_4 = cnn.add_pooling_layer(conv_layer_13)
# (3, 5, *)
fully_connected_layer_1 = cnn.add_dense_layer(
                            max_pool_4,
                            [[2 * 3 * 96, 1024], [1024], [-1, 2 * 3 * 96]],
                            func='relu'
                            )
# drop_out_layer_1 = cnn.add_drop_out_layer(fully_connected_layer_1, keep_prob)
fully_connected_layer_2 = cnn.add_dense_layer(
                            fully_connected_layer_1,
                            [[1024, 512], [1024], [-1, 1024]],
                            func='relu'
                            )
# (1, 1024)
# drop out layer halves training accuracy from unseen valid set
# drop_out_layer_2 = cnn.add_drop_out_layer(fully_connected_layer_2, keep_prob)
# (1, 1024)
logits = cnn.add_read_out_layer(fully_connected_layer_2, [[512, 8], [8]])

# train
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=_y)
loss = tf.reduce_mean(cross_entropy)
# loss = tf.losses.softmax_cross_entropy(onehot_labels=_y, logits=logits)
train_step = tf.train.RMSPropOptimizer(learning_rate=ALPHA).minimize(loss)

# eval
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

if not EVAL:

    # prepare data feed
    train_file_array, train_label_array, valid_file_array, valid_label_array = \
            generate_data_skeleton(root_dir=IMAGE_PATH + 'train', valid_size=.2)
    train_image_batch, train_label_batch = \
            data_pipe(train_file_array, train_label_array, num_epochs=None, shuffle=True)
    valid_image_batch, valid_label_batch = \
            data_pipe(valid_file_array, valid_label_array, num_epochs=None, shuffle=True)

    init_op = tf.group(
                        tf.local_variables_initializer(),
                        tf.global_variables_initializer()
                    )

    sess.run(init_op)

    with sess:
        train(MAX_STEPS, sess, x, _y, keep_prob, train_image_batch,
        train_label_batch, valid_image_batch, valid_label_batch, train_step,
        accuracy, loss)
        save_session(sess, path=MODEL_PATH)
        sess.close()

elif EVAL:

    test_file_array, _ = \
            generate_data_skeleton(root_dir=IMAGE_PATH + 'test_stg1', valid_size=None)
    # there is no shuffling or more than 1 epoch of test set, only through once.
    test_image_batch, _ = \
            data_pipe(test_file_array, _, num_epochs=1, shuffle=False)

    init_op = tf.group(
                        tf.local_variables_initializer(),
                        tf.global_variables_initializer()
                    )

    sess.run(init_op)

    with sess:
        restore_session(sess, MODEL_PATH)
        probs = predict(sess, x, keep_prob, logits, test_image_batch)
        submit(probs, IMAGE_PATH)
        sess.close()
