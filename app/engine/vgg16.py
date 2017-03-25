# -*- coding: utf-8 -*-

import tensorflow as tf

from app.main import EVAL
from app.models.cnn import ConvolutionalNeuralNet
from app.settings import (IMAGE_PATH, IMAGE_SHAPE, MODEL_PATH, MAX_STEPS,
                          ALPHA, BETA)
from app.pipeline import data_pipe, generate_data_skeleton
from app.controllers import (train, save_session, predict, submit,
                             restore_session)


tf.set_random_seed(7)
sess = tf.Session()

flattened_shape = (None,
                   IMAGE_SHAPE[2],
                   IMAGE_SHAPE[0] * IMAGE_SHAPE[1])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x):
    """max pooling with kernal size 2x2 and slide by 2 pixels each time"""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


x = tf.reshape(
    tf.placeholder(dtype=tf.float32, shape=flattened_shape, name='feature'),
    (-1, ) + IMAGE_SHAPE)

_y = tf.placeholder(dtype=tf.float32, shape=[None, 8], name='label')

# keep prob seems to behave differet from normal variables
keep_prob = tf.placeholder(tf.float32)
# (90, 160, 3)

with tf.name_scope('hidden_layer_1'):
    W_conv1 = weight_variable([3, 3, 3, 6])
    b_conv1 = bias_variable([6])

    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)


with tf.name_scope('hidden_layer_2'):
    W_conv2 = weight_variable([3, 3, 6, 6])
    b_conv2 = bias_variable([6])

    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
    h_pool1 = max_pool(h_conv2)

# (45, 80, *)

with tf.name_scope('hidden_layer_3'):
    W_conv3 = weight_variable([3, 3, 6, 12])
    b_conv3 = bias_variable([12])

    h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)

with tf.name_scope('hidden_layer_4'):
    W_conv4 = weight_variable([3, 3, 12, 12])
    b_conv4 = bias_variable([12])

    h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
    h_pool2 = max_pool(h_conv4)

# (23, 40, *)

with tf.name_scope('hidden_layer_5'):
    W_conv5 = weight_variable([3, 3, 12, 24])
    b_conv5 = bias_variable([24])

    h_conv5 = tf.nn.relu(conv2d(h_pool2, W_conv5) + b_conv5)

with tf.name_scope('hidden_layer_6'):
    W_conv6 = weight_variable([3, 3, 24, 24])
    b_conv6 = bias_variable([24])

    h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)

with tf.name_scope('hidden_layer_7'):
    W_conv7 = weight_variable([3, 3, 24, 24])
    b_conv7 = bias_variable([24])

    h_conv7 = tf.nn.relu(conv2d(h_conv6, W_conv7) + b_conv7)
    h_pool3 = max_pool(h_conv7)

# (12, 20, *)
with tf.name_scope('hidden_layer_8'):
    W_conv8 = weight_variable([3, 3, 24, 48])
    b_conv8 = bias_variable([48])

    h_conv8 = tf.nn.relu(conv2d(h_pool3, W_conv8) + b_conv8)

with tf.name_scope('hidden_layer_9'):
    W_conv9 = weight_variable([3, 3, 48, 48])
    b_conv9 = bias_variable([48])

    h_conv9 = tf.nn.relu(conv2d(h_conv8, W_conv9) + b_conv9)

with tf.name_scope('hidden_layer_10'):
    W_conv10 = weight_variable([3, 3, 48, 48])
    b_conv10 = bias_variable([48])

    h_conv10 = tf.nn.relu(conv2d(h_conv9, W_conv10) + b_conv10)
    h_pool4 = max_pool(h_conv10)

# (6, 10, *)
with tf.name_scope('hidden_layer_11'):
    W_conv11 = weight_variable([3, 3, 48, 48])
    b_conv11 = bias_variable([48])

    h_conv11 = tf.nn.relu(conv2d(h_pool4, W_conv11) + b_conv11)

with tf.name_scope('hidden_layer_12'):
    W_conv12 = weight_variable([3, 3, 48, 48])
    b_conv12 = bias_variable([48])

    h_conv12 = tf.nn.relu(conv2d(h_conv11, W_conv12) + b_conv12)

with tf.name_scope('hidden_layer_13'):
    W_conv13 = weight_variable([3, 3, 48, 48])
    b_conv13 = bias_variable([48])

    h_conv13 = tf.nn.relu(conv2d(h_conv12, W_conv13) + b_conv13)
    h_pool5 = max_pool(h_conv13)

# (3, 5, *)
with tf.name_scope('dense_conn_1'):
    W_fc1 = weight_variable([3 * 5 * 48, 2048])
    b_fc1 = bias_variable([2048])

    h_pool4_flat = tf.reshape(h_pool2, [-1, 3 * 5 * 48])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

with tf.name_scope('dense_conn_2'):
    W_fc2 = weight_variable([2048, 1024])
    b_fc2 = bias_variable([1024])

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

with tf.name_scope('drop_out_layer_1'):
    h_drop = tf.nn.dropout(h_fc2, keep_prob)

with tf.name_scope('read_out'):
    W_fc3 = weight_variable([1024, 8])
    b_fc3 = bias_variable([8])

    logits = tf.matmul(h_drop, W_fc3) + b_fc3

# applying label weights to loss function
class_weight = tf.constant([[0.544876886, 0.947047922, 0.969023034,
                             0.982261054, 0.876886418, 0.920836643,
                             0.953402171, 0.805665872]])

# loss function
cross_entropy = \
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=_y)
loss = tf.reduce_mean(cross_entropy)

# weighted loss per class
# weight_per_label = tf.transpose(tf.matmul(_y, tf.transpose(class_weight)))
# loss = tf.reduce_mean(tf.multiply(weight_per_label, cross_entropy))

# add L2 regularization on weights from readout layer
# out_weights = [var for var in tf.trainable_variables()
#                if var.name == 'Variable_30:0'][0]
# regularizer = tf.nn.l2_loss(out_weights)
# loss = tf.reduce_mean(loss + BETA * regularizer)

# train Ops
train_step = tf.train.RMSPropOptimizer(learning_rate=ALPHA).minimize(loss)

# eval
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


if not EVAL:
    # prepare data feed
    train_file_array, train_label_array, valid_file_array, valid_label_array =\
        generate_data_skeleton(root_dir=IMAGE_PATH + 'train', valid_size=.15)
    train_image_batch, train_label_batch = data_pipe(
                                            train_file_array,
                                            train_label_array,
                                            num_epochs=None,
                                            shuffle=True)
    valid_image_batch, valid_label_batch = data_pipe(
                                            valid_file_array,
                                            valid_label_array,
                                            num_epochs=None,
                                            shuffle=True)

    init_op = tf.group(tf.local_variables_initializer(),
                       tf.global_variables_initializer())
    sess.run(init_op)

    with sess:
        train(MAX_STEPS, sess, x, _y, keep_prob, train_image_batch,
              train_label_batch, valid_image_batch, valid_label_batch,
              train_step, accuracy, loss)
        save_session(sess, path=MODEL_PATH)
    del sess

elif EVAL:

    test_file_array, _ = \
        generate_data_skeleton(root_dir=IMAGE_PATH + 'test_stg1',
                               valid_size=None)
    # no shuffling or more than 1 epoch of test set, only through once.
    test_image_batch, _ = data_pipe(
                            test_file_array,
                            _,
                            num_epochs=1,
                            shuffle=False)

    init_op = tf.group(tf.local_variables_initializer(),
                       tf.global_variables_initializer())
    sess.run(init_op)

    with sess:
        restore_session(sess, MODEL_PATH)
        probs = predict(sess, x, keep_prob, logits, test_image_batch)
        input('press to produce submission.')
        submit(probs, IMAGE_PATH)
    del sess
