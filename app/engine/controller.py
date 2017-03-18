# -*- coding: utf-8 -*-

import tensorflow as tf


def generate_validation_set(sess, valid_image_batch, valid_label_batch):
    """generate validation set from pipeline"""
    whole_valid_images = list()
    whole_valid_labels = list()
    for _ in range(20):
        try:
            valid_image, valid_label = sess.run([valid_image_batch, valid_label_batch])
            whole_valid_images.append(valid_image)
            whole_valid_labels.append(valid_label)
        except tf.errors.OutOfRangeError as e:
            # pipe exhausted with pre-determined number of epochs
            whole_valid_images = [data for array in whole_valid_images for data in array]
            whole_valid_labels = [data for array in whole_valid_labels for data in array]
            break
    return whole_valid_images, whole_valid_labels


def train(sess, optimiser, metric, loss):
    """train neural network with multi-threading data pipeline."""

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for global_step in range(2):
        train_image, train_label = sess.run([train_image_batch, train_label_batch])
        optimiser.run(feed_dict={x: train_image, _y: train_label})
        #, keep_prob: .5})

        if global_step % 5 == 0:
            valid_accuracy, loss_score = \
                sess.run(
                        [metric, loss],
                        feed_dict={x: whole_valid_images, _y: whole_valid_labels}
                                        # keep_prob: .5}
                        )
            print("step {0}, validation accuracy {1:.4f}, loss {2:.4f}".
                                format(global_step, valid_accuracy, loss_score))

    coord.request_stop()
    coord.join(threads)
