# -*- coding: utf-8 -*-
import os
import time
import tensorflow as tf


def multi_threading(func):
    """decorator using tensorflow threading ability."""
    def wrapper(*args, **kwargs):
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        func_output = func(*args, **kwargs)
        try:
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)
        except (tf.errors.CancelledError, RuntimeError) as e:
            pass
        return func_output
    return wrapper


def timeit(func):
    """calculate time for a function to complete"""

    def wrapper(*args, **kwargs):
        start = time.time()
        output = func(*args, **kwargs)
        end = time.time()
        print('function {0} took {1:0.3f} s'.format(func.__name__, (end - start) * 1))
        return output
    return wrapper


@multi_threading
@timeit
def train(n, sess, x, _y, keep_prob, train_image_batch, train_label_batch,
            valid_image_batch, valid_label_batch, optimiser, metric, loss):
    """train neural network and produce accuracies with validation set."""

    for global_step in range(n):
        train_image, train_label = sess.run([train_image_batch, train_label_batch])
        optimiser.run(feed_dict={x: train_image, _y: train_label, keep_prob: 0.5})
        print(global_step, train_label[0])

        if global_step % 10 == 0:
            valid_image, valid_label = \
                sess.run([valid_image_batch, valid_label_batch])
            training_accuracy, loss_score = \
                sess.run([metric, loss], feed_dict={x: valid_image,
                _y: valid_label, keep_prob: 1.0})
            print("step {0} of {3}, training accuracy: {1:.4f}, log loss: {2:.4f}".
                            format(global_step, training_accuracy, loss_score, n))


@multi_threading
@timeit
def predict(sess, x, keep_prob, logits, test_image_batch):
    """predict test set using graph previously trained and saved."""

    complete_probs = list()
    for _ in range(50):
        try:
            test_image = sess.run(test_image_batch)
            probs = sess.run(tf.nn.softmax(logits),
                                    feed_dict={x: test_image, keep_prob: 1.0})
            complete_probs.append(probs)
            for i in probs:
                print(i)
        except tf.errors.OutOfRangeError as e:
            # pipe exhausted with pre-determined number of epochs i.e. 1
            complete_probs = [list(data) for array in complete_probs for data in array]
            break
    input('press.')
    return complete_probs


@timeit
def submit(complete_probs, path):
    """"produce an output file with predicted probabilities."""

    import pandas as pd
    from datetime import datetime

    now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    template = pd.read_csv(
                filepath_or_buffer=path + 'sample_submission_stg1.csv',
                encoding='utf8',
                index_col=0)
    for i in complete_probs:
        print(i)
    df = pd.DataFrame(
                data=complete_probs,
                columns=template.columns,
                index=template.index,
                dtype=float)
    df.to_csv(
                path + 'submission_{0}.csv'.format(now),
                encoding='utf8',
                header=True,
                index=True)


@timeit
def restore_session(sess, path):
    """restore hard trained model for predicting."""
    eval_saver = tf.train.import_meta_graph(tf.train.latest_checkpoint(path) + '.meta')
    eval_saver.restore(sess, tf.train.latest_checkpoint(path))
    print('Restore successful.')


@timeit
def save_session(sess, path):
    """save hard trained model for future predicting."""
    from datetime import datetime

    now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    saver = tf.train.Saver()
    if not os.path.exists(path):
        os.makedirs(path)
    save_path = saver.save(sess, path + "model_{0}.ckpt".format(now))
    print("Model saved in: {0}".format(save_path))
