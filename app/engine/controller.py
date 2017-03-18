# -*- coding: utf-8 -*-
import os
import tensorflow as tf


def multi_threading(func):
    def wrapper(*args, **kwargs):
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        func_output = func(*args, **kwargs)
        coord.request_stop()
        coord.join(threads)
        return func_output
    return wrapper


@multi_threading
def generate_validation_set(sess, valid_image_batch, valid_label_batch):
    """generate validation set from pipeline as we need the entirety of
    validation data to check model training progress.
    """
    whole_valid_images = list()
    whole_valid_labels = list()
    for _ in range(20):
        try:
            valid_image, valid_label = sess.run([valid_image_batch, valid_label_batch])
            whole_valid_images.append(valid_image)
            whole_valid_labels.append(valid_label)
        except tf.errors.OutOfRangeError as e:
            # pipe exhausted with pre-determined number of epochs i.e. 1
            whole_valid_images = [data for array in whole_valid_images for data in array]
            whole_valid_labels = [data for array in whole_valid_labels for data in array]
            break
    return whole_valid_images, whole_valid_labels


@multi_threading
def train(sess, x, _y, train_image_batch, train_label_batch, whole_valid_images,
            whole_valid_labels, optimiser, metric, loss):
    """train neural network and produce accuracies with validation set."""

    for global_step in range(10):
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

@multi_threading
def predict(sess, x, keep_prob, read_out, test_image_batch, path, test):
    """predict test set using graph previously trained and saved."""
    eval_saver = tf.train.import_meta_graph(path + 'model.ckpt.meta')
    eval_saver.restore(save_path=path + 'model.ckpt', sess=sess)
    complete_probs = list()

    for _ in range(20):
        try:
            probs = sess.run(tf.nn.softmax(read_out),
                feed_dict={x: test_image_batch, keep_prob: 1.0}
                )
            complete_probs.append(probs)
        except tf.errors.OutOfRangeError as e:
            # pipe exhausted with pre-determined number of epochs i.e. 1
            complete_probs = [data for array in complete_probs for data in array]
            break

    return complete_probs


def submit(complete_probs, path):
    """"produce an output file with predicted probabilities."""

    from datetime import datetime

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    template = pd.read_csv(
                filepath_or_buffer=path + 'sample_submission_stg1.csv',
                encoding='utf8',
                header=True,
                index_col='image'
        )
    df = pd.DataFrame(
                data=complete_probs,
                columns=template.columns,
                dtype=np.float32,
                index=template.index
        )
    df.to_csv(
                path + 'submission_{0}.csv'.format(now),
                encoding='utf8',
                header=True,
                index=True
        )


def save_sess(sess, path):
    """save hard trained model for predicting."""
    saver = tf.train.Saver()
    if not os.path.exists(path):
        os.makedirs(path)
    save_path = saver.save(sess, path + "model.ckpt")
    print("Model saved in: {0}".format(save_path))
