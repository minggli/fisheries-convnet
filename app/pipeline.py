#! ./venv/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn import preprocessing, model_selection

from .settings import IMAGE_PATH, IMAGE_SHAPE, BATCH_SIZE


def folder_traverse(root_dir):
    """map all image-only files in a folder"""
    if not os.path.exists(root_dir):
        raise RuntimeError('{0} doesn\'t exist.'.format(root_dir))
    file_structure = dict()
    # using os.walk instead of new os.scandir for backward compatibility reason
    for root, _, files in os.walk(root_dir):
        image_list = [i for i in files if i.endswith('.jpg')]
        if image_list:
            file_structure[root] = image_list
    return file_structure


def generate_data_skeleton(file_structure, test_size=None):
    """turn file structure into human-readable pandas dataframe"""
    reversed_fs = {k + '/' + f : k.split('/')[-1]
        for k, v in file_structure.items() for f in v}
    df = pd.DataFrame.from_dict(data=reversed_fs, orient='index').reset_index()
    df.rename(columns={'index': 'filename', 0: 'species'}, inplace=True)
    df.sort_values(by=['species', 'filename'], inplace=True)
    df.reset_index(inplace=True, drop=True)
    df['labels'] = pd.Categorical(df['species']).codes

    X, y = np.array(df['filename']), np.array(df['labels'])

    if test_size:
        sss = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        train_index, test_index = zip(*sss.split(X, y))
        return X[train_index], y[train_index], X[test_index], y[test_index]
    else:
        return X, y, None, None


def make_queue(paths_to_image, labels, num_epochs=None):
    """returns an Ops Tensor with queued image and label pair"""
    images = tf.convert_to_tensor(paths_to_image, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.uint8)
    input_queue = tf.train.slice_input_producer(
        tensor_list=[images, labels],
        num_epochs=num_epochs,
        shuffle=True
        )
    return input_queue


def decode_transform(input_queue, shape=IMAGE_SHAPE, standardize=True):
    """a single decode and transform function that applies standardization with
    no mean centralisation. Sparsed data such as image data, mean
    centralisation is not suited.
    """
    image_queue = tf.read_file(input_queue[0])
    original_image = tf.image.decode_image(image_queue, channels=shape[2])

    # crop larger images (e.g. 1280*974) to 1280*720, this func doesn't resize.
    cropped_image_content = tf.image.resize_image_with_crop_or_pad(
        image=original_image,
        target_height=shape[0]*10,
        target_width=shape[1]*10
        )

    # resize cropped images to desired shape
    resize_image_content = tf.image.resize_images(
        images = cropped_image_content,
        size = [shape[0], shape[1]]
        )

    resize_image_content.set_shape(shape)

    label_queue = input_queue[1]
    one_hot_label_queue = tf.one_hot(
        indices = label_queue,
        depth = 8,
        on_value = 1,
        off_value = 0
        )

    if standardize:
        # TODO how to standardize queued image data
        pass

    return resize_image_content, one_hot_label_queue


def batch_generator(image, label, batch_size=BATCH_SIZE):
    return tf.train.batch([image, label], batch_size = batch_size)


def datapipe(root_dir, test_size=None):
    train_images_array, train_label_array, test_images_array, test_label_array = \
        generate_data_skeleton(folder_traverse(root_dir), test_size=test_size)

    train_resized_image_queue, train_label_queue = \
        decode_transform(make_queue(train_images_array, train_label_array))
    train_image_batch, train_label_batch = \
        batch_generator(train_resized_image_queue, train_label_queue)

    if test_size:
        test_resized_image_queue, test_label_queue = \
            decode_transform(make_queue(test_images_array, test_label_array))
        test_image_batch, test_label_batch = \
            batch_generator(test_resized_image_queue, test_label_queue)
        return train_image_batch, train_label_batch, test_image_batch, test_label_batch
    else:
        return train_image_batch, train_label_batch
