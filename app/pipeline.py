# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn import preprocessing

from settings import IMAGE_PATH, IMAGE_SHAPE, BATCH_SIZE


def folder_traverse(root_dir):
    """map all image-only files in a folder"""
    if not os.path.exists(root_dir):
        raise RuntimeError('directory doesn\'t exist')
    file_structure = dict()
    # using os.walk instead of new os.scandir for backward compatibility reason
    for root, _, files in os.walk(root_dir):
        image_list = [i for i in files if i.endswith('.jpg')]
        if image_list:
            file_structure[root] = image_list
    return file_structure


def generate_data_skeleton(file_structure):
    """turn file structure into human-readable pandas dataframe"""
    reversed_fs = {k + '/' + f : k.split('/')[-1]
        for k, v in file_structure.items() for f in v}
    df = pd.DataFrame.from_dict(data=reversed_fs, orient='index').reset_index()
    df.rename(columns={'index': 'filename', 0: 'labels'}, inplace=True)
    df.sort_values(by=['labels', 'filename'], inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df['filename'].tolist(), df['labels'].tolist()


def make_queue(paths_to_image, labels, num_epochs=None):
    """returns an Ops Tensor with queued image and label pair"""
    images = tf.convert_to_tensor(paths_to_image, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.string)
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
    image_content = tf.image.decode_image(image_queue, channels=shape[2])

    resized_image_content = tf.image.resize_image_with_crop_or_pad(
        image=image_content,
        target_height=shape[0],
        target_width=shape[1]
        )

    resized_image_content.set_shape(IMAGE_SHAPE)

    if standardize:
        # TODO how to standardize queued image data
        pass

    label_queue = input_queue[1]

    return resized_image_content, label_queue

def batch_generator(image, label, batch_size=BATCH_SIZE):
    return tf.train.batch([image, label], batch_size = batch_size)
