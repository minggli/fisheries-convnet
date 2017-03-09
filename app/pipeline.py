#!/venv/bin python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn import preprocessing

from .settings import IMAGE_PATH


def image_resize_op(image, shape):
    """tensorflow op to resize and centrally crop image"""
    return tf.image.resize_image_with_crop_or_pad(image=image, target_height=shape[0], target_width=shape[1])


def file_system(directory):
    """map all image-only files in a folder"""
    file_structure = dict()
    for root, _, files in os.walk(directory):
        image_list = [i for i in files if i.endswith('.jpg')]
        if image_list:
            file_structure[root] = image_list
        else:
            continue
    return file_structure


def generate_data_skeleton(file_structure):
    """turn file structure into human-readable pandas dataframe"""
    reversed_fs = {k + '/' + f : k.split('/')[-1] 
    for k, v in file_structure.items() for f in v}
    df = pd.DataFrame.from_dict(data=reversed_fs, orient='index').reset_index()
    df.rename(columns={'index': 'filename', 0: 'labels'}, inplace=True)
    df.sort_values(by=['label', 'filename'], inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df['filename'].tolist(), df['labels'].tolist()


def read_image(paths_to_image, labels, num_epoch):
    """"""
    images = tf.convert_to_tensor(paths_to_image, dtype=dtype.string)
    labels = tf.convert_to_tensor(labels, dtype=dtype.string)
    input_queue = tf.train.slice_input_producer(
        string_tensor=[images, labels],
        num_epoch=num_epoch
        shuffle=True
        )
    return input_queue


# print(generate_data_skeleton(file_system(IMAGE_PATH)).ix[0, :]['filename'])

def transform(input_queue, shape, standardize=True):
    """a single transformation function that applies standardization with
    no mean centralisation. For sparsed data such as image data, mean
    centralisation is not suitable.
    """
    image = tf.read_file(input_queue[0])
    image_content = tf.image.decode_image(image, channels=None, name=None)

    label = input_queue[1]

    


    if pixels is not None:
        img = pd.DataFrame.from_dict(data=pixels, orient='index', dtype=np.float32)

    if normalize:
        data = data.apply(preprocessing.scale, with_mean=True, with_std=True, axis=0)
        if pixels is not None:
            img = img.apply(preprocessing.scale, with_mean=True, with_std=True, axis=0)

    margins = data.ix[:, data.columns.str.startswith('margin')]
    shapes = data.ix[:, data.columns.str.startswith('shape')]
    textures = data.ix[:, data.columns.str.startswith('texture')]


    if dim > 1:

        if label is not None and pixels is not None:
            transformed = \
                [(np.concatenate((margins.ix[i, :], shapes.ix[i, :], textures.ix[i, :], img.ix[i, :]), axis=0).reshape(dim, input_shape), label.ix[i, :]) for i in data.index]
        if label is not None and pixels is None:
            transformed = \
                [(np.concatenate((margins.ix[i, :], shapes.ix[i, :], textures.ix[i, :]), axis=0).reshape(dim, input_shape), label.ix[i, :]) for i in data.index]
        if label is None and pixels is not None:
            transformed = \
                [np.concatenate((margins.ix[i, :], shapes.ix[i, :], textures.ix[i, :], img.ix[i, :]), axis=0).reshape(dim, input_shape) for i in data.index]
        if label is None and pixels is None:
            transformed = \
                [np.concatenate((margins.ix[i, :], shapes.ix[i, :], textures.ix[i, :]), axis=0).reshape(dim, input_shape) for i in data.index]


    return np.array(transformed)