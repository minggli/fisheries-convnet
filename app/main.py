#! ./venv/bin python3 -m
# -*- coding: utf-8 -*-

import tensorflow as tf
from app.engine.core import sess

if __name__ == '__main__':

    with sess:
        initializer = tf.global_variables_initializer()
        sess.run(initializer)

        print('Hello, World')
