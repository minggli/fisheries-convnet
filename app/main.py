#! ./venv/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from PIL import Image

from pipeline import (file_system, generate_data_skeleton, make_queue,
                    decode_transform
                    )
from settings import IMAGE_PATH, IMAGE_SHAPE

test_image = IMAGE_PATH + 'trial'

session = tf.Session()

with session as sess:

    fs = file_system(test_image)
    print(fs)
    paths_to_images, labels = generate_data_skeleton(fs)
    print(paths_to_images, labels)
    # Below Ops are using Tensorflow
    queue = make_queue(paths_to_image=paths_to_images, labels=labels, num_epochs=3)

    resized_image_queue, label_queue = decode_transform(input_queue=queue)

    initializer = tf.global_variables_initializer()
    sess.run(initializer)

    Image.fromarray(np.array(resized_image_queue.eval())).show()
