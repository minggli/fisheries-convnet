# -*- coding: utf-8 -*-
"""
fetchsamples

obtain positive and negative samples from ImageNet to train Haar Cascade.
"""

import os
import random
import shutil
import requests
import uuid

from multiprocessing import Pool
from itertools import repeat
from socket import timeout
from requests.exceptions import (ConnectTimeout, ConnectionError,
                                 ProtocolError, Timeout, ReadTimeoutError)

from ..controllers import timeit


@timeit
def generate_sample_skeleton(synset_dict, sample_size, base_url):
    """produces urls of images belonging to certain synset on ImageNet"""
    synset_urls = list()
    for key, wnid in synset_dict.items():
        try:
            r = requests.get(base_url.format(wnid),
                             allow_redirects=True,
                             timeout=5)
        except (ConnectTimeout, ConnectionError) as e:
            raise RuntimeError('no active Internet connection.')
        synset_urls.append(r.text.split('\r\n'))

    unravelled = [url for nested_set in synset_urls for url in nested_set
                  if 'http' in url and '.' in os.path.split(url)[1]]

    random.shuffle(unravelled)

    if 0 < sample_size <= 1:
        return random.sample(unravelled, int(sample_size * len(unravelled)))
    elif sample_size > 1:
        try:
            return random.sample(unravelled, int(sample_size))
        except ValueError:
            return unravelled


@timeit
def batch_retrieve(func, iterable, path):
    """processing through iterable (e.g. list)"""

    with Pool(4) as p:
        p.starmap(func, zip(iterable, repeat(path)))


def retrieve_image(image_url, path):
    """download single image and save in path"""
    try:
        r = requests.get(image_url,
                         allow_redirects=False,
                         timeout=5,
                         stream=True)
    except (timeout, Timeout, Exception, ProtocolError, ReadTimeoutError) as e:
        return None
    code = r.status_code
    print(code, image_url, flush=True)
    if code == 200:
        fname = str(uuid.uuid4()) + os.path.splitext(image_url)[1]
        with open(path + '/' + fname, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)
