# -*- coding: utf-8 -*-
"""
sampling

obtain positive and negative samples from ImageNet to train Haar Cascade.
"""

import requests
import os
import random
import shutil
import numpy as np

from app.controllers import timeit
from app.settings import CV_SAMPLE_PATH

np.random.seed(1)
BASE_URL = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid={0}'

synset_ids_pos = {
                    'Tuna_Bluefin': 'n02627292',
                    'Tuna_Yellowfin': 'n02627532',
                    'Tuna_Albacore': 'n02627037',
                    'DOL': 'n02581957',
                    'LAG': 'n02545841',
                    'SHA': 'n01484285',
                    'OTHER': 'n02512053'
}
synset_ids_neg = {
                    'ocean': 'n09376198',
                    'people': 'n07942152',
                    'poop deck': 'n03982642'
}


@timeit
def generate_sample_skeleton(synset_dict, sample_size):
    """produces urls of images belonging to certain synset on ImageNet"""
    synset_urls = list()
    for key, wnid in synset_dict.items():
        try:
            r = requests.get(BASE_URL.format(wnid),
                             allow_redirects=True,
                             timeout=5)
        except requests.exceptions.ConnectionError:
            raise RuntimeError('no active Internet connection.')
        synset_urls.append(r.text.split('\r\n'))

    unravelled = [url for nested_set in synset_urls for url in nested_set
                  if 'http' in url and '.' in url.split('/')[-1]]

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
    from multiprocessing import Pool
    from itertools import repeat

    with Pool(4) as p:
        p.starmap(func, zip(iterable, repeat(path)))


def retrieve_image(image_url, path):
    """download single image and save in path"""
    try:
        r = requests.get(image_url,
                         allow_redirects=False,
                         timeout=5,
                         stream=True)
    except (requests.exceptions.ConnectTimeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout,
            requests.packages.urllib3.exceptions.ReadTimeoutError) as e:
        return None
    code = r.status_code
    print(code, image_url, flush=True)
    if code == 200:
        fname = image_url.split('/')[-1]
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path + '/' + fname, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)


sample_pos = generate_sample_skeleton(synset_ids_pos, sample_size=1000)
sample_neg = generate_sample_skeleton(synset_ids_neg, sample_size=5000)

print(len(sample_pos))
print(len(sample_neg))

batch_retrieve(func=retrieve_image,
               iterable=sample_neg,
               path=CV_SAMPLE_PATH + 'neg')
batch_retrieve(func=retrieve_image,
               iterable=sample_pos,
               path=CV_SAMPLE_PATH + 'pos')
