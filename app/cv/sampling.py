# -*- coding: utf-8 -*-
"""
sampling

obtain positive and negative samples from ImageNet to train Haar Cascade.
"""
import random
import requests

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
synset_urls_pos = dict()
synset_urls_neg = dict()


def generate_sample_skeleton(synset_dict, sample_size):
    """produces urls of images belonging to certain synset on ImageNet
    """
    synset_urls = list()
    for key, wnid in synset_dict.items():
        r = requests.get(BASE_URL.format(wnid))
        synset_urls.append(r.text.split('\r\n'))
    return synset_urls
