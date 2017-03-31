# -*- coding: utf-8 -*-
"""
sampling

obtain positive and negative samples from ImageNet to train Haar Cascade.
"""

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

for key, wnid in synset_ids_pos.items():
    r = requests.get(BASE_URL.format(wnid))
    synset_urls_pos[key] = r.text.split('\r\n')
    print('{0}: {1}'.format(key, len(synset_urls_pos[key])))

for key, wnid in synset_ids_neg.items():
    r = requests.get(BASE_URL.format(wnid))
    synset_urls_neg[key] = r.text.split('\r\n')
    print('{0}: {1}'.format(key, len(synset_urls_neg[key])))
