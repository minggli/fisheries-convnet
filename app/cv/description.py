"""
description

Generate correctly formatted description file for opencv_createsamples.
Targeted positive images must have been cropped and have only one object as
this script will take assumption that from (0, 0) to (width, height), there is
the entirety of the single object.

April 3 Update
Thanks to https://www.kaggle.com/shuaiwang, there are bounding boxes manually
annotated by shuai for each of image in training set, which is allowed.
Using these bounding boxes to generate description file for OpenCV to learn to
detect in test set. It is against the competition rule to manually annotate
test set.
"""

import os
import json
import cv2

from app.pipeline import folder_traverse
from app.settings import CV_SAMPLE_PATH


def deserialize_json(rootdir, ext=('json')):
    """concatenate and deserialize bounding boxes in json format"""
    bb_file_structure = folder_traverse(rootdir, ext=ext)
    fish_annotation = list()
    for folder, filelist in bb_file_structure.items():
        for filename in filelist:
            with open(folder+filename) as f:
                label = json.load(f)
                fish_annotation.append(label)
    # individual json object from nested lists
    fish_annotation = {json_object['filename']: json_object for nested_list
                       in fish_annotation for json_object in nested_list}
    return fish_annotation


# producing positive samples with required description format by OpenCV
bb = deserialize_json(os.path.join(
                 os.path.dirname(os.path.realpath(__file__)), 'bb/'))
file_structure = folder_traverse(
                 os.path.join(os.path.realpath('.'), CV_SAMPLE_PATH + 'pos/'))
f = open(os.path.dirname(os.path.realpath(__file__)) + '/positives.dat', 'w')
for folder, filelist in file_structure.items():
    for filename in filelist:
        data = bb[filename]['annotations']
        path_to_image = '{0}{1} '.format(folder, filename)
        num_object = '{0}'.format(len(data))
        boundingbox_template = ' {0} {1} {2} {3}'
        bbs = ''
        endline = '\n'
        for i in range(int(num_object)):
            x = str(int(data[i]['x']))
            y = str(int(data[i]['y']))
            height = str(int(data[i]['height']))
            width = str(int(data[i]['width']))
            bbs += boundingbox_template.format(x, y, width, height)
        print_string = ''.join([path_to_image, num_object, bbs, endline])
        f.write(print_string)
f.close()


# produce negative set...simply list of negative images location in abspath
file_structure = folder_traverse(
                os.path.join(os.path.realpath('.'), CV_SAMPLE_PATH + 'neg/'))
f = open(os.path.dirname(os.path.realpath(__file__)) + '/negatives.dat', 'w')
for folder, filelist in file_structure.items():
    for filename in filelist:
        img = cv2.imread(folder+filename, -1)
        cv2.imwrite(folder+filename, img)
        string = '{0}{1}\n'.format(folder, filename)
        f.write(string)
f.close()
