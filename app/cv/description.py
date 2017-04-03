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
import cv2

from app.pipeline import folder_traverse, generate_data_skeleton
from app.settings import CV_CROPPED_SAMPLE_PATH, CV_SAMPLE_PATH

bb_file_array, _ = generate_data_skeleton(
                        os.path.dirname(os.path.realpath(__file__)) + 'bb/')


file_structure = folder_traverse(
                os.path.join(os.path.realpath('.'), CV_SAMPLE_PATH + 'pos/'))

f = open(os.path.dirname(os.path.realpath(__file__)) + '/positives.dat', 'w')

for folder, filelist in file_structure.items():
    for filename in filelist:
        h, w = cv2.imread(folder+filename, -1).shape[:2]
        string = '{0}{1} 1 0 0 {2} {3}\n'.format(folder, filename, w, h)
        f.write(string)
f.close()

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
