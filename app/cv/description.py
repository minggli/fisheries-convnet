"""
description

Generate correctly formatted description file for opencv_createsamples.
Targeted positive images must have been cropped and have only one object as
this script will take assumption that from (0, 0) to (width, height), there is
the entirety of the single object.

Thanks to https://www.kaggle.com/shuaiwang, there are bounding boxes manually
annotated by shuai for each of image in training set. It is against rule to
annotate test set but it is allowed for training set annotation. Using these
bounding boxes to generate description file for OpenCV to train and generalize
in test set.
"""

import os
import cv2

from ..serializer import deserialize_json
from ..pipeline import folder_traverse
from ..settings import CV_SAMPLE_PATH, BOUNDINGBOX

# producing positive samples with required description format by OpenCV
bbox = deserialize_json(BOUNDINGBOX)
file_structure = folder_traverse(os.path.join(os.path.realpath('.'),
                                 CV_SAMPLE_PATH + 'pos/'))
f = open(os.path.dirname(os.path.realpath(__file__)) + '/positives.dat', 'w')
for folder, filelist in file_structure.items():
    for filename in filelist:
        data = bbox[filename]['annotations']
        path_to_image = '{0}{1} '.format(folder + '/', filename)
        num_object = '{0}'.format(len(data))
        boundingbox_template = ' {0} {1} {2} {3}'
        bboxes = ''
        endline = '\n'
        for i in range(int(num_object)):
            x = str(int(data[i]['x']))
            y = str(int(data[i]['y']))
            height = str(int(data[i]['height']))
            width = str(int(data[i]['width']))
            bboxes += boundingbox_template.format(x, y, width, height)
        print_string = ''.join([path_to_image, num_object, bboxes, endline])
        f.write(print_string)
f.close()


# produce negative set...simply list of negative images location in abspath
file_structure = folder_traverse(os.path.join(os.path.realpath('.'),
                                 CV_SAMPLE_PATH + 'neg/'))
f = open(os.path.dirname(os.path.realpath(__file__)) + '/negatives.dat', 'w')
for folder, filelist in file_structure.items():
    for filename in filelist:
        # use OpenCV I/O to make sure correct jpeg file
        img = cv2.imread(folder + '/' + filename, -1)
        cv2.imwrite(folder + '/' + filename, img)
        string = '{0}{1}\n'.format(folder, filename)
        f.write(string)
f.close()
