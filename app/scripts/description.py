#!/usr/bin/env python3
"""
description

Generate correctly formatted description file for opencv_createsamples.
Targeted positive images must have been cropped and have only one object as
this script will take assumption that from (0, 0) to (width, height), there is
the entirety of the single object.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.realpath('..')))
from PIL import Image
from app.pipeline import folder_traverse
from app.settings import CV_CROPPED_SAMPLE_PATH

file_structure = folder_traverse(sys.path[0] + CV_CROPPED_SAMPLE_PATH[1:])


def get_size(img):
    """get image size width and height"""
    width, height = Image.open(img).size
    return width, height


f = open('positives.dat', 'w')
for folder, filelist in file_structure.items():
    for filename in filelist:
        w, h = get_size(img=folder+filename)
        string = '{0}{1} 1 0 0 {2} {3}\n'.format(folder, filename, w, h)
        f.write(string)
f.close()
