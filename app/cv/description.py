"""
description

Generate correctly formatted description file for opencv_createsamples.
Targeted positive images must have been cropped and have only one object as
this script will take assumption that from (0, 0) to (width, height), there is
the entirety of the single object.
"""

from PIL import Image

from app.pipeline import folder_traverse
from app.settings import CV_CROPPED_SAMPLE_PATH, CV_SAMPLE_PATH, HAARCASCADE

file_structure = folder_traverse(CV_CROPPED_SAMPLE_PATH)
f = open(HAARCASCADE + 'positives.dat', 'w')
for folder, filelist in file_structure.items():
    for filename in filelist:
        w, h = Image.open(folder+filename).size
        string = '{0}{1} 1 0 0 {2} {3}\n'.format(folder, filename, w, h)
        f.write(string)
f.close()

file_structure = folder_traverse(CV_SAMPLE_PATH + 'neg/')
f = open(HAARCASCADE + 'negatives.dat', 'w')
for folder, filelist in file_structure.items():
    for filename in filelist:
        string = '{0}{1}\n'.format(folder, filename)
        f.write(string)
f.close()