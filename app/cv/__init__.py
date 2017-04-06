import os

from ..settings import HAARCASCADE, CV_IM_SAMPLE_PATH

make = [HAARCASCADE, CV_IM_SAMPLE_PATH + 'neg', CV_IM_SAMPLE_PATH + 'pos']

for directory in make:
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass
