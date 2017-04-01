import os

from app.settings import HAARCASCADE, CV_SAMPLE_PATH

make = [HAARCASCADE, CV_SAMPLE_PATH + 'neg', CV_SAMPLE_PATH + 'pos']

for directory in make:
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass
