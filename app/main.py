# -*- coding: utf-8 -*-

import sys

CV_TRAIN = True if 'CV_TRAIN' in map(str.upper, sys.argv[1:]) else False
FETCH = True if 'FETCH' in map(str.upper, sys.argv[1:]) else False
EVAL = True if 'EVAL' in map(str.upper, sys.argv[1:]) else False

if __name__ == '__main__':
    if FETCH or CV_TRAIN:
        from .cv import cv
    else:
        if EVAL:
            from .engine import vgg16
