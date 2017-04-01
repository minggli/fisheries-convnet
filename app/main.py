# -*- coding: utf-8 -*-

import sys

FETCH = True if 'FETCH' in map(str.upper, sys.argv[1:]) else False
EVAL = True if 'EVAL' in map(str.upper, sys.argv[1:]) else False

if __name__ == '__main__':
    if FETCH:
        from .cv import cv
    else:
        if EVAL:
            from .engine import vgg16
