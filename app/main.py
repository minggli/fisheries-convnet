# -*- coding: utf-8 -*-

import sys

EVAL = True if 'EVAL' in map(str.upper, sys.argv[1:]) else False

if __name__ == '__main__':
    from .engine import vgg16
