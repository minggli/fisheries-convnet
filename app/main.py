#! ./venv/bin/python3 -m app.main
# -*- coding: utf-8 -*-

import sys
import tensorflow as tf
import numpy as np

from PIL import Image

from app.pipeline import data_pipe
from app.settings import IMAGE_PATH

EVAL = True if 'EVAL' in map(str.upper, sys.argv[1:]) else False

if __name__ == '__main__':
    print('Hello, World.')
