# -*- coding: utf-8 -*-

import multiprocessing as mp

from app.pipeline import generate_data_skeleton
from app.settings import IMAGE_PATH
from app.localizer import Localizer


with mp.Pool(4) as p:
    p.map(Localizer.localize,
          generate_data_skeleton(IMAGE_PATH + 'test_stg1')[0]
          )
