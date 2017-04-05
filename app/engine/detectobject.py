# -*- coding: utf-8 -*-

import multiprocessing as mp

from ..pipeline import generate_data_skeleton
from ..settings import IMAGE_PATH
from ..localizer import Localizer


with mp.Pool(4) as p:
    p.map(Localizer.localize,
          generate_data_skeleton(IMAGE_PATH)[0])
