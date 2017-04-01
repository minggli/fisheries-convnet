import os

from app.settings import HAARCASCADE

if not os.path.exists(HAARCASCADE):
    os.makedirs(HAARCASCADE)
