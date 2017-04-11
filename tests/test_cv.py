# -*- coding: utf-8 -*-
"""
test_cv

using pre-trained front face and eye detector to identify a totally random face
"""
import cv2
import random
from app.settings import CV_SAMPLE_PATH, HAARCASCADE, HAARPARAMS
from app.pipeline import generate_data_skeleton

# random.seed(3)
file_arary = generate_data_skeleton(CV_SAMPLE_PATH + 'pos')[0]

cascade = cv2.CascadeClassifier(HAARCASCADE + 'cascade.xml')

# reads in image in BGR rather than RGB format
img = cv2.imread(random.choice(file_arary), -1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

fish = cascade.detectMultiScale(gray,
                                scaleFactor=HAARPARAMS['scaleFactor'],
                                minNeighbors=HAARPARAMS['minNeighbors'],
                                minSize=HAARPARAMS['minSize'],
                                maxSize=HAARPARAMS['maxSize']
                                )

for (x, y, w, h) in fish:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
