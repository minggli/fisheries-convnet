#! ./venv/bin/python3 -m app.engine.core
# -*- coding: utf-8 -*-


import tensorflow as tf
from app.models.cnn import ConvolutionalNeuralNet as cnn

new_nn = cnn(shape=(None, 3, 8))
