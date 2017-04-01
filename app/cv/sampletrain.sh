#!/usr/bin/env bash
# -*- coding: utf-8 -*-

opencv_createsamples -info positives.dat -vec pos_vector.vec -w 48 -h 36
opencv_traincascade -bg negatives.dat -data fishcascade -vec pos_vector.vec -numPos 30 -numNeg 100 -w 48 -h 36 -mode ALL
