#!/usr/bin/env bash
# -*- coding: utf-8 -*-

opencv_createsamples -info positives.dat -vec pos.vec -w 40 -h 40 -num 3300
opencv_traincascade -data fishcascade -vec pos.vec -bg negatives.dat \
  -numStages 20 -minHitRate 0.99 -maxFalseAlarmRate 0.5 -numPos 1500 \
  -numNeg 400 -w 40 -h 40 -mode ALL -precalcValBufSize 1024\
  -precalcIdxBufSize 1024
