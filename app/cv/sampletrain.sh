#!/usr/bin/env bash
# -*- coding: utf-8 -*-

opencv_createsamples -info positives.dat -vec pos.vec -w 96 -h 72
opencv_traincascade -data fishcascade -vec pos.vec -bg negatives.dat \
  -numStages 20 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 10 \
  -numNeg 100 -w 96 -h 72 -mode ALL -precalcValBufSize 1024\
  -precalcIdxBufSize 1024
