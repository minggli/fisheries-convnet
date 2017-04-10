#!/usr/bin/env bash
# -*- coding: utf-8 -*-


export SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

cd $SCRIPT_DIR
opencv_createsamples -info positives.dat -vec pos.vec -w 40 -h 40 -num 3300
opencv_traincascade -data fishcascade -vec pos.vec -bg negatives.dat \
  -numStages 10 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 500 \
  -numNeg 3000 -w 40 -h 40 -mode ALL -precalcValBufSize 1024\
  -precalcIdxBufSize 1024
