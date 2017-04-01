#!/usr/bin/env bash
# -*- coding: utf-8 -*-

opencv_createsamples -info positives.dat -vec pos.vec -w 48 -h 24
opencv_traincascade -bg negatives.dat -data fishcascade -vec pos.vec -numPos 10 -numNeg 100 -w 48 -h 24 -mode ALL
