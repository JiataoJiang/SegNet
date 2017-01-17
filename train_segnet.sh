#!/bin/bash
gpuid=5
#weight=caffemodel/dep.100000.caffemodel,caffemodel/rgb.100000.caffemodel

LOG="segnet.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./caffe-segnet/build/tools/caffe train \
    --gpu $gpuid \
    --solver Models/segnet_basic_solver.prototxt \

