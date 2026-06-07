#!/bin/bash

images=( "2009_001255" )

ROOT=/usr/local/google/home/lcchen/data/pascal/VOCdevkit/VOC2012
GT=SegmentationClassAug_Visualization
IMG=JPEGImages

for img in ${images[@]}; do
  cp ${ROOT}/${IMG}/${img}.jpg ./img
  cp ${ROOT}/${GT}/${img}.png ./gt
done
