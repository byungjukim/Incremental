#!/bin/bash

nvidia-docker run -v `pwd`:`pwd` -v /home/user/dataset/:/home/user/dataset/ -w `pwd` -it --name $1 gcr.io/tensorflow/tensorflow:latest-devel-gpu
#nvidia-docker run -v `pwd`:`pwd` -w `pwd` -it --name $1 gcr.io/tensorflow/tensorflow:latest-devel-gpu
