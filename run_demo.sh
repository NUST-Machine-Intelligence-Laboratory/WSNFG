#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

python demo.py --data data/web-bird --model /farm/zcy/BCNN/use_for_drawing/models/resnet50_81.pth
