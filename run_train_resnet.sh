#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3
export PYTHONWARNINGS="ignore"

# resnet18, resnet50
export NET='resnet18'
export path='model'
export data='data/fg-web-data/web-bird'
export N_CLASSES=200
export lr=0.01
export w_decay=1e-5
export epochs=80
export batchsize=32
export droprate=0.25
export denoise=True
export smooth=True
export label_weight=0.5


python train.py --net ${NET} --n_classes ${N_CLASSES} --denoise ${denoise} --droprate ${droprate} --smooth ${smooth} --label_weight ${label_weight}  --path ${path} --data_base ${data}  --lr ${lr} --w_decay ${w_decay} --batch_size ${batchsize} --epochs ${epochs}
