#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2
export PYTHONWARNINGS="ignore"

# bcnn
export NET='bcnn'
export path='model'
export data='data/fg-web-data/web-bird'
export N_CLASSES=200
export lr=0.1
export w_decay=1e-8
export epochs=60
export batchsize=64
export step=1
export droprate=0.25
export denoise=True
export smooth=True
export label_weight=0.6
export tk=2

python train.py --net ${NET} --n_classes ${N_CLASSES} --denoise ${denoise} --droprate ${droprate} --smooth ${smooth} --label_weight ${label_weight}  --path ${path} --data_base ${data}  --lr ${lr} --w_decay ${w_decay} --batch_size ${batchsize} --epochs ${epochs} --tk ${tk} --step ${step}

sleep 300

export lr=0.01
export w_decay=1e-5
export batchsize=32
export step=2

python train.py --net ${NET} --n_classes ${N_CLASSES} --denoise ${denoise} --droprate ${droprate} --smooth ${smooth} --label_weight ${label_weight}  --path ${path} --data_base ${data}  --lr ${lr} --w_decay ${w_decay} --batch_size ${batchsize} --epochs ${epochs} --tk ${tk}  --step ${step}

