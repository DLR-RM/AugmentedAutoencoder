#!/bin/bash

python generate_syn_det_train.py \
       --output_path /shared-folder/autoencoder_ws/data/ikea_mug/synthetic_data_occluded \
       --model /shared-folder/autoencoder_ws/data/ikea_mug/cad \
       --num 10000 \
       --vocpath /shared-folder/autoencoder_ws/data/VOC2012/JPEGImages \
       --model_type 'cad' \
       --scale 1.
