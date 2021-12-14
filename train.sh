#!/bin/bash

for column in "ensemble" "composer" "key"
do
    for model in "svd" "lu"
    do
        python train.py --dataset-meta-path ~/Downloads/musicnet_metadata.csv \
                        --fingerprints-cache-path fingerprints_full_20211213.npz \
                        --out-path "${column}_${model}_25k_20211214.joblib" \
                        --random-state 0 \
                        --test-split 0.2 \
                        --train-subsample-size 25000 \
                        --column $column \
                        --model $model \
                        --split-by recording \
                        --verbose 1
    done
done
