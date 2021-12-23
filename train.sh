#!/bin/bash

dataset_meta_path="musicnet_metadata.csv"
fingerprints_cache_path="fingerprints_full_20211223.npz"
n=1
c=8
random_state=0
test_split=0.2
train_subsample_size=100000
mem_per_cpu="2G"
time="0-2:00:00"
suffix="100k_20211223"

for column in "ensemble" "composer" "key"
do
    sbatch -n $n \
           -c $c \
           --mem-per-cpu $mem_per_cpu \
           --time ${time} \
           --error "logs/${column}_xgboost_${suffix}.err" \
           --output "logs/${column}_xgboost_${suffix}.log" \
           --wrap "python train.py --dataset-meta-path ${dataset_meta_path}
                                   --fingerprints-cache-path ${fingerprints_cache_path}
                                   --out-path model_outputs/${column}_xgboost_${suffix}.joblib
                                   --random-state ${random_state}
                                   --test-split ${test_split}
                                   --train-subsample-size ${train_subsample_size}
                                   --column ${column}
                                   --model xgboost
                                   --split-by recording
                                   --verbose 1" &

    for model in "svd" "lu"
    do
        for dict_rank in "4" "8" "16" "32"
        do
            for error_model in "xgboost" "svm" "nearest"
            do
                echo $column $model $dict_rank $error_model
                model_name="${column}_${model}_${error_model}_rank_${dict_rank}_suffix"
                sbatch -n $n \
                       -c $c \
                       --mem-per-cpu $mem_per_cpu \
                       --time ${time} \
                       --error "logs/${model_name}.err" \
                       --output "logs/${model_name}.log" \
                       --wrap "python train.py --dataset-meta-path ${dataset_meta_path}
                                               --fingerprints-cache-path ${fingerprints_cache_path}
                                               --out-path model_outputs/${model_name}.joblib
                                               --random-state ${random_state}
                                               --test-split ${test_split}
                                               --train-subsample-size ${train_subsample_size}
                                               --column ${column}
                                               --model ${model}
                                               --error-model ${error_model}
                                               --dict-rank ${dict_rank}
                                               --split-by recording
                                               --verbose 1" &
            done
        done
    done
done
