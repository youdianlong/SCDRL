#!/usr/bin/env bash

set -euo pipefail

learning_rates=(1e-2 5e-3 1e-3 5e-4 1e-4)
batch_sizes=(100)
target_datasets=(wisconsin texas cornell chameleon squirrel cora citeseer pubmed computers photo)
all_datasets=(wisconsin texas cornell chameleon squirrel cora citeseer pubmed computers photo)

backbone="MoE"
few_shot=1
split_method="RandomWalk"
backbone_tuning=true
pretrain_gpu=1
adapt_gpu=0

for target_dataset in "${target_datasets[@]}"; do
    echo "============================================================"
    echo "Now running downstream task on dataset: ${target_dataset}"
    echo "============================================================"

    source_datasets=()
    for dataset in "${all_datasets[@]}"; do
        if [ "${dataset}" != "${target_dataset}" ]; then
            source_datasets+=("${dataset}")
        fi
    done

    source_dataset_str=$(IFS=,; echo "${source_datasets[*]}")
    echo "Pretraining on sources: ${source_dataset_str}"
    echo "Fine-tuning target: ${target_dataset}"
    echo "------------------------------------------------------------"

    CUDA_VISIBLE_DEVICES=${pretrain_gpu} python src/main.py pretrain \
        --save_dir "storage/${backbone}/pretrained_model" \
        --source_datasets "${source_datasets[@]}" \
        --task node \
        --reconstruct 0.2 \
        --node_feature_dim 100 \
        --backbone ${backbone} \
        --split_method ${split_method} \
        --cross_link 1 \
        --cl_init_method learnable \
        --moe_num_experts 10

    pretrained_file="storage/${backbone}/pretrained_model/${source_dataset_str}_pretrained_model.pt"

    for lr in "${learning_rates[@]}"; do
        for bs in "${batch_sizes[@]}"; do
            CUDA_VISIBLE_DEVICES=${adapt_gpu} python src/main.py adapt \
                --save_dir "storage/${backbone}/result" \
                --few_shot ${few_shot} \
                --reconstruct 0.0 \
                --node_feature_dim 100 \
                --target_dataset ${target_dataset} \
                --task node \
                --backbone ${backbone} \
                --saliency_model none \
                --moe_num_experts 10 \
                --pretrained_file "${pretrained_file}" \
                --batch_size ${bs} \
                --lr ${lr} \
                --backbone_tuning ${backbone_tuning}
        done
    done
done
