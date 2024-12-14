#!/bin/bash

# custom config
DATA="/c/Users/sbp5911/Visual Studio Code Workspace/CSE 597 - VL/Project/PromptKD/DATA"
TRAINER=PromptKD

DATASET=$1 # 'imagenet' 'caltech101' 'dtd' 'eurosat' 'fgvc_aircraft' 'oxford_flowers' 'food101' 'oxford_pets' 'stanford_cars' 'sun397' 'ucf101'
SEED=$2

CFG=vit_b16_c2_ep20_batch8_4+4ctx
SHOTS=0
LOADEP=20 # Load the trained model from epoch 20
SUB=new # Subsample only novel classes

MODEL_DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed_${SEED}
DIR=output/base2new/test_${SUB}/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed_${SEED}

if [ -d "$DIR" ]; then
    echo "Evaluating model on novel classes"
    echo "Results are available in ${DIR}. Resuming..."

    CUDA_VISIBLE_DEVICES=0 python train.py \
        --root "${DATA}" \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --load-epoch ${LOADEP} \
        --eval-only \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB}

else
    echo "Evaluating model on novel classes"
    echo "Running the first phase job and saving the output to ${DIR}"

    CUDA_VISIBLE_DEVICES=0 python train.py \
        --root "${DATA}" \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --load-epoch ${LOADEP} \
        --eval-only \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB}
fi
