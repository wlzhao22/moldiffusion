#!/bin/bash


NUM_NODES=1
NUM_GPUS_PER_NODE=4
NODE_RANK=0
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
DATESTR=$(date +"%m-%d-%H-%M")

# --- Hyperparameters ---
TOTAL_BATCH_SIZE=128
ACCUM_STEP=1
ENCODER_LR=1e-4
DECODER_LR=1e-4
EPOCHS=5
WARMUP_RATIO=0.02
INPUT_SIZE=512
DIFFUSION_TIMESTEPS=480

# --- Paths and Naming ---
DATA_ROOT=./data
SAVE_ROOT=./output
EXP_NAME="pretrain_diffusion_molnextr_${DATESTR}"
SAVE_PATH=${SAVE_ROOT}/${EXP_NAME}

mkdir -p ${SAVE_PATH}
set -x # Print commands

# --- Launch Training ---
torchrun \
    --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --node_rank $NODE_RANK \
    --master_addr=localhost --master_port=$MASTER_PORT \
    main.py \
    --data_path ${DATA_ROOT} \
    --pretrain_dataset_path "data/molparser-7M" \
    --vocab_file "MolNexTR/vocab/vocab_chars.json" \
    --formats "chartok_coords" \
    --save_path ${SAVE_PATH} \
    \
    --do_pretrain \
    \
    --encoder "swin_base" \
    --use_checkpoint \
    --dec_num_layers 12 \
    --dec_hidden_size 512 \
    --dec_attn_heads 16 \
    \
    --decode_steps ${DIFFUSION_TIMESTEPS} \
    --cfg_dropout_prob 0.0 \
    --cfg_guidance_scale 1.0 \
    --temperature 0.0 \
    --block_length 8 \
    \
    --input_size ${INPUT_SIZE} \
    --coord_bins 64 --sep_xy \
    \
    --augment \
    \
    --epochs ${EPOCHS} \
    --batch_size $((TOTAL_BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps ${ACCUM_STEP} \
    --encoder_lr ${ENCODER_LR} \
    --decoder_lr ${DECODER_LR} \
    --weight_decay 1e-6 \
    --max_grad_norm 1.0 \
    --scheduler "cosine" \
    --warmup_ratio ${WARMUP_RATIO} \
    --save_mode "last" \
    \
    --fp16 \
    --backend "nccl" \
    --print_freq 200 \
    2>&1 | tee ${SAVE_PATH}/training_log.txt