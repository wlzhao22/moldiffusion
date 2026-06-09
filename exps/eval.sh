#!/bin/bash


NUM_NODES=1
NUM_GPUS_PER_NODE=4
NODE_RANK=0
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
DATESTR=$(date +"%m-%d-%H-%M")

# --- Hyperparameters ---
TOTAL_BATCH_SIZE=128
ACCUM_STEP=1
INPUT_SIZE=512
DIFFUSION_TIMESTEPS=480

# --- Paths and Naming ---
DATA_ROOT=./data
SAVE_PATH='your_save_path'

mkdir -p ${SAVE_PATH}
set -x # Print commands

# --- Launch Training ---
torchrun \
    --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --node_rank $NODE_RANK \
    --master_addr=localhost --master_port=$MASTER_PORT \
    main.py \
    --data_path ${DATA_ROOT} \
    --train_file "pubchem/train_pubchem.csv" \
    --aux_file uspto_mol/train_uspto.csv --coords_file aux_file \
    --valid_file "synthetic/indigo.csv" \
    --test_file "synthetic/chemdraw.csv" \
    --vocab_file "MolNexTR/vocab/vocab_chars.json" \
    --formats "chartok_coords,edges" \
    --save_path ${SAVE_PATH} \
    --load_path "ckpts/moldiffusion_best.pth" \
    \
    --do_test \
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
    --block_length 4 \
    \
    --input_size ${INPUT_SIZE} \
    --coord_bins 64 --sep_xy \
    \
    --batch_size $((TOTAL_BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    \
    --fp16 \
    --backend "nccl" \
    --print_freq 10 \
    2>&1 