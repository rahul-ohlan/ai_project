#!/bin/bash

# constants
DATA_DIR="/rohlan/workspace/data/bbbc021_all"
EMBEDDINGS_FILE="unique_smiles_morgan_fingerprints.pkl"
CHECKPOINT_DIR="/rohlan/workspace/checkpoints/"
MODEL_CHKPT_DIR="clip"
FREEZE_ENCODER="false"

# data args
USE_WANDB="yes"
WANDB_PROJECT="ai_project_colab"
WANDB_ENTITY="ai_project_colab"
BATCH_SIZE=256
NUM_EPOCHS=200
LR=0.0005
PATIENCE=25
DELTA=0.0001
DEVICE_ID=3 # 0, 1, 2, 3
DEVICE_IDS="0,1,2,3"
DISTRIBUTED="false"
LOSS_FN="cloome"
LEARNABLE_INV_TAU="false"
INV_TAU=14.3   # 0.2, 2.3, 5.0, 14.3
HOPFIELD_SCALE=0.3
HOPFIELD_INPUT_DIM=64
INV_TAU_CLAMP="yes"

EXPERIMENT_ID="cloome_inv_tau_${INV_TAU}_dosage_10.0"
echo "Running experiment with inv tau: $INV_TAU and experiment ID: $EXPERIMENT_ID"

python train_clip.py \
    --experiment_ID $EXPERIMENT_ID \
    --use_wandb $USE_WANDB \
    --wandb_project $WANDB_PROJECT \
    --wandb_entity $WANDB_ENTITY \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --lr $LR \
    --patience $PATIENCE \
    --delta $DELTA \
    --device_id $DEVICE_ID \
    --device_ids $DEVICE_IDS \
    --distributed $DISTRIBUTED \
    --loss_fn $LOSS_FN \
    --learnable_inv_tau $LEARNABLE_INV_TAU \
    --inv_tau $INV_TAU \
    --hopfield_scale $HOPFIELD_SCALE \
    --hopfield_input_dim $HOPFIELD_INPUT_DIM \
    --inv_tau_clamp $INV_TAU_CLAMP \
    --dosage_level 10.0 \
    --add_dosage false 

# Array of a parameter to experiment with
# INV_TAU=(0.2 1.0 5.0 14.3)
# # Loop over each learning rate
# for i in "${!INV_TAU[@]}";
# do
#     inv_tau=${INV_TAU[$i]}
#     EXPERIMENT_ID="cloome_inv_tau_${inv_tau}"
#     echo "Running experiment with inv tau: $inv_tau and experiment ID: $EXPERIMENT_ID"
#     python train_clip.py \
#     --experiment_ID $EXPERIMENT_ID \
#     --use_wandb $USE_WANDB \
#     --wandb_project $WANDB_PROJECT \
#     --wandb_entity $WANDB_ENTITY \
#     --batch_size $BATCH_SIZE \
#     --num_epochs $NUM_EPOCHS \
#     --lr $LR \
#     --patience $PATIENCE \
#     --delta $DELTA \
#     --device_id $DEVICE_ID \
#     --device_ids $DEVICE_IDS \
#     --distributed $DISTRIBUTED \
#     --loss_fn $LOSS_FN \
#     --learnable_inv_tau $LEARNABLE_INV_TAU \
#     --inv_tau $INV_TAU \
#     --hopfield_scale $HOPFIELD_SCALE \
#     --hopfield_input_dim $HOPFIELD_INPUT_DIM \
#     --inv_tau_clamp $INV_TAU_CLAMP
    
# done