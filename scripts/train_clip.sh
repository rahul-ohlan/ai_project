#!/bin/bash
# Fixed arguments
EPOCHS=200
MOL_PROJ_HIDDEN_SIZES="256,256"
MOL_PROJ_INPUT_SIZE=2048
MOL_PROJ_OUTPUT_SIZE=64
FREEZE_ENCODER=False
LEARNABLE_INV_TAU=True
INV_TAU=14.3
LOSS_FN="infoNCE"
HOPFIELD_SCALE=0.3
USE_WANDB="yes"
TRAIN_BATCH_SIZE=512  # Fixed batch size

# Array of learning rates to experiment with
LEARNING_RATES=(0.0005 0.001 0.01)
# Loop over each learning rate
for LR in "${LEARNING_RATES[@]}"
do
    EXPERIMENT_ID="experiment_lr_${LR}"
    echo "Running experiment with learning rate: $LR and experiment ID: $EXPERIMENT_ID"
    
    python train_clip.py \
        --num_epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --mol_proj_hidden_sizes $MOL_PROJ_HIDDEN_SIZES \
        --mol_proj_input_size $MOL_PROJ_INPUT_SIZE \
        --mol_proj_output_size $MOL_PROJ_OUTPUT_SIZE \
        --freeze_encoder $FREEZE_ENCODER \
        --ge_type $GE_TYPE \
        --learnable_inv_tau $LEARNABLE_INV_TAU \
        --inv_tau $INV_TAU \
        --loss_fn $LOSS_FN \
        --hopfield_scale $HOPFIELD_SCALE \
        --use_wandb $USE_WANDB \
        --experiment_ID $EXPERIMENT_ID \
        --lr $LR
done