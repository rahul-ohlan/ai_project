#!/bin/bash

python train_clip.py \
    --experiment_ID 'cloome_scale_0.3' \
    --wandb_project 'ai_project_colab' \
    --wandb_entity 'ai_project_colab' \
    --train_batch_size 32 \
    --val_batch_size 32 \
    --lr 0.0005 \
    --num_epochs 100 \
    --use_wandb 'yes' \
    --device_id 1 \
    --loss_fn 'cloome' \
    --image_proj_output_size 64 \
    --image_proj_hidden_sizes '256, 128' \
    --mol_proj_hidden_sizes '512, 256, 128' \
    --mol_proj_input_size 2048 \
    --mol_proj_output_size 64 \
    --hopfield_input_dim 64 \
    --freeze_encoder 'no' \
    --learnable_inv_tau 'yes' \
    --inv_tau 14.3 \
    --hopfield_scale 0.3 


# Array of learning rates to experiment with
# LEARNING_RATES=(0.0005 0.001 0.01)
# # Loop over each learning rate
# for LR in "${LEARNING_RATES[@]}"
# do
#     EXPERIMENT_ID="experiment_lr_${LR}"
#     echo "Running experiment with learning rate: $LR and experiment ID: $EXPERIMENT_ID"
    
#     python train_clip.py \
#         --num_epochs $EPOCHS \
#         --batch_size $BATCH_SIZE \
#         --mol_proj_hidden_sizes $MOL_PROJ_HIDDEN_SIZES \
#         --mol_proj_input_size $MOL_PROJ_INPUT_SIZE \
#         --mol_proj_output_size $MOL_PROJ_OUTPUT_SIZE \
#         --freeze_encoder $FREEZE_ENCODER \
#         --ge_type $GE_TYPE \
#         --learnable_inv_tau $LEARNABLE_INV_TAU \
#         --inv_tau $INV_TAU \
#         --loss_fn $LOSS_FN \
#         --hopfield_scale $HOPFIELD_SCALE \
#         --use_wandb $USE_WANDB \
#         --experiment_ID $EXPERIMENT_ID \
#         --lr $LR
# done