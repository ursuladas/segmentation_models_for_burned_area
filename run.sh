#!/bin/bash

# Path to the Python script
SCRIPT_PATH="/home/udas/Desktop/UD_Data_Copy/Segmentation_Models/train.py"

# Root directory containing the dataset
ROOT_DIR="/home/udas/Desktop/UD_Data_Copy/b.final_burned_area"

# Paths to the training and validation statistics pickle files
TRAIN_STATS="/home/udas/Desktop/UD_Data_Copy/Spatial_Models/train_stats.pkl"
VAL_STATS="/home/udas/Desktop/UD_Data_Copy/Spatial_Models/val_stats.pkl"

# Input variables and target variable
INPUT_VARS="ignition_points ssrd smi d2m t2m wind_speed"
TARGET="burned_areas"

# Model configuration
MODEL_NAME="DeepLabV3"
ENCODER_NAME="resnet34"

# Training parameters
TRAIN_YEARS="2017 2018 2019 2020"
VAL_YEARS="2022"
CROP_SIZE=32
BATCH_SIZE=16
EPOCHS=25
LEARNING_RATE=1e-4
LOSS="BCE_raw"
ROTATE=True
# Run the Python script with the specified arguments
python3 "$SCRIPT_PATH" \
  --root_dir "$ROOT_DIR" \
  --train_stat_dict "$TRAIN_STATS" \
  --val_stat_dict "$VAL_STATS" \
  --input_vars $INPUT_VARS \
  --target "$TARGET" \
  --model_name "$MODEL_NAME" \
  --encoder_name "$ENCODER_NAME" \
  --train_years $TRAIN_YEARS \
  --val_years $VAL_YEARS \
  --crop_size "$CROP_SIZE" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --learning_rate "$LEARNING_RATE" \
  --loss "$LOSS" \
  --desc "$MODEL_NAME"+"$LOSS"  \
  --rotate $ROTATE
