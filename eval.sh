#!/bin/bash

# Path to the Python script
SCRIPT_PATH="/home/udas/Desktop/UD_Data_Copy/Segmentation_Models/test.py"

# Root directory containing the dataset
ROOT_DIR="/home/udas/Desktop/UD_Data_Copy/b.final_burned_area"

# Paths to the training and validation statistics pickle files
TEST_STATS="/home/udas/Desktop/UD_Data_Copy/Spatial_Models/graph_class_scripts/stat_dicts/test_stats.pkl"


# Input variables and target variable
# INPUT_VARS="ignition_points ssrd smi d2m t2m wind_speed wind_direction slope aspect"
INPUT_VARS="ignition_points ndvi roads_distance slope smi lst_day lst_night sp t2m wind_direction wind_speed lc_agriculture lc_forest lc_grassland lc_settlement lc_shrubland lc_sparse_vegetation"
TARGET="burned_areas"
STAT="min_max"

# Model configuration
MODEL_NAME="Unet++"
ENCODER_NAME="efficientnet-b7"
MODEL_PATH='/home/udas/Desktop/UD_Data_Copy/Segmentation_Models/models/20241108_183210_Unet++_BCE_raw.pth'


TEST_YEARS="2021 2022"

WANDB="--wandb"
CROP_SIZE=32
BATCH_SIZE=64
THRESHOLD=0.5
LOSS="BCE_raw"
ROTATE="--rotate"
# Run the Python script with the specified arguments
python3 "$SCRIPT_PATH" \
  --root_dir "$ROOT_DIR" \
  --stat "$STAT"\
  --test_stat_dict "$TEST_STATS" \
  --seg_threshold "$THRESHOLD"\
  --input_vars $INPUT_VARS \
  --target "$TARGET" \
  --model_name "$MODEL_NAME" \
  --encoder_name "$ENCODER_NAME" \
  --test_years $TEST_YEARS \
  --crop_size "$CROP_SIZE" \
  --batch_size "$BATCH_SIZE" \
  --loss "$LOSS" \
  --desc "TEST"+"$MODEL_NAME"+"$LOSS"+"$ENCODER_NAME"  \
  --model_path "$MODEL_PATH" \
  # $ROTATE
  $WANDB