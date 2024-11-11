#!/bin/bash

# Path to the Python script
SCRIPT_PATH="/home/udas/Desktop/UD_Data_Copy/Segmentation_Models/train.py"

# Root directory containing the dataset
ROOT_DIR="/home/udas/Desktop/UD_Data_Copy/b.final_burned_area"

# Paths to the training and validation statistics pickle files
TRAIN_STATS="/home/udas/Desktop/UD_Data_Copy/Spatial_Models/graph_class_scripts/stat_dicts/train_stats.pkl"
VAL_STATS="/home/udas/Desktop/UD_Data_Copy/Spatial_Models/graph_class_scripts/stat_dicts/val_stats.pkl"

# Input variables and target variable
# INPUT_VARS="ignition_points ssrd smi d2m t2m wind_speed wind_direction slope aspect"
INPUT_VARS="ignition_points ndvi roads_distance slope smi lst_day lst_night sp t2m wind_direction wind_speed lc_agriculture lc_forest lc_grassland lc_settlement lc_shrubland lc_sparse_vegetation"
TARGET="burned_areas"
STAT="min_max"

# Model configuration
MODEL_NAME="DeepLabV3"
ENCODER_NAME="efficientnet-b7"

# Training parameters
TRAIN_YEARS="2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019"
# TRAIN_YEARS="2006"

VAL_YEARS="2020"
# VAL_YEARS="2007"

CROP_SIZE=32
BATCH_SIZE=128
EPOCHS=200
LEARNING_RATE=0.001
WEIGHT_DECAY=0.0005
LOSS="BCE_raw"
ROTATE="--rotate"
# Run the Python script with the specified arguments
python3 "$SCRIPT_PATH" \
  --root_dir "$ROOT_DIR" \
  --stat "$STAT"\
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
  --weight_decay "$WEIGHT_DECAY" \
  --loss "$LOSS" \
  --desc "$MODEL_NAME"+"$LOSS"+"$ENCODER_NAME"  \
  # $ROTATE