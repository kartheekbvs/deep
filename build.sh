#!/bin/bash
# Render Build Script — Trains CharSenseNet-V4 during deployment
#
# This script runs during the Render build phase to:
# 1. Train the model if no pre-trained weights exist
# 2. Save the model to the models/ directory
#
# On Render GPU instances, training takes ~10-30 minutes.
# On Render CPU instances, training takes ~2-5 hours.
#
# Set SKIP_TRAINING=1 in Render environment variables to skip training
# and use pre-existing model weights.

set -e

echo "=== CharSense-V4 Build Script ==="

# Skip training if environment variable is set
if [ "${SKIP_TRAINING:-0}" = "1" ]; then
    echo "SKIP_TRAINING=1, skipping model training"
    if [ -f "models/universal_cnn_best.pth" ]; then
        echo "Using existing model: models/universal_cnn_best.pth"
    else
        echo "WARNING: No pre-trained model found!"
    fi
    exit 0
fi

# Check if model already exists
if [ -f "models/universal_cnn_best.pth" ]; then
    echo "Model already exists: models/universal_cnn_best.pth"
    echo "Skipping training. Delete the model file to retrain."
    exit 0
fi

# Set Render flag for full training
export RENDER=1
export TORCH_THREADS=4

echo "Training CharSenseNet-V4 model with RL-enhanced training..."
echo "This will train on EMNIST ByClass data with 4000 samples/class."
python3 train_v4_rl.py

echo "Build script complete!"
