"""
Configuration settings for the breast cancer diagnosis project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
ML_MODELS_DIR = os.path.join(MODELS_DIR, 'ml')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, ML_MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Dataset configurations
WISCONSIN_DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
WISCONSIN_DATASET_PATH = os.path.join(RAW_DATA_DIR, "wdbc.data")
WISCONSIN_COLUMNS = ["id", "diagnosis"] + [f"feature_{i}" for i in range(1, 31)]

# Model configurations
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# ML model hyperparameters
ML_MODELS = {
    "logistic_regression": {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"]
    },
    "random_forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10]
    },
    "xgboost": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2]
    }
}

# XAI configurations
N_SHAP_SAMPLES = 100

# Deep learning directories
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
DL_MODELS_DIR = os.path.join(MODELS_DIR, 'dl')
HYBRID_MODELS_DIR = os.path.join(MODELS_DIR, 'hybrid')

# Create model directories
os.makedirs(DL_MODELS_DIR, exist_ok=True)
os.makedirs(HYBRID_MODELS_DIR, exist_ok=True)

# DL model configurations
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
