"""
Configuration file for Hard Drive RUL Prediction Pipeline
==========================================================
This module centralizes all configuration parameters for the streaming ML pipeline.
"""

import os

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Path to raw data folders
DATA_FOLDER_Q4_2024 = os.path.join("data", "data_Q4_2024")
DATA_FOLDER_Q1_2025 = os.path.join("data", "data_Q1_2025")

# Default data folder to use
DEFAULT_DATA_FOLDER = DATA_FOLDER_Q1_2025

# Output file for preprocessed data
PREPROCESSED_FILE = "stream_data.csv"

# ============================================================================
# PREPROCESSING CONFIGURATION
# ============================================================================

# Base columns (always included)
BASE_COLUMNS = ["date", "serial_number", "model", "failure", "capacity_bytes"]

# Columns to drop during model training (non-feature columns)
COLUMNS_TO_DROP = ["date", "serial_number", "model", "failure", "capacity_bytes"]

# Feature normalization options
NORMALIZATION_OPTIONS = {
    "raw": False,           # Use raw SMART values (very large numbers)
    "log": True,            # Apply log(1+x) transformation
    "normalized": True      # Apply StandardScaler (done in model pipeline)
}

# Default normalization method
DEFAULT_NORMALIZATION = "log"  # Options: "raw", "log", "normalized"

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# --- Baseline Model (Linear Regression) ---
BASELINE_CONFIG = {
    "name": "Linear Regression Baseline",
    "use_scaler": True,
    "description": "Simple linear model with StandardScaler"
}

# --- Hoeffding Tree Model ---
HOEFFDING_TREE_CONFIG = {
    "name": "Hoeffding Tree Regressor",
    "grace_period": 50,              # Number of instances between split attempts
    "leaf_prediction": "mean",       # Safe mode: use mean (alternatives: "model", "adaptive")
    "model_selector_decay": 0.9,     # Decay factor for model selection
    "use_scaler": True,
    "description": "Single incremental decision tree for streaming data"
}

# --- SRP Ensemble Model ---
SRP_CONFIG = {
    "name": "Streaming Random Patches (SRP)",
    "n_models": 10,                  # Number of trees in the ensemble
    "grace_period": 50,
    "leaf_prediction": "mean",
    "seed": 42,
    "use_scaler": True,
    "description": "Ensemble of Hoeffding Trees (like Random Forest for streams)"
}

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

# Metrics to track
METRICS_CONFIG = {
    "primary": "MAE",               # Mean Absolute Error (in days)
    "secondary": ["RMSE", "R2"]     # Additional metrics to track
}

# Reporting frequency (print progress every N instances)
REPORT_FREQUENCY = 10000

# Plotting frequency for real-time visualization
PLOT_FREQUENCY = 1000

# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================

# Directory to save results
RESULTS_DIR = "results"

# Directory to save plots
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# Directory to save trained models
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================

# Memory settings for preprocessing
PREPROCESSING_CONFIG = {
    "n_threads": 4,                  # Number of CPU threads for polars
    "batch_size": 10,                # Files to process before reporting progress
    "safe_mode": True                # Use diagonal_relaxed concat (safer but slower)
}

# Streaming settings
STREAMING_CONFIG = {
    "shuffle": False,                # Keep temporal order (critical for RUL prediction)
    "chunk_size": None,              # Process one sample at a time (None = streaming)
    "handle_missing": "fill_zero"    # How to handle missing values
}
