# # DEEP NEURAL NETWORKS - ASSIGNMENT 3: RNN vs TRANSFORMER FOR TIME SERIES
#
# ## Recurrent Neural Networks vs Transformers for Time Series Prediction
#

# STUDENT INFORMATION (REQUIRED - DO NOT DELETE)
#
# BITS ID: [Enter your BITS ID here - e.g., 2025AA1234]
#
# Name: [Enter your full name here - e.g., JOHN DOE]
#
# Email: [Enter your email]
#
# Date: [Submission date]

# + active=""
# """
# ASSIGNMENT OVERVIEW
#
# This assignment requires you to implement and compare two approaches for 
# time series forecasting:
# 1. LSTM or GRU using Keras/PyTorch
# 2. Transformer encoder using Keras/PyTorch layers
#
# Learning Objectives:
# - Build recurrent neural networks for sequential data
# - Use transformer architecture for time series
# - Implement or integrate positional encoding
# - Compare RNN vs Transformer architectures
# - Understand time series preprocessing and evaluation
#
# IMPORTANT: 
# - Positional encoding MUST be added to transformer
# - Use torch.nn.TransformerEncoder or keras.layers.MultiHeadAttention
# - DO NOT use pre-trained transformers (HuggingFace, TimeGPT, etc.)
# - Use temporal train/test split (NO shuffling)
#
# """

# + active=""
# """
#  IMPORTANT SUBMISSION REQUIREMENTS - STRICTLY ENFORCED 
#
# 1. FILENAME FORMAT: <BITS_ID>_rnn_assignment.ipynb
#    Example: 2025AA05036_rnn_assignment.ipynb
#     Wrong filename = Automatic 0 marks
#
# 2. STUDENT INFORMATION MUST MATCH:
#     BITS ID in filename = BITS ID in notebook (above)
#     Name in folder = Name in notebook (above)
#     Mismatch = 0 marks
#
# 3. EXECUTE ALL CELLS BEFORE SUBMISSION:
#    - Run: Kernel → Restart & Run All
#    - Verify all outputs are visible
#     No outputs = 0 marks
#
# 4. FILE INTEGRITY:
#    - Ensure notebook opens without errors
#    - Check for corrupted cells
#     Corrupted file = 0 marks
#
# 5. IMPLEMENTATION REQUIREMENTS:
#    - MUST add positional encoding to transformer (custom or built-in)
#    - CAN use torch.nn.TransformerEncoder or keras.layers.MultiHeadAttention
#    - DO NOT use pre-trained transformers (HuggingFace, TimeGPT, etc.)
#    - DO NOT shuffle time series data (temporal order required)
#     Missing positional encoding = 0 marks for transformer section
#
# 6. DATASET REQUIREMENTS:
#    - Minimum 1000 time steps
#    - Train/test split: 90/10 OR 85/15 (temporal split only)
#    - Sequence length: 10-50 time steps
#    - Prediction horizon: 1-10 time steps
#
# 7. USE KERAS OR PYTORCH:
#    - Use framework's LSTM/GRU layers
#    - Use torch.nn.TransformerEncoder or keras.layers.MultiHeadAttention
#    - Add positional encoding (custom implementation or built-in)
#    - Use standard training methods
#
# 8. FILE SUBMISSION:
#    - Submit ONLY the .ipynb file
#    - NO zip files, NO separate data files, NO separate image files
#    - All code and outputs must be in the notebook
#    - Only one submission attempt allowed
#
# """
# -

# Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import json
import os
import math

# + active=""
# """
# PART 1: DATASET LOADING AND EXPLORATION (Informational)
#
# Instructions:
# 1. Choose ONE dataset from the allowed list
# 2. Load and explore the time series data
# 3. Fill in ALL required metadata fields below
# 4. Provide justification for your primary metric choice
#
# ALLOWED DATASETS:
# - Stock Prices (daily/hourly closing prices)
# - Weather Data (temperature, humidity, pressure)
# - Energy Consumption (electricity/power usage)
# - Sensor Data (IoT sensor readings)
# - Custom time series (with approval)
#
# REQUIRED OUTPUT:
# - Print all metadata fields
# - Time series plots
# - Stationarity analysis
# - Train/test split visualization
# """
# -

# ## 1.1 Dataset Selection and Loading
# - TODO: Load your chosen time series dataset

# REQUIRED: Fill in these metadata fields
dataset_name = "TODO: Enter dataset name"
dataset_source = "TODO: Enter source"
n_samples = 0  # TODO: Total number of time steps
n_features = 1  # TODO: Number of features (1 for univariate, >1 for multivariate)
sequence_length = 0  # TODO: Lookback window (10-50)
prediction_horizon = 0  # TODO: Forecast steps ahead (1-10)
problem_type = "time_series_forecasting"

# Primary metric selection
primary_metric = "TODO: MAE OR RMSE OR MAPE"
metric_justification = """
TODO: Write 1-2 sentences explaining your metric choice.
"""

print("DATASET INFORMATION")
print(f"Dataset: {dataset_name}")
print(f"Source: {dataset_source}")
print(f"Total Samples: {n_samples}")
print(f"Number of Features: {n_features}")
print(f"Sequence Length: {sequence_length}")
print(f"Prediction Horizon: {prediction_horizon}")
print(f"Primary Metric: {primary_metric}")
print(f"Metric Justification: {metric_justification}")

# ## 1.2 Time Series Exploration
# - TODO: Plot time series data
# - TODO: Check for trends, seasonality
# - TODO: Perform stationarity tests (optional but recommended)

# ### 1.3 Data Preprocessing
# - TODO: Preprocess data
# - TODO: Create sequences

def preprocess_timeseries(data):
    """
    Preprocess time series data
    
    Args:
        data: raw time series data
    
    Returns:
        preprocessed data, scaler
    """
    # TODO: Normalize/standardize data
    # TODO: Handle missing values if any
    pass

def create_sequences(data, seq_length, pred_horizon):
    """
    Create sequences for time series prediction
    
    Args:
        data: preprocessed time series data
        seq_length: lookback window
        pred_horizon: forecast steps ahead
    
    Returns:
        X: input sequences, y: target values
    """
    # TODO: Implement sliding window approach
    # Input: [t-n, t-n+1, ..., t-1, t]
    # Target: [t+1] or [t+1, ..., t+h]
    pass

# REQUIRED: Temporal train/test split (NO SHUFFLING)
train_test_ratio = "TODO: 90/10 OR 85/15"
train_samples = 0  # TODO: Number of training sequences
test_samples = 0  # TODO: Number of test sequences

print(f"\nTrain/Test Split: {train_test_ratio}")
print(f"Training Samples: {train_samples}")
print(f"Test Samples: {test_samples}")
print("  IMPORTANT: Temporal split used (NO shuffling)")


# + active=""
# """
# PART 2: LSTM/GRU IMPLEMENTATION (5 MARKS)
#
# REQUIREMENTS:
# - Build LSTM OR GRU using Keras/PyTorch layers
# - Architecture must include:
#   * At least 2 stacked recurrent layers
#   * Output layer for prediction
# - Use model.compile() and model.fit() (Keras) OR standard PyTorch training
# - Track initial_loss and final_loss
#
# GRADING:
# - LSTM/GRU architecture with stacked layers: 2 marks
# - Model properly compiled/configured: 1 mark
# - Training completed with loss tracking: 1 mark
# - All metrics calculated correctly: 1 mark
# """
# -

# ### 2.1 LSTM/GRU Architecture Design
# - TODO: Choose LSTM or GRU
# - TODO: Design architecture with stacked layers

def build_rnn_model(model_type, input_shape, hidden_units, n_layers, output_size):
    """
    Build LSTM or GRU model
    
    Args:
        model_type: string ('LSTM' or 'GRU')
        input_shape: tuple (sequence_length, n_features)
        hidden_units: number of hidden units per layer
        n_layers: number of stacked layers (minimum 2)
        output_size: prediction horizon
    
    Returns:
        model: compiled RNN model
    """
    # TODO: Implement LSTM or GRU architecture
    # TODO: Stack at least 2 layers
    # TODO: Add output layer
    pass

# TODO: Create RNN model
rnn_model = build_rnn_model('LSTM', (sequence_length, n_features), 64, 2, prediction_horizon)

# +
# TODO: Compile model
# For Keras: model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# For PyTorch: define optimizer and loss function
# -

# ### 2.2 Train RNN Model

print("\nRNN MODEL TRAINING")
# Track training time
rnn_start_time = time.time()

# +
# TODO: Train your model
# For Keras: history = rnn_model.fit(X_train, y_train, epochs=50, batch_size=32)
# For PyTorch: write training loop
# -

rnn_training_time = time.time() - rnn_start_time

# REQUIRED: Track initial and final loss
rnn_initial_loss = 0.0  # TODO: Get from training history (first epoch)
rnn_final_loss = 0.0  # TODO: Get from training history (last epoch)

print(f"Training completed in {rnn_training_time:.2f} seconds")
print(f"Initial Loss: {rnn_initial_loss:.4f}")
print(f"Final Loss: {rnn_final_loss:.4f}")


# ### 2.3 Evaluate RNN Model
# - TODO: Make predictions on test set
# - TODO: Inverse transform if data was normalized
# - TODO: Calculate all 4 required metrics

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    # TODO: Implement MAPE calculation
    # MAPE = mean(|y_true - y_pred| / |y_true|) * 100
    pass

# REQUIRED: Calculate all 4 metrics
rnn_mae = 0.0  # TODO: mean_absolute_error(y_test, y_pred)
rnn_rmse = 0.0  # TODO: sqrt(mean_squared_error(y_test, y_pred))
rnn_mape = 0.0  # TODO: calculate_mape(y_test, y_pred)
rnn_r2 = 0.0  # TODO: r2_score(y_test, y_pred)

print("\nRNN Model Performance:")
print(f"MAE:   {rnn_mae:.4f}")
print(f"RMSE:  {rnn_rmse:.4f}")
print(f"MAPE:  {rnn_mape:.4f}%")
print(f"R² Score: {rnn_r2:.4f}")


# ### 2.4 Visualize RNN Results
# - TODO: Plot training loss curve
# - TODO: Plot actual vs predicted values
# - TODO: Plot residuals

# + active=""
# """
# PART 3: TRANSFORMER IMPLEMENTATION (5 MARKS)
#
# REQUIREMENTS:
# - Build Transformer encoder using Keras/PyTorch layers
# - MUST add positional encoding to input:
#   * Custom sinusoidal implementation OR
#   * Use built-in positional encoding (if framework provides)
# - Use torch.nn.TransformerEncoder or keras.layers.MultiHeadAttention
# - Use standard training methods
# - Track initial_loss and final_loss
#
# PROHIBITED:
# - Using pre-trained transformers (HuggingFace, TimeGPT, etc.)
# - Skipping positional encoding entirely
#
# GRADING:
# - Positional encoding added: 1 mark
# - Transformer architecture properly configured: 2 marks
# - Training completed with loss tracking: 1 mark
# - All metrics calculated correctly: 1 mark
# """
# -

# ### 3.1 Positional Encoding Implementation

def positional_encoding(seq_length, d_model):
    """
    Generate sinusoidal positional encodings
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Args:
        seq_length: length of the sequence
        d_model: dimension of the model
    
    Returns:
        positional encodings: array of shape (seq_length, d_model)
    """
    # TODO: Implement sinusoidal positional encoding
    # OR use framework's built-in positional encoding if available
    pass

# ###  3.2 Transformer Encoder Architecture
# - Option A: Using Pytorch
# - Option B: Using Keras

# +
# Option A: Using PyTorch
# TODO: write your code here

# +
# Option B: Using Keras
# TODO: write your code here
# -

# ### 3.3 Build Your Transformer Model

# +
# TODO: Create Transformer model using PyTorch or Keras
# Example for PyTorch:
# transformer_model = TransformerModel(n_features, d_model=64, n_heads=4, n_layers=2, d_ff=256, output_size=prediction_horizon)
# Example for Keras:
# transformer_model = build_transformer_model(sequence_length, n_features, d_model=64, n_heads=4, n_layers=2, d_ff=256, output_size=prediction_horizon)

# +
# TODO: Define optimizer and loss
# For PyTorch: optimizer = torch.optim.Adam(transformer_model.parameters(), lr=0.001); criterion = nn.MSELoss()
# For Keras: model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# For PyTorch: define optimizer and loss function
# -

# ### 3.4 Train Transformer Model

print("\nTRANSFORMER MODEL TRAINING")
# Track training time
transformer_start_time = time.time()

# +
# TODO: Train your model
# For Keras: history = transformer_model.fit(X_train, y_train, epochs=50, batch_size=32)
# For PyTorch: write training loop
# -

transformer_training_time = time.time() - transformer_start_time

# REQUIRED: Track initial and final loss
transformer_initial_loss = 0.0  # TODO: Get from training history (first epoch)
transformer_final_loss = 0.0  # TODO: Get from training history (last epoch)

print(f"Training completed in {transformer_training_time:.2f} seconds")
print(f"Initial Loss: {transformer_initial_loss:.4f}")
print(f"Final Loss: {transformer_final_loss:.4f}")

# ### 3.5 Evaluate Transformer Model
#
# - TODO: Make predictions on test set
# - TODO: Inverse transform if data was normalized
# - TODO: Calculate all 4 required metrics

# REQUIRED: Calculate all 4 metrics
transformer_mae = 0.0  # TODO: mean_absolute_error(y_test, y_pred)
transformer_rmse = 0.0  # TODO: sqrt(mean_squared_error(y_test, y_pred))
transformer_mape = 0.0  # TODO: calculate_mape(y_test, y_pred)
transformer_r2 = 0.0  # TODO: r2_score(y_test, y_pred)

print("\nTransformer Model Performance:")
print(f"MAE:   {transformer_mae:.4f}")
print(f"RMSE:  {transformer_rmse:.4f}")
print(f"MAPE:  {transformer_mape:.4f}%")
print(f"R² Score: {transformer_r2:.4f}")

# ### 3.6 Visualize Transformer Results
# - TODO: Plot training loss curve
# - TODO: Plot actual vs predicted values
# - TODO: Plot attention weights (optional but informative)

# + active=""
# """
# PART 4: MODEL COMPARISON AND VISUALIZATION (Informational)
#
# Compare both models on:
# - Performance metrics
# - Training time
# - Model complexity
# - Convergence behavior
# - Ability to capture long-term dependencies
# """
# -

# ### 4.1 Metrics Comparison

comparison_df = pd.DataFrame({
    'Metric': ['MAE', 'RMSE', 'MAPE (%)', 'R² Score', 'Training Time (s)', 'Parameters'],
    'RNN (LSTM/GRU)': [
        rnn_mae,
        rnn_rmse,
        rnn_mape,
        rnn_r2,
        rnn_training_time,
        0  # TODO: Fill with RNN total parameters
    ],
    'Transformer': [
        transformer_mae,
        transformer_rmse,
        transformer_mape,
        transformer_r2,
        transformer_training_time,
        0  # TODO: Fill with Transformer total parameters
    ]
})

print(comparison_df.to_string(index=False))

# ### 4.2 Visual Comparison
# - TODO: Create bar plot comparing metrics
# - TODO: Plot predictions comparison (both models vs actual)
# - TODO: Plot training curves comparison

"""
PART 5: ANALYSIS (2 MARKS)

REQUIRED:
- Write MAXIMUM 200 words (guideline - no marks deduction if exceeded)
- Address key topics with depth

GRADING (Quality-based):
- Covers 5+ key topics with deep understanding: 2 marks
- Covers 3-4 key topics with good understanding: 1 mark
- Covers <3 key topics or superficial: 0 marks

Key Topics:
1. Performance comparison with specific metrics
2. RNN vs Transformer architecture advantages
3. Impact of attention mechanism vs recurrent connections
4. Long-term dependency handling comparison
5. Computational cost comparison
6. Convergence behavior differences
"""

analysis_text = """
TODO: Write your analysis here (maximum 200 words guideline)

Address:
1. Which model performed better and by how much?
   [Compare specific metrics]

2. RNN vs Transformer architecture advantages?
   [Discuss sequential processing vs parallel processing]

3. Impact of attention mechanism?
   [Discuss how attention captures dependencies]

4. Long-term dependency handling?
   [Compare vanishing gradients vs attention]

5. Computational cost comparison?
   [Compare training time, parameters]

6. Convergence behavior?
   [Discuss training stability, loss curves]
"""

# REQUIRED: Print analysis with word count
print("ANALYSIS")
print(analysis_text)
print(f"Analysis word count: {len(analysis_text.split())} words")
if len(analysis_text.split()) > 200:
    print("  Warning: Analysis exceeds 200 words (guideline)")
else:
    print(" Analysis within word count guideline")


# + active=""
# """
# PART 6: ASSIGNMENT RESULTS SUMMARY (REQUIRED FOR AUTO-GRADING)
#
# DO NOT MODIFY THE STRUCTURE BELOW
# This JSON output is used by the auto-grader
# Ensure all field names are EXACT
# """
# -

def get_assignment_results():
    """
    Generate complete assignment results in required format
    
    Returns:
        dict: Complete results with all required fields
    """
    
    framework_used = "keras"  # TODO: Change to "pytorch" if using PyTorch
    rnn_model_type = "LSTM"  # TODO: Change to "GRU" if using GRU
    
    results = {
        # Dataset Information
        'dataset_name': dataset_name,
        'dataset_source': dataset_source,
        'n_samples': n_samples,
        'n_features': n_features,
        'sequence_length': sequence_length,
        'prediction_horizon': prediction_horizon,
        'problem_type': problem_type,
        'primary_metric': primary_metric,
        'metric_justification': metric_justification,
        'train_samples': train_samples,
        'test_samples': test_samples,
        'train_test_ratio': train_test_ratio,
        
        # RNN Model Results
        'rnn_model': {
            'framework': framework_used,
            'model_type': rnn_model_type,
            'architecture': {
                'n_layers': 0,  # TODO: Number of stacked layers
                'hidden_units': 0,  # TODO: Hidden units per layer
                'total_parameters': 0  # TODO: Calculate total parameters
            },
            'training_config': {
                'learning_rate': 0.001,  # TODO: Your actual learning rate
                'n_epochs': 50,  # TODO: Your actual epochs
                'batch_size': 32,  # TODO: Your actual batch size
                'optimizer': 'Adam',  # TODO: Your actual optimizer
                'loss_function': 'MSE'  # TODO: Your actual loss
            },
            'initial_loss': rnn_initial_loss,
            'final_loss': rnn_final_loss,
            'training_time_seconds': rnn_training_time,
            'mae': rnn_mae,
            'rmse': rnn_rmse,
            'mape': rnn_mape,
            'r2_score': rnn_r2
        },
        
        # Transformer Model Results
        'transformer_model': {
            'framework': framework_used,
            'architecture': {
                'n_layers': 0,  # TODO: Number of transformer layers
                'n_heads': 0,  # TODO: Number of attention heads
                'd_model': 0,  # TODO: Model dimension
                'd_ff': 0,  # TODO: Feed-forward dimension
                'has_positional_encoding': True,  # MUST be True
                'has_attention': True,  # MUST be True
                'total_parameters': 0  # TODO: Calculate total parameters
            },
            'training_config': {
                'learning_rate': 0.001,  # TODO: Your actual learning rate
                'n_epochs': 50,  # TODO: Your actual epochs
                'batch_size': 32,  # TODO: Your actual batch size
                'optimizer': 'Adam',  # TODO: Your actual optimizer
                'loss_function': 'MSE'  # TODO: Your actual loss
            },
            'initial_loss': transformer_initial_loss,
            'final_loss': transformer_final_loss,
            'training_time_seconds': transformer_training_time,
            'mae': transformer_mae,
            'rmse': transformer_rmse,
            'mape': transformer_mape,
            'r2_score': transformer_r2
        },
        
        # Analysis
        'analysis': analysis_text,
        'analysis_word_count': len(analysis_text.split()),
        
        # Training Success Indicators
        'rnn_loss_decreased': rnn_final_loss < rnn_initial_loss if rnn_initial_loss and rnn_final_loss else False,
        'transformer_loss_decreased': transformer_final_loss < transformer_initial_loss if transformer_initial_loss and transformer_final_loss else False,
    }
    
    return results

# Generate and print results
try:
    assignment_results = get_assignment_results()
    print("ASSIGNMENT RESULTS SUMMARY")
    print(json.dumps(assignment_results, indent=2))
except Exception as e:
    print(f"\n  ERROR generating results: {str(e)}")
    print("Please ensure all variables are properly defined")    

# + active=""
# """
# ENVIRONMENT VERIFICATION - SCREENSHOT REQUIRED
#
# IMPORTANT: Take a screenshot of your environment showing account details
#
# For Google Colab:
# - Click on your profile icon (top right)
# - Screenshot should show your email/account clearly
# - Include the entire Colab interface with notebook name visible
#
# For BITS Virtual Lab:
# - Screenshot showing your login credentials/account details
# - Include the entire interface with your username/session info visible
#
# Paste the screenshot below this cell or in a new markdown cell.
# This helps verify the work was done by you in your environment.
#
# """
# -

# Display system information
import platform
import sys
from datetime import datetime

# +
print("ENVIRONMENT INFORMATION")
print("\n  REQUIRED: Add screenshot of your Google Colab/BITS Virtual Lab")
print("showing your account details in the cell below this one.")

# include the screen shot here

# + active=""
# """
# FINAL CHECKLIST - VERIFY BEFORE SUBMISSION
#
# □ Student information filled at the top (BITS ID, Name, Email)
# □ Filename is <BITS_ID>_rnn_assignment.ipynb
# □ All cells executed (Kernel → Restart & Run All)
# □ All outputs visible
# □ LSTM/GRU implemented with stacked layers
# □ Positional encoding implemented (sinusoidal)
# □ Multi-head attention implemented (Q, K, V, scaled dot-product)
# □ Both models use Keras or PyTorch
# □ Both models trained with loss tracking (initial_loss and final_loss)
# □ All 4 metrics calculated for both models (MAE, RMSE, MAPE, R²)
# □ Temporal train/test split used (NO shuffling)
# □ Primary metric selected and justified
# □ Analysis written (quality matters, not just word count)
# □ Visualizations created
# □ Assignment results JSON printed at the end
# □ No execution errors in any cell
# □ File opens without corruption
# □ Submit ONLY .ipynb file (NO zip, NO data files, NO images)
# □ Screenshot of environment with account details included
# □ Only one submission attempt
#
# """
