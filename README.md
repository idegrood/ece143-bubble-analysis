# ECE 143 - Bubble Analysis Project: AI - Boom or Bust

**Group 17**

A comprehensive analysis framework for identifying and predicting economic bubbles across multiple technology sectors. This project analyzes historical bubbles (Dotcom, Crypto, EV) to predict company survival probabilities during the current AI bubble.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Data Collection](#data-collection)
  - [Data Visualization](#data-visualization)
  - [Model Training & Prediction](#model-training--prediction)
- [Methodology](#methodology)
- [Bubble Types Analyzed](#bubble-types-analyzed)
- [Model Details](#model-details)
- [File Descriptions](#file-descriptions)

## Overview

This project provides a data-driven approach to analyze economic bubbles by:

1. **Collecting** financial data from WRDS (CRSP + Compustat) for companies across different bubble periods
2. **Calculating** bubble indices using normalized valuation metrics (P/S, P/B, P/E, EV/Sales)
3. **Training** machine learning models on historical bubbles to predict company survival probabilities
4. **Visualizing** bubble dynamics and company-specific metrics over time

The primary goal is to predict which companies in the current AI bubble are likely to survive based on patterns learned from past bubbles (Dotcom, Crypto, EV).

## Project Structure

```
ece143-bubble-analysis/
├── bubble_config.json          # Configuration for all bubble types
├── bubble_data_collector.py    # Main data collection script
├── bubble_type.py              # Enum for bubble type management
├── bubble_plotting_updated.py  # Interactive plotting functions
├── plotting_bubble_data.py     # Original plotting scripts
├── data/                       # Generated CSV datasets
│   ├── ai_merged.csv
│   ├── crypto_merged.csv
│   ├── dotcom_merged.csv
│   └── ev_merged.csv
└── modeling/                   # Machine learning pipeline
    ├── preprocess.py           # Data preprocessing and feature engineering
    ├── random_forest.py        # Random Forest regression model
    └── run.py                  # Main training and prediction script
```

## Features

- **Automated Data Collection**: Pulls daily stock prices and quarterly fundamentals from WRDS
- **Multi-Bubble Analysis**: Supports analysis of 4 different bubble types (AI, Crypto, EV, Dotcom)
- **Bubble Index Calculation**: Computes normalized z-scores across multiple valuation metrics
- **Survival Probability Prediction**: Uses Random Forest to predict company survival based on historical patterns
- **Interactive Visualization**: Jupyter notebook widgets for exploring bubble data
- **Configurable**: JSON-based configuration for easy addition of new bubble types

## Prerequisites

- Python 3.8+
- WRDS account and credentials (for data collection)
- Required Python packages:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`
  - `wrds` (WRDS Python API)
  - `ipywidgets` (for interactive plots)

## Installation

1. Clone or download this repository

2. Install required packages:
```bash
pip install pandas numpy matplotlib scikit-learn wrds ipywidgets
```

3. Set up WRDS credentials:
   - First-time users will be prompted to enter WRDS username and password
   - Credentials are stored locally for future use

## Configuration

The `bubble_config.json` file defines all bubble types with their associated:
- **Keywords**: Search terms for identifying companies in Compustat
- **Tickers**: Stock ticker symbols
- **Names**: Full company names (for exact matching)
- **Date Ranges**: Start and end dates for the bubble period
- **Data Paths**: Output CSV file locations

Example configuration structure:
```json
{
  "ai": {
    "keywords": ["Artificial Intelligence", "AI", "LLM", ...],
    "tickers": ["NVDA", "AMD", "MSFT", ...],
    "names": ["NVIDIA CORP", "MICROSOFT CORP", ...],
    "start_date": "2020-01-01",
    "end_date": "2025-11-30",
    "data_path": "data/ai_merged.csv"
  }
}
```

## Usage

### Data Collection

To collect data for all bubbles defined in `bubble_config.json`:

```bash
python bubble_data_collector.py
```

This will:
1. Connect to WRDS
2. For each bubble type:
   - Read `start_date` and `end_date` from the config file
   - Search Compustat for matching companies
   - Link to CRSP for stock price data
   - **Filter CRSP daily data by date range** (`date >= start_date` and `date <= end_date`)
   - Merge with quarterly fundamentals
   - Calculate valuation metrics and bubble indices
   - Save to CSV in the `data/` directory

**Date Handling:**
- Dates are specified in `bubble_config.json` for each bubble type (format: `"YYYY-MM-DD"`)
- The script uses these dates to filter CRSP stock price data
- If `end_date` is not specified in config, it will pull data up to the latest available date
- Dates can also be passed directly to `build_bubble_panel()` function parameters

To collect data for a specific bubble programmatically:

```python
from bubble_data_collector import build_bubble_panel

# Dates from config file (default)
df = build_bubble_panel("ai")

# Or specify custom dates
df = build_bubble_panel("ai", start_date="2020-01-01", end_date="2024-12-31")
```

### Data Visualization

#### Interactive Plotting (Jupyter Notebook)

Use `bubble_plotting_updated.py` in a Jupyter notebook for interactive exploration. The file contains both the plotting function and widget setup code:

```python
import pandas as pd

# Load merged data (can load multiple bubbles and concatenate)
merged = pd.read_csv('data/ai_merged.csv')
# Or load all bubbles:
# merged = pd.concat([
#     pd.read_csv('data/ai_merged.csv'),
#     pd.read_csv('data/crypto_merged.csv'),
#     pd.read_csv('data/dotcom_merged.csv'),
#     pd.read_csv('data/ev_merged.csv')
# ])

# Import the plotting module (this will set up the interactive widgets)
from bubble_plotting_updated import plot_bubble_interactive
import ipywidgets as widgets

# The widgets are set up in the file - they will appear automatically
# Select bubble type and company from dropdown menus to explore the data
```

Features:
- Aggregate bubble index over time
- Company-specific price vs. sales analysis
- P/S ratio with bubble thresholds
- Interactive dropdowns for bubble type and company selection

#### Static Plotting

Use `plotting_bubble_data.py` for static visualizations:
- Bubble index time series
- Price vs. sales comparisons
- Market cap growth analysis
- P/S ratio trends

### Model Training & Prediction

Train a Random Forest model on historical bubbles and predict AI company survival:

```bash
cd modeling
python run.py
```

This script:
1. **Preprocesses** data from Crypto, Dotcom, and EV bubbles (training set)
2. **Identifies** bubble bounds (peak to 30% drawdown)
3. **Calculates** survival probabilities based on price retention during bubble burst
4. **Trains** Random Forest regressor on valuation metrics
5. **Predicts** survival probabilities for AI companies
6. **Displays** feature importances and predictions

**Output:**
- Feature importance rankings
- Predicted survival probabilities for each AI company
- Model performance metrics (if validation data available)

#### Custom Model Training

```python
# Note: When running from the modeling directory, the path is already set up
# by preprocess.py. If running from project root, adjust imports accordingly.

from modeling.run import ModelRunner
from modeling.random_forest import RandomForestRegressorModel
from bubble_type import BUBBLE_TYPE

# Define features and target
feature_cols = ['prc', 'mktcap', 'ps', 'pb', 'pe', 'ev', 'ev_sales', 
                'ps_z', 'pb_z', 'ev_sales_z']
target_col = 'survival_prob'

# Initialize model
rf_model = RandomForestRegressorModel(feature_cols=feature_cols, 
                                      target_col=target_col)
runner = ModelRunner(model=rf_model)

# Train on historical bubbles
train_data = runner.run_preprocessing([
    BUBBLE_TYPE.CRYPTO.value, 
    BUBBLE_TYPE.DOTCOM.value, 
    BUBBLE_TYPE.EV.value
])
rf_model = runner.run_training(train_data)

# Predict for AI bubble
ai_data = runner.run_preprocessing([BUBBLE_TYPE.AI.value], is_train=False)
predictions = runner.run_inference(ai_data[feature_cols])
```

## Methodology

### Bubble Index Calculation

For each company, the bubble index is computed as:

1. **Valuation Metrics**: Calculate P/S, P/B, P/E, and EV/Sales ratios
2. **Z-Score Normalization**: Normalize each metric by company-specific mean and standard deviation
3. **Aggregate Index**: Average of all normalized z-scores

```
bubble_index = mean(ps_z, pb_z, pe_z, ev_sales_z)
```

A bubble index > 2 indicates potential overvaluation relative to historical norms.

### Survival Probability

Survival probability is calculated based on price retention during the bubble burst:

1. **Identify Bubble Period**: Peak (rolling maximum) to 30% drawdown
2. **Calculate Drop Percentage**: `(price_start - price_end) / price_start`
3. **Convert to Survival Metric**: `survival_percentage = -drop_percentage` (companies with smaller drops have higher values)
4. **Normalize**: Scale to [0, 1] range across all companies using min-max normalization

Companies with higher survival probabilities are those that maintained more of their value (dropped less) during the bubble burst period.

### Model Features

The Random Forest model uses the following features:
- `prc`: Stock price
- `mktcap`: Market capitalization
- `ps`, `pb`, `pe`: Price-to-sales, price-to-book, price-to-earnings ratios
- `ev`, `ev_sales`: Enterprise value and EV-to-sales ratio
- `ps_z`, `pb_z`, `ev_sales_z`: Z-score normalized valuation metrics

## Bubble Types Analyzed

### 1. **Dotcom Bubble (1998-2000)**
- Period: January 1998 - March 2000
- Companies: Internet and technology companies (Amazon, Cisco, Yahoo, etc.)
- Characteristics: Rapid valuation growth followed by sharp correction

### 2. **Crypto Bubble (2016-2018)**
- Period: January 2016 - December 2018
- Companies: Cryptocurrency exchanges, mining companies, blockchain-related firms
- Characteristics: Extreme volatility and speculative trading

### 3. **EV Bubble (2017-2022)**
- Period: January 2017 - December 2022
- Companies: Electric vehicle manufacturers (Tesla, NIO, Rivian, etc.)
- Characteristics: Growth in EV adoption and market expansion

### 4. **AI Bubble (2020-Present)**
- Period: January 2020 - November 2025
- Companies: AI/ML companies, chip manufacturers, cloud providers
- Characteristics: Generative AI boom, GPU demand surge, LLM development

## Model Details

### Random Forest Regressor

- **Algorithm**: Scikit-learn RandomForestRegressor
- **Target**: Survival probability (continuous, 0-1)
- **Features**: 10 valuation and normalized metrics
- **Training Data**: Historical bubbles (Crypto, Dotcom, EV)
- **Prediction Target**: Current AI bubble companies

### Preprocessing Pipeline

1. **Date Filtering**: Extract data within bubble date ranges
2. **Bubble Detection**: Identify peak and 30% drawdown points
3. **Aggregation**: Average metrics per company over bubble period
4. **Survival Calculation**: Compute survival probabilities for training data
5. **Feature Engineering**: Calculate normalized z-scores

## File Descriptions

### Core Scripts

- **`bubble_data_collector.py`**: Main data collection script that queries WRDS and generates merged datasets
- **`bubble_type.py`**: Enum class for managing bubble type constants
- **`bubble_config.json`**: Configuration file defining all bubble parameters

### Visualization

- **`bubble_plotting_updated.py`**: Interactive plotting functions with Jupyter widgets
- **`plotting_bubble_data.py`**: Original static plotting scripts

### Modeling

- **`modeling/preprocess.py`**: Data preprocessing, bubble detection, and survival probability calculation
- **`modeling/random_forest.py`**: Random Forest regression model wrapper
- **`modeling/run.py`**: Main script for training and prediction pipeline

### Data

- **`data/*_merged.csv`**: Generated datasets containing:
  - Daily stock prices (CRSP)
  - Quarterly fundamentals (Compustat)
  - Calculated valuation metrics
  - Bubble indices

## Notes

- **WRDS Access**: Requires active WRDS subscription and credentials
- **Data Updates**: Historical data is static; update dates in config for new data
- **Model Limitations**: Predictions are based on historical patterns and may not account for unique factors in current markets
- **Bubble Definition**: Bubble periods are defined by peak-to-drawdown methodology; adjust thresholds in `preprocess.py` if needed

## License

This project is part of ECE 143 coursework (Group 17).

## Contact

For questions or issues, please refer to the course materials or contact the project team.
