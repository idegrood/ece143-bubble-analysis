# ECE 143 - Bubble Analysis Project: AI - Boom or Bust

**Group 17**

A data analysis framework for identifying and predicting economic bubbles across multiple technology sectors. This project analyzes historical bubbles (Dotcom, Crypto, EV) to predict company survival probabilities during the current AI bubble.

## File Structure

```
ece143-bubble-analysis/
├── bubble_config.json          # Configuration for all bubble types (keywords, tickers, dates)
├── bubble_data_collector.py    # Main data collection script (queries WRDS)
├── bubble_type.py              # Enum class for bubble type constants
├── bubble_plotting_updated.py  # Interactive plotting functions with Jupyter widgets
├── plotting_bubble_data.py     # Static plotting scripts
├── data/                       # Generated CSV datasets
│   ├── ai_merged.csv
│   ├── crypto_merged.csv
│   ├── dotcom_merged.csv
│   └── ev_merged.csv
└── modeling/                   # Machine learning pipeline
    ├── preprocess.py           # Data preprocessing and survival probability calculation
    ├── random_forest.py        # Random Forest regression model wrapper
    └── run.py                  # Main training and prediction script
```

### File Descriptions

**Core Scripts:**
- `bubble_data_collector.py`: Collects daily stock prices (CRSP) and quarterly fundamentals (Compustat) from WRDS, merges data, calculates valuation metrics and bubble indices
- `bubble_type.py`: Enum class defining bubble types (AI, CRYPTO, EV, DOTCOM)
- `bubble_config.json`: JSON configuration file with company keywords, tickers, names, date ranges, and data paths for each bubble type

**Visualization:**
- `bubble_plotting_updated.py`: Interactive plotting functions with Jupyter notebook widgets for exploring bubble data
- `plotting_bubble_data.py`: Static plotting scripts for bubble index and company-specific metrics

**Modeling:**
- `modeling/preprocess.py`: Preprocesses data, identifies bubble bounds (peak to 30% drawdown), calculates survival probabilities
- `modeling/random_forest.py`: Wrapper class for scikit-learn RandomForestRegressor
- `modeling/run.py`: Main script that trains model on historical bubbles and predicts AI company survival

**Data:**
- `data/*_merged.csv`: Generated datasets containing daily stock prices, quarterly fundamentals, calculated valuation metrics (P/S, P/B, P/E, EV/Sales), and bubble indices

## Third-Party Modules

The following third-party Python packages are required:

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Plotting and visualization
- **scikit-learn**: Machine learning (RandomForestRegressor, metrics)
- **wrds**: WRDS Python API for accessing CRSP and Compustat databases
- **ipywidgets**: Interactive widgets for Jupyter notebooks

### Installation

```bash
pip install pandas numpy matplotlib scikit-learn wrds ipywidgets
```

**Note:** WRDS access requires an active subscription. First-time users will be prompted for WRDS username and password when running the data collection script.

## How to Run

### 1. Data Collection

Collect data for all bubbles defined in `bubble_config.json`:

```bash
python bubble_data_collector.py
```

This script:
- Connects to WRDS
- For each bubble type in the config:
  - Searches Compustat for matching companies
  - Links to CRSP for stock price data
  - Filters by date range (`start_date` to `end_date` from config)
  - Merges with quarterly fundamentals
  - Calculates valuation metrics and bubble indices
  - Saves to CSV in the `data/` directory

**Date Configuration:** Dates are specified in `bubble_config.json` for each bubble type (format: `"YYYY-MM-DD"`). The script uses these dates to filter CRSP stock price data.

### 2. Data Visualization

#### Interactive Plotting (Jupyter Notebook)

```python
import pandas as pd
from bubble_plotting_updated import plot_bubble_interactive
import ipywidgets as widgets

# Load data (can concatenate multiple bubbles)
merged = pd.read_csv('data/ai_merged.csv')
# Interactive widgets will appear - select bubble type and company from dropdowns
```

#### Static Plotting

```python
from plotting_bubble_data import plotting
import pandas as pd

merged = pd.read_csv('data/ev_merged.csv')
plotting(merged)
```

### 3. Model Training & Prediction

Train a Random Forest model on historical bubbles and predict AI company survival:

```bash
cd modeling
python run.py
```

This script:
1. Preprocesses data from Crypto, Dotcom, and EV bubbles (training set)
2. Identifies bubble bounds (peak to 30% drawdown)
3. Calculates survival probabilities based on price retention
4. Trains Random Forest regressor on valuation metrics
5. Predicts survival probabilities for AI companies
6. Displays feature importances and predictions

**Output:**
- Feature importance rankings
- Predicted survival probabilities for each AI company

## Configuration

The `bubble_config.json` file defines all bubble types with:
- **Keywords**: Search terms for identifying companies in Compustat
- **Tickers**: Stock ticker symbols
- **Names**: Full company names (for exact matching)
- **Date Ranges**: `start_date` and `end_date` for the bubble period
- **Data Paths**: Output CSV file locations

Example:
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

## Methodology Summary

**Bubble Index:** Calculated as the mean of normalized z-scores for P/S, P/B, P/E, and EV/Sales ratios. A bubble index > 2 indicates potential overvaluation.

**Survival Probability:** Based on price retention during bubble burst period (peak to 30% drawdown). Companies that maintained more value (dropped less) have higher survival probabilities.

**Model Features:** Stock price, market cap, valuation ratios (P/S, P/B, P/E, EV/Sales), and their z-score normalized versions.

## Notes

- **WRDS Access**: Requires active WRDS subscription and credentials
- **Data Updates**: Update dates in `bubble_config.json` to collect new data
- **Model Limitations**: Predictions are based on historical patterns and may not account for unique factors in current markets
