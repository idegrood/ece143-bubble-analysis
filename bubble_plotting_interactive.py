# bubble_dashboard.py

import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact
from bubble_data_collector import load_bubble_config
# ---------------------------------------------------------
# Bubble plotting function
# ---------------------------------------------------------
def plot_bubble_interactive(merged: pd.DataFrame, bubble_type: str, company: str = 'All companies', config=None):
    """
    Plot bubble analysis for a single company or all companies.

    Parameters
    ----------
    merged : pd.DataFrame
        Merged dataset containing price, sales, bubble indexes, etc.
    bubble_type : str
        The bubble category to filter by.
    company : str, default 'All companies'
        Specific ticker symbol to visualize, or choose 'All companies'.
    """

    required_cols = [
        "bubble_type", "tic", "date", "prc", "saleq", "mktcap",
        "bubble_index", "conm"
    ]
    for col in required_cols:
        assert col in merged.columns, f"Missing required column: {col}"

    assert isinstance(bubble_type, str), "bubble_type must be a string."
    assert isinstance(company, str), "company must be a string."

    assert bubble_type in merged["bubble_type"].unique(), \
        f"Invalid bubble_type '{bubble_type}'."

    df = merged[merged['bubble_type'] == bubble_type].copy()
    if bubble_type in config:
      start = config[bubble_type]["start_date"]
      end = config[bubble_type]["end_date"]
      df = df[(df['date'] >= start) & (df['date'] <= end)]
    if df.empty:
        print("No data available for selected bubble type.")
        return

    # ---------------------------
    # Case 1: All companies
    # ---------------------------
    if company == 'All companies':
        ts = df.groupby('date')['bubble_index'].max()
        ts_m = df.groupby('date')['bubble_index'].mean()

        plt.figure(figsize=(12, 5))
        plt.plot(ts, label=f"{bubble_type.upper()} Bubble Index (max)")
        plt.plot(ts_m, label=f"{bubble_type.upper()} Bubble Index (mean)")
        plt.axhline(0, linestyle='--')
        plt.axhline(2, color='red', linestyle='--', label='Bubble Threshold')
        plt.title(f"Cumulative Bubble Index: {bubble_type.upper()}")
        plt.xlabel("Date")
        plt.ylabel("Bubble Index (Z-score)")
        plt.grid(True)
        plt.legend()
        plt.show()
        return

    # ---------------------------
    # Case 2: Specific company
    # ---------------------------
    comp_df = df[df['tic'] == company].sort_values('date')

    if comp_df.empty:
        print("No data available for this company.")
        return

    comp_df['saleq_interp'] = comp_df['saleq'].interpolate(method='linear')
    company_name = comp_df['conm'].iloc[0]

    # Price
    plt.figure(figsize=(12, 5))
    plt.plot(comp_df['date'], comp_df['prc'], linewidth=2)
    plt.title(f"{company_name}: Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()

    # Sales (Interpolated)
    plt.figure(figsize=(12, 5))
    plt.plot(comp_df['date'], comp_df['saleq_interp'], linewidth=2)
    plt.title(f"{company_name}: Sales (Quarterly, Interpolated)")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.grid(True)
    plt.show()

    # Combined price & sales
    plt.figure(figsize=(12, 5))
    plt.plot(comp_df['date'], comp_df['prc'], label='Price', linewidth=2)
    plt.plot(comp_df['date'], comp_df['saleq_interp'], label='Sales', linewidth=2)
    plt.title(f"{company_name}: Price vs Sales")
    plt.xlabel("Date")
    plt.ylabel("Price / Sales")
    plt.legend()
    plt.grid(True)
    plt.show()

    # P/S ratio
    comp_df['ps_ratio'] = comp_df['mktcap'] / comp_df['saleq_interp']
    comp_df['ps_ratio'] = comp_df['ps_ratio'].replace([np.inf, -np.inf], np.nan)

    threshold = np.nanmean(comp_df['ps_ratio']) + 2 * np.nanstd(comp_df['ps_ratio'])

    plt.figure(figsize=(12, 5))
    plt.plot(comp_df['date'], comp_df['ps_ratio'], label='P/S Ratio', linewidth=2)
    plt.axhline(threshold, color='red', linestyle='--', label='Z=2 Threshold')
    ymin, ymax = plt.ylim()
    plt.ylim(min(ymin, threshold * 0.95), max(ymax, threshold * 1.05))
    plt.title(f"{company_name}: P/S Ratio with Bubble Threshold")
    plt.xlabel("Date")
    plt.ylabel("P/S Ratio")
    plt.legend()
    plt.grid(True)
    plt.show()


# ---------------------------------------------------------
# Load merged CSVs
# ---------------------------------------------------------
def load_merged_data(data_dir="data"):
    """
    Load all CSV files in the given directory ending with '_merged.csv'.

    Parameters
    ----------
    data_dir : str, default "data"
        Directory containing *_merged.csv files.

    Returns
    -------
    pd.DataFrame
        Combined dataset with an added 'bubble_type' column.
    """

    assert isinstance(data_dir, str), "data_dir must be a string."

    csv_files = glob.glob(f"{data_dir}/*_merged.csv")
    assert len(csv_files) > 0, f"No *_merged.csv files found in '{data_dir}'"

    merged_list = []

    for file in csv_files:
        bubble_type = file.split('/')[-1].replace("_merged.csv", "")
        df = pd.read_csv(file, parse_dates=['date', 'datadate'])
        df['bubble_type'] = bubble_type
        merged_list.append(df)

    merged = pd.concat(merged_list, ignore_index=True)
    assert not merged.empty, "Merged dataset is empty."

    return merged


def create_dashboard(merged, config):
    """
    Create an interactive dashboard with dropdowns
    for bubble types and companies.

    Parameters
    ----------
    merged : pd.DataFrame
        Full dataset with bubble_type annotations.
    """

    assert isinstance(merged, pd.DataFrame), "merged must be a DataFrame."
    assert "bubble_type" in merged.columns, "merged must contain 'bubble_type' column."

    def get_companies_for_bubble(df, bubble):
        """Return list of companies for a given bubble type."""
        df_b = df[df['bubble_type'] == bubble]
        return ['All companies'] + sorted(df_b['tic'].dropna().unique().tolist())

    bubble_dropdown = widgets.Dropdown(
        options=merged['bubble_type'].unique(),
        description='Bubble:',
        value=merged['bubble_type'].unique()[0]
    )

    company_dropdown = widgets.Dropdown(
        options=get_companies_for_bubble(merged, bubble_dropdown.value),
        description='Company:',
        value='All companies'
    )

    def update_company_options(change):
        """Update available companies when bubble type changes."""
        company_dropdown.options = get_companies_for_bubble(merged, change['new'])
        company_dropdown.value = 'All companies'

    bubble_dropdown.observe(update_company_options, names='value')

    interact(
        plot_bubble_interactive,
        merged=widgets.fixed(merged),
        bubble_type=bubble_dropdown,
        company=company_dropdown,
        config=widgets.fixed(config),
    )


if __name__ == "__main__":
    merged = load_merged_data("data")
    config = load_bubble_config("bubble_config.json")
    create_dashboard(merged, config)

