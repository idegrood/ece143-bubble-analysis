import matplotlib.pyplot as plt
import pandas as pd
import glob
import numpy as np
import ipywidgets as widgets
from ipywidgets import interact
import plotly.graph_objects as go

# ------------------------------
# Line plot function
# ------------------------------
def plot_bubble_interactive(merged: pd.DataFrame, bubble_type: str, company: str = 'All companies'):
    """
    Interactive plotting of bubble analysis for companies.
    """
    required_columns = ['date', 'tic', 'conm', 'bubble_type', 'bubble_index', 'prc', 'saleq', 'mktcap']
    for col in required_columns:
        assert col in merged.columns, f"merged must contain column '{col}'"

    df = merged[merged['bubble_type'] == bubble_type].copy()
    if df.empty:
        print(f"No data for bubble {bubble_type}")
        return

    if company == 'All companies':
        ts = df.groupby('date')['bubble_index'].max()
        ts_m = df.groupby('date')['bubble_index'].mean()
        plt.figure(figsize=(12,5))
        plt.plot(ts,  label=f"{bubble_type.upper()} Bubble Index (max)")
        plt.plot(ts_m,  label=f"{bubble_type.upper()} Bubble Index (mean)")
        plt.axhline(0, linestyle='--')
        plt.axhline(2, color='red', linestyle='--', label='Bubble Threshold')
        plt.title(f"Cumulative Bubble Index: {bubble_type.upper()}")
        plt.xlabel("Date")
        plt.ylabel("Bubble Index (Z-score)")
        plt.legend()
        plt.grid(True)
        plt.show()
        return

    comp_df = df[df['tic'] == company].sort_values('date')
    if comp_df.empty:
        print(f"No data for company {company} in bubble {bubble_type}")
        return
    company_name = comp_df['conm'].iloc[0]
    comp_df['saleq_interp'] = comp_df['saleq'].interpolate(method='linear')

    # Price-only
    plt.figure(figsize=(12,5))
    plt.plot(comp_df['date'], comp_df['prc'], linewidth=2)
    plt.title(f"{company_name}: Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()

    # Sales-only
    plt.figure(figsize=(12,5))
    plt.plot(comp_df['date'], comp_df['saleq_interp'], linewidth=2)
    plt.title(f"{company_name}: Sales (Interpolated) Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.grid(True)
    plt.show()

    # Combined
    plt.figure(figsize=(12,5))
    plt.plot(comp_df['date'], comp_df['prc'], label='Price', linewidth=2)
    plt.plot(comp_df['date'], comp_df['saleq_interp'], label='Sales (Interpolated)', linewidth=2)
    plt.title(f"{company_name}: Price vs Sales")
    plt.xlabel("Date")
    plt.ylabel("Price / Sales")
    plt.legend()
    plt.grid(True)
    plt.show()

    # P/S ratio
    comp_df['ps_ratio'] = comp_df['mktcap'] / comp_df['saleq_interp']
    comp_df['ps_ratio'] = comp_df['ps_ratio'].replace([np.inf, -np.inf], np.nan)
    threshold = np.nanmean(comp_df['ps_ratio']) + 2*np.nanstd(comp_df['ps_ratio'])
    plt.figure(figsize=(12,5))
    plt.plot(comp_df['date'], comp_df['ps_ratio'], label='P/S Ratio', linewidth=2)
    plt.axhline(threshold, color='red', linestyle='--', label='Z=2 Threshold')
    ymin, ymax = plt.ylim()
    plt.ylim(min(ymin, threshold*0.95), max(ymax, threshold*1.05))
    plt.title(f"{company_name}: P/S Ratio with Bubble Threshold")
    plt.xlabel("Date")
    plt.ylabel("P/S Ratio")
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f"Company: {company_name}")





# ------------------------------
# Self-executing block
# ------------------------------
if __name__ == "__main__":
    # Load CSVs
    data_dir = "data"
    csv_files = glob.glob(f"{data_dir}/*_merged.csv")
    assert csv_files, f"No CSV files found in {data_dir}"

    merged_list = []
    for file in csv_files:
        df = pd.read_csv(file, parse_dates=['date','datadate'])
        bubble_type = file.split('/')[-1].replace("_merged.csv","")
        df['bubble_type'] = bubble_type
        merged_list.append(df)
    merged = pd.concat(merged_list, ignore_index=True)
    assert not merged.empty, "Merged DataFrame is empty"

    # ------------------------------
    # Widgets for line plots
    # ------------------------------
    def get_companies(df, bubble):
        return ['All companies'] + sorted(df[df['bubble_type']==bubble]['tic'].dropna().unique().tolist())

    initial_bubble = merged['bubble_type'].unique()[0]
    bubble_dropdown = widgets.Dropdown(options=merged['bubble_type'].unique(), description='Bubble:', value=initial_bubble)
    company_dropdown = widgets.Dropdown(options=get_companies(merged, initial_bubble), description='Company:', value='All companies')

    def update_company_options(change):
        new_bubble = change['new']
        company_dropdown.options = get_companies(merged, new_bubble)
        company_dropdown.value = 'All companies'
    bubble_dropdown.observe(update_company_options, names='value')

    interact(plot_bubble_interactive, merged=widgets.fixed(merged), bubble_type=bubble_dropdown, company=company_dropdown)

  
