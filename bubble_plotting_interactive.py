import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
from IPython.display import display, Image

# ------------------------------
# Settings
# ------------------------------
SAVE_DIR = "static_plots"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------------
# Helper: Save and display PNG
# ------------------------------
def save_and_display(fig, name):
    path = os.path.join(SAVE_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    display(Image(filename=path))  # embed in notebook

# ------------------------------
# Plot one representative plot for GitHub
# ------------------------------
def plot_one_for_github(df, bubble_type):
    df_bubble = df[df['bubble_type']==bubble_type]
    ts_max = df_bubble.groupby("date")['bubble_index'].max()
    ts_mean = df_bubble.groupby("date")['bubble_index'].mean()
    
    fig = plt.figure(figsize=(12,5))
    plt.plot(ts_max, label="Bubble Index (Max)")
    plt.plot(ts_mean, label="Bubble Index (Mean)")
    plt.axhline(0, linestyle='--')
    plt.axhline(2, color='red', linestyle='--', label="Bubble Threshold")
    plt.title(f"{bubble_type.upper()} Bubble Index (All Companies)")
    plt.xlabel("Date")
    plt.ylabel("Bubble Index")
    plt.legend()
    plt.grid(True)
    
    save_and_display(fig, f"{bubble_type}_all_companies")

# ------------------------------
# Load CSVs
# ------------------------------
data_dir = "data"
csv_files = glob.glob(f"{data_dir}/*_merged.csv")
merged_list = []
for file in csv_files:
    df = pd.read_csv(file, parse_dates=['date','datadate'])
    bubble_type = os.path.basename(file).replace("_merged.csv","")
    df['bubble_type'] = bubble_type
    merged_list.append(df)
merged = pd.concat(merged_list, ignore_index=True)

# ------------------------------
# Generate one plot per bubble for GitHub
# ------------------------------
for bubble in merged['bubble_type'].unique():
    plot_one_for_github(merged, bubble)

# ------------------------------
# Keep interactive widgets for Colab
# ------------------------------
try:
    import ipywidgets as widgets
    from ipywidgets import interact
    initial_bubble = merged['bubble_type'].unique()[0]
    bubble_dropdown = widgets.Dropdown(
        options=merged['bubble_type'].unique(),
        description='Bubble:',
        value=initial_bubble
    )
    company_dropdown = widgets.Dropdown(
        options=['All companies'] + sorted(merged[merged['bubble_type']==initial_bubble]['tic'].dropna().tolist()),
        description='Company:',
        value='All companies'
    )

    def update_companies(change):
        new_bubble = change['new']
        company_dropdown.options = ['All companies'] + sorted(merged[merged['bubble_type']==new_bubble]['tic'].dropna().tolist())
        company_dropdown.value = 'All companies'

    bubble_dropdown.observe(update_companies, names='value')
    interact(lambda bubble, company: plot_one_for_github(merged, bubble),
             bubble=bubble_dropdown, company=company_dropdown)

except ImportError:
    pass
