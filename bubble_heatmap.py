import pandas as pd
import glob
import numpy as np
import plotly.graph_objects as go
import os
from IPython.display import display, Image

# ------------------------------
# Settings
# ------------------------------
SAVE_DIR = "static_heatmaps"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------------
# Save Plotly figure as PNG and display inline
# ------------------------------
def save_and_display_plotly(fig, name):
    path = os.path.join(SAVE_DIR, f"{name}.png")
    fig.write_image(path, scale=2)  # scale=2 for high-res
    display(Image(filename=path))
    print(f"[Saved and displayed] {path}")

# ------------------------------
# Static Heatmap Function
# ------------------------------
def plot_static_bubble_heatmaps(merged: pd.DataFrame):
    required_cols = {'bubble_type', 'tic', 'date', 'bubble_index'}
    assert required_cols.issubset(merged.columns), f"Missing columns: {required_cols - set(merged.columns)}"

    bubble_types = merged['bubble_type'].unique()

    for bubble in bubble_types:
        df_bubble = merged[merged['bubble_type'] == bubble].copy()
        pivot = df_bubble.pivot_table(index='tic', columns='date', values='bubble_index', fill_value=0)
        z = pivot.values.astype(float)
        zmin_val = 0
        zmax_val = max(np.nanmax(z) if not np.isnan(np.nanmax(z)) else 0, 2.0)

        # Original color scheme
        colorscale = [
            [0.0, 'rgb(255,255,255)'],
            [2.0 / zmax_val, 'rgb(0,0,180)']
        ]
        if zmax_val > 2.0:
            colorscale.append([2.0 / zmax_val, 'rgb(255,0,0)'])
            colorscale.append([1.0, 'rgb(139,0,0)'])
        else:
            colorscale.append([1.0, 'rgb(255,0,0)'])

        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=z,
            x=pivot.columns,
            y=pivot.index,
            colorscale=colorscale,
            zmin=zmin_val,
            zmax=zmax_val,
            text=z,
            texttemplate="%{text:.2f}",
            hovertemplate="Company: %{y}<br>Date: %{x}<br>Bubble Index: %{z:.2f}<extra></extra>"
        ))

        fig.update_layout(
            title=f"{bubble} Bubble Index Heatmap",
            xaxis_title="Date",
            yaxis_title="Company",
            template="plotly_white"
        )

        save_and_display_plotly(fig, f"heatmap_{bubble}")

# ------------------------------
# Load CSVs and run
# ------------------------------
if __name__ == "__main__":
    data_dir = "data"
    csv_files = glob.glob(f"{data_dir}/*_merged.csv")
    assert csv_files, f"No CSV files found in {data_dir}"

    merged_list = []
    for file in csv_files:
        df = pd.read_csv(file, parse_dates=['date', 'datadate'])
        bubble_type = os.path.basename(file).replace("_merged.csv", "")
        df['bubble_type'] = bubble_type
        merged_list.append(df)
    merged = pd.concat(merged_list, ignore_index=True)
    assert not merged.empty, "Merged DataFrame is empty"

    plot_static_bubble_heatmaps(merged)


