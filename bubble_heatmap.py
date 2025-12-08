
import pandas as pd
import glob
import numpy as np
import plotly.graph_objects as go

# ------------------------------
# Heatmap function (no Kaleido)
# ------------------------------
def plot_interactive_bubble_heatmap(merged: pd.DataFrame):
    """
    Plotly interactive bubble index heatmap across companies with buttons to toggle bubble type.
    """
    required_cols = {'bubble_type', 'tic', 'date', 'bubble_index'}
    assert not merged.empty, "Input DataFrame 'merged' is empty"
    assert required_cols.issubset(merged.columns), f"DataFrame missing columns: {required_cols - set(merged.columns)}"

    bubble_types = merged['bubble_type'].unique()
    fig = go.Figure()

    # Add a heatmap trace for each bubble type
    for i, bubble in enumerate(bubble_types):
        df_bubble = merged[merged['bubble_type']==bubble].copy()
        pivot = df_bubble.pivot_table(index='tic', columns='date', values='bubble_index', fill_value=0)
        pivot = pivot.sort_index()
        z = pivot.values.astype(float)
        zmin_val = 0
        zmax_val = max(np.nanmax(z) if not np.isnan(np.nanmax(z)) else 0, 2.0)

        colorscale = [
            [0.0, 'rgb(255,255,255)'],
            [2.0 / zmax_val, 'rgb(0,0,180)'],
            [1.0, 'rgb(255,0,0)']
        ]

        fig.add_trace(go.Heatmap(
            z=z,
            x=pivot.columns,
            y=pivot.index,
            visible=(i==0),
            colorscale=colorscale,
            zmin=zmin_val,
            zmax=zmax_val,
            text=z,
            texttemplate="%{text:.2f}",
            hovertemplate="Company: %{y}<br>Date: %{x}<br>Bubble Index: %{z:.2f}<extra></extra>"
        ))

    # Create buttons to switch between bubble types
    buttons = []
    for i, bubble in enumerate(bubble_types):
        buttons.append(dict(
            label=bubble,
            method='update',
            args=[{'visible':[j==i for j in range(len(bubble_types))]},
                  {'title':f"Bubble Index Heatmap: {bubble}"}]
        ))

    fig.update_layout(
        updatemenus=[dict(active=0, buttons=buttons, x=0.1, y=1.15, xanchor='left', yanchor='top')],
        title=f"Bubble Index Heatmap: {bubble_types[0]}",
        xaxis_title="Date",
        yaxis_title="Company",
        template="plotly_white"
    )

    fig.show()  # Only show, no image export needed

# ------------------------------
# Self-executing block
# ------------------------------
if __name__ == "__main__":
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

    plot_interactive_bubble_heatmap(merged)


