import plotly.graph_objects as go
import pandas as pd
import glob
import numpy as np
import os
from IPython.display import display, Image

# Detect interactive environment (Colab)
try:
    import ipywidgets as widgets
    from ipywidgets import interact
    IN_INTERACTIVE_ENV = True
except ImportError:
    IN_INTERACTIVE_ENV = False

# Directory to save static images for GitHub
STATIC_DIR = "static_heatmaps"
os.makedirs(STATIC_DIR, exist_ok=True)

def save_static_plot(fig, name):
    path = os.path.join(STATIC_DIR, f"{name}.png")
    # Use kaleido-free method
    fig.write_image(path, engine="orca") if hasattr(fig, "write_image") else fig.write_html(path.replace(".png",".html"))
    display(Image(filename=path))
    print(f"[Saved static plot] {path}")

def plot_interactive_bubble_heatmap(merged: pd.DataFrame):
    """
    Plotly heatmap with dropdown for Colab; static images for GitHub.
    """
    bubble_types = merged['bubble_type'].unique()

    if IN_INTERACTIVE_ENV:
        # ------------------------------
        # Interactive version for Colab
        # ------------------------------
        fig = go.Figure()
        for i, bubble in enumerate(bubble_types):
            df_bubble = merged[merged['bubble_type']==bubble].copy()
            pivot = df_bubble.pivot_table(index='tic', columns='date', values='bubble_index', fill_value=0)
            fig.add_trace(go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                visible=(i==0)
            ))

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
            yaxis_title="Company"
        )
        fig.show()

    else:
        # ------------------------------
        # GitHub / static version
        # ------------------------------
        import matplotlib.pyplot as plt
        for bubble in bubble_types:
            df_bubble = merged[merged['bubble_type']==bubble].copy()
            pivot = df_bubble.pivot_table(index='tic', columns='date', values='bubble_index', fill_value=0)
            pivot = pivot.sort_index()

            fig, ax = plt.subplots(figsize=(12, max(4, len(pivot)//3)))
            cax = ax.imshow(pivot.values, aspect='auto', cmap='RdBu_r', vmin=0, vmax=max(2.0, np.nanmax(pivot.values)))

            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels([d.strftime('%Y-%m-%d') for d in pivot.columns], rotation=90)
            ax.set_title(f"Bubble Index Heatmap: {bubble}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Company")
            fig.colorbar(cax, ax=ax, label='Bubble Index')

            path = os.path.join(STATIC_DIR, f"heatmap_{bubble}.png")
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            display(Image(filename=path))
            print(f"[Saved static plot] {path}")

# ------------------------------
# Run if main
# ------------------------------
if __name__ == "__main__":
    data_dir = "data"
    csv_files = glob.glob(f"{data_dir}/*_merged.csv")
    merged_list = []
    for file in csv_files:
        df = pd.read_csv(file, parse_dates=['date','datadate'])
        bubble_type = os.path.basename(file).replace("_merged.csv","")
        df['bubble_type'] = bubble_type
        merged_list.append(df)
    merged = pd.concat(merged_list, ignore_index=True)

    plot_interactive_bubble_heatmap(merged)

