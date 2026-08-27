"""
Sample Visualizer
-----------------
Usage:
    python visualizer.py sample.csv
    python visualizer.py sample.csv --output plot.png
    python visualizer.py Sample_OD42L50.csv --output sobol_plot.png    
    python visualizer.py sample.csv --dpi 150 --title "My Sample"
    python visualizer.py ../data_folder/1_param/FDData_SobS_OD40L50G3_Shell_Fixed_PD.csv --output sobol_plot_new_OD40L50G3.png 
    python visualizer.py ../data_folder/1_param/Complete_Sobol.csv --output Complete_Sobol_PREPRES.png 

Produces a scatter matrix (pair plot) showing pairwise distributions
of every parameter, with marginal histograms on the diagonal.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

# ── palette ─────────────────────────────────────────────────
BG          = "#0f1117"
PANEL_BG    = "#161b27"
GRID_COL    = "#1e2a3a"
TEXT_COL    = "#cdd6f4"
SCATTER_COL = "#4fc3f7"
HIST_COL    = "#1e88e5"
ACCENT      = "#4fc3f7"

# ── column whitelist ─────────────────────────────────────────
# List the CSV column names you want to appear in the scatter matrix.
# Order here controls the order of rows/columns in the plot.
# Set to None (or an empty list) to show every column in the CSV.
COLUMN_WHITELIST = [
    "R",
    "A",
    "CC",
    "VC",
    # "T",
    # "AUC",
    # "CFE",
    # "Volume",
    # "SEA",
    # "Obj",
]


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def style_axis(ax, xlabel="", ylabel=""):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=7, length=3)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
        spine.set_linewidth(0.8)
    ax.grid(True, color=GRID_COL, linewidth=0.5, linestyle="--", alpha=0.6)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, labelpad=4)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, labelpad=4)


def draw_scatter(ax, x, y):
    ax.scatter(x, y, c=SCATTER_COL, s=12, alpha=0.7, linewidths=0, zorder=3)


def draw_hist(ax, values, col_name):
    clean = values.dropna()

    # Treat a column as discrete integers if it is an integer dtype OR if every
    # non-null value is a whole number (handles float64 columns like CC and VC
    # that contain only whole numbers but are stored as float).
    is_int_valued = (
        pd.api.types.is_integer_dtype(clean)
        or (pd.api.types.is_float_dtype(clean) and (clean % 1 == 0).all())
    )

    if is_int_valued:
        # Place bin edges at half-integers so each bar is exactly 1 unit wide,
        # perfectly centred on its integer value, with no gaps between bars.
        unique_ints = sorted(clean.astype(int).unique())
        bins = [v - 0.5 for v in unique_ints] + [unique_ints[-1] + 0.5]
    else:
        bins = max(6, len(clean) // 10)

    ax.hist(clean, bins=bins, color=HIST_COL, alpha=0.85,
            edgecolor=PANEL_BG, linewidth=0.4, zorder=3)
    ax.set_xlabel(col_name, fontsize=8, labelpad=4)


def build_figure(df: pd.DataFrame, title: str) -> plt.Figure:
    cols = df.columns.tolist()
    n    = len(cols)

    fig_w = max(8, n * 1.8)
    fig_h = max(7, n * 1.8 + 0.8)
    fig   = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)

    gs = GridSpec(n, n, figure=fig,
                  left=0.07, right=0.97, top=0.91, bottom=0.07,
                  hspace=0.08, wspace=0.08)

    for row in range(n):
        for col in range(n):
            ax = fig.add_subplot(gs[row, col])
            style_axis(ax)

            if row == col:
                draw_hist(ax, df[cols[row]], cols[row])
                for spine in ax.spines.values():
                    spine.set_edgecolor(ACCENT)
                    spine.set_linewidth(1.0)
            else:
                draw_scatter(ax, df[cols[col]], df[cols[row]])

            # Labels on outer edges only
            if row == n - 1:
                ax.set_xlabel(cols[col], fontsize=8, labelpad=4, color=TEXT_COL)
            else:
                ax.set_xticklabels([])

            if col == 0:
                ax.set_ylabel(cols[row], fontsize=8, labelpad=4, color=TEXT_COL)
            else:
                ax.set_yticklabels([])

            ax.xaxis.set_major_locator(ticker.MaxNLocator(3, prune="both"))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(3, prune="both"))

    fig.suptitle(f"{title}  —  {len(df)} samples · {n} parameters",
                 color=TEXT_COL, fontsize=12, fontweight="bold", y=0.97)

    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualise a sample CSV as a scatter matrix.")
    parser.add_argument("csv",              help="Path to the CSV file.")
    parser.add_argument("--output", "-o",   default=None,
                        help="Output image path (e.g. plot.png). Displays interactively if omitted.")
    parser.add_argument("--dpi",            type=int, default=120,
                        help="Resolution of saved image (default: 120).")
    parser.add_argument("--title",          default="Sample Distribution",
                        help="Plot title.")
    args = parser.parse_args()

    df = load_data(args.csv)
    print(f"Loaded {len(df)} samples, columns: {list(df.columns)}")

    if COLUMN_WHITELIST:
        missing = [c for c in COLUMN_WHITELIST if c not in df.columns]
        if missing:
            print(f"Warning: whitelist columns not found in CSV and will be skipped: {missing}")
        keep = [c for c in COLUMN_WHITELIST if c in df.columns]
        df   = df[keep]
        print(f"Whitelist applied — visualising: {keep}")

    fig = build_figure(df, args.title)

    if args.output:
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor=BG)
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()