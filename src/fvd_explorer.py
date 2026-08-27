"""
FvD Explorer — Superimposed Force vs. Displacement viewer filtered by SEA and/or CFE.

For each dataset folder that has a corresponding _PD.csv, loads that processed data,
filters rows by SEA and/or CFE ranges, matches each surviving row to its raw FvD
CSV in Mesh_Conversion_Sample/<folder>/, and draws all matched curves on a single
superimposed graph.

Usage (from project root)
-------------------------
# Filter by SEA range only
python src/fvd_explorer.py --folder FDData_SobS_OD40L50G3_Shell_Fixed --sea-min 3.5 --sea-max 4.5

# Filter by CFE range only
python src/fvd_explorer.py --folder FDData_SobS_OD40L50G3_Shell_Fixed --cfe-min 0.70 --cfe-max 0.80

# Both ranges simultaneously (intersection)
python src/fvd_explorer.py --folder FDData_SobS_OD40L50G3_Shell_Fixed \\
    --sea-min 3.0 --sea-max 5.0 --cfe-min 0.65 --cfe-max 0.80

# Both ranges but apply only the SEA filter (ignore CFE args)
python src/fvd_explorer.py --folder FDData_SobS_OD40L50G3_Shell_Fixed \\
    --sea-min 3.0 --sea-max 5.0 --cfe-min 0.65 --cfe-max 0.80 --filter-mode sea

# Color each line by its R value (gradient across the R domain [2.0, 8.8])
python src/fvd_explorer.py --folder FDData_SobS_OD40L50G3_Shell_Fixed \\
    --sea-min 3.5 --sea-max 4.5 --color-by R

# Save to saves/ instead of temp/ and cap at 40 curves
python src/fvd_explorer.py --folder FDData_SobS_OD40L50G3_Shell_Fixed \\
    --sea-min 3.5 --sea-max 4.5 --color-by A --save --max-lines 40

# USING #
python src/fvd_explorer.py --folder FDData_SobS_OD40L50G3_Solid --sea-min 5

# See which folders have _PD.csv files available
python src/fvd_explorer.py --list-folders
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# Reuse the filename parser from analysis.py — it handles all known filename
# formats (N6AShellR..., N6ASolidR..., N600AID...R..., N6AShellID...R...).
from analysis import parse_param_string

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SRC_DIR, '..', 'data_folder')
_FVD_BASE = os.path.join(_DATA_DIR, '1_param', 'Mesh_Conversion_Sample')
_PD_BASE  = os.path.join(_DATA_DIR, '1_param')
_OUT_BASE = os.path.join(_DATA_DIR, 'output')


# ---------------------------------------------------------------------------
# Parameter domains — used for color gradient normalization.
# These are the full valid input domains, not the per-run filtered range.
# Using full domains keeps the color scale consistent across different runs.
# ---------------------------------------------------------------------------

PARAM_DOMAINS: dict[str, tuple[float, float]] = {
    "R":  (2.0,  8.8),
    "A":  (30.0, 90.0),
    "CC": (4.0,  22.0),
    "VC": (4.0,  10.0),
}

# Color palette used when no --color-by is given.
# tab20 provides 20 perceptually distinct colors and cycles naturally.
_DEFAULT_CMAP = cm.get_cmap("tab20")

# Colormaps per variable for --color-by.  All are single-hue sequential maps
# so a darker shade always means a higher value.
_VAR_CMAPS: dict[str, str] = {
    "R":  "Reds",
    "A":  "Blues",
    "CC": "Greens",
    "VC": "Oranges",
}

# Clip [0.30, 0.95] of the colormap to avoid invisibly pale low-end colors
# on a white background.
_CMAP_LO, _CMAP_HI = 0.30, 0.95


# ---------------------------------------------------------------------------
# File index
# ---------------------------------------------------------------------------

def _pd_path(folder_name: str) -> str:
    return os.path.join(_PD_BASE, f"{folder_name}_PD.csv")


def _fvd_dir(folder_name: str) -> str:
    return os.path.join(_FVD_BASE, folder_name)


def _make_key(r: float, a: float, cc: float, vc: float, t: int) -> tuple:
    """
    Build a hashable lookup key from parameter values.

    R and A are rounded to 2 decimal places to absorb the /100 integer
    representation used in filenames (e.g. R200 -> 2.00).  CC and VC are
    always whole numbers in practice, so they are cast to int.  T (solid/shell
    twist flag) defaults to 0 when absent from the filename.
    """
    return (round(r, 2), round(a, 2), int(round(cc)), int(round(vc)), int(t))


def build_fvd_index(folder_name: str) -> dict[tuple, str]:
    """
    Scan all CSV files in Mesh_Conversion_Sample/<folder_name>/ and build a
    parameter-keyed index:  (R, A, CC, VC, T) -> absolute filepath.

    parse_param_string() handles every known filename format — the /100 integer
    encoding is built into the parser.  Keys with missing R/A/CC/VC are skipped
    with a warning.
    """
    fvd_dir = _fvd_dir(folder_name)
    if not os.path.isdir(fvd_dir):
        raise FileNotFoundError(
            f"FvD directory not found: {fvd_dir}\n"
            f"Run --list-folders to see available options."
        )

    csv_files = [f for f in os.listdir(fvd_dir) if f.lower().endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {fvd_dir}")

    index: dict[tuple, str] = {}
    skipped = 0
    for fname in csv_files:
        stem = os.path.splitext(fname)[0]
        try:
            params = parse_param_string(stem)
        except Exception:
            skipped += 1
            continue

        r  = params.get('R')
        a  = params.get('A')
        cc = params.get('CC')
        vc = params.get('VC')
        if any(v is None for v in (r, a, cc, vc)):
            skipped += 1
            continue

        t   = int(params.get('T', 0))
        key = _make_key(r, a, cc, vc, t)
        index[key] = os.path.join(fvd_dir, fname)

    if skipped:
        print(f"  [index] Skipped {skipped} files (missing required params in name)")

    return index


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def apply_filter(
    df: pd.DataFrame,
    sea_min: float | None,
    sea_max: float | None,
    cfe_min: float | None,
    cfe_max: float | None,
    filter_mode: str,
) -> pd.DataFrame:
    """
    Return rows of df that pass the SEA and/or CFE range filter.

    filter_mode
    -----------
    'sea'  : apply only the SEA range (ignore cfe_min/cfe_max)
    'cfe'  : apply only the CFE range (ignore sea_min/sea_max)
    'both' : intersection — row must satisfy BOTH ranges simultaneously
    """
    mask = pd.Series(True, index=df.index)

    if filter_mode in ('sea', 'both'):
        if sea_min is not None:
            mask &= df['SEA'] >= sea_min
        if sea_max is not None:
            mask &= df['SEA'] <= sea_max

    if filter_mode in ('cfe', 'both'):
        if cfe_min is not None:
            mask &= df['CFE'] >= cfe_min
        if cfe_max is not None:
            mask &= df['CFE'] <= cfe_max

    return df[mask].copy()


def _auto_filter_mode(
    sea_min, sea_max, cfe_min, cfe_max, user_mode: str
) -> str:
    """
    If the user did not explicitly set --filter-mode (default='both') but only
    provided one type of range, automatically narrow to just that type so the
    user doesn't have to think about it.
    """
    has_sea = sea_min is not None or sea_max is not None
    has_cfe = cfe_min is not None or cfe_max is not None

    if user_mode == 'both':
        if has_sea and not has_cfe:
            return 'sea'
        if has_cfe and not has_sea:
            return 'cfe'
    return user_mode


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _var_to_color(val: float, var_name: str) -> tuple:
    """
    Map val to an RGBA color using that variable's known domain and its
    assigned single-hue colormap.  The [_CMAP_LO, _CMAP_HI] slice of the
    colormap is used to avoid pale, low-contrast colors at the low end.
    """
    lo, hi = PARAM_DOMAINS[var_name]
    t = np.clip((val - lo) / (hi - lo), 0.0, 1.0)
    t_mapped = _CMAP_LO + t * (_CMAP_HI - _CMAP_LO)
    return cm.get_cmap(_VAR_CMAPS[var_name])(t_mapped)


def _default_color(i: int, n: int) -> tuple:
    """Cycle through tab20 for multi-line plots with no single-out variable."""
    return _DEFAULT_CMAP(i % 20)


# ---------------------------------------------------------------------------
# Legend label
# ---------------------------------------------------------------------------

def _legend_label(row: pd.Series, color_by: str | None) -> str:
    """
    Compact label for each FvD curve in the legend.

    Format (when color_by='R'):
        [R=3.27] A=60.0  CC=12  VC=6  |  SEA=3.978  CFE=0.665

    The singled-out variable is bracketed and moved first for quick visual
    cross-referencing with the colorbar.
    """
    r   = float(row['R'])
    a   = float(row['A'])
    cc  = int(round(float(row['CC'])))
    vc  = int(round(float(row['VC'])))
    sea = float(row['SEA'])
    cfe = float(row['CFE'])

    param_order = ['R', 'A', 'CC', 'VC']
    if color_by and color_by in param_order:
        param_order = [color_by] + [p for p in param_order if p != color_by]

    parts = []
    for p in param_order:
        if p == 'R':
            s = f"R={r:.2f}"
        elif p == 'A':
            s = f"A={a:.1f}"
        elif p == 'CC':
            s = f"CC={cc}"
        else:
            s = f"VC={vc}"
        parts.append(f"[{s}]" if p == color_by else s)

    param_str  = "  ".join(parts)
    metric_str = f"SEA={sea:.3f}  CFE={cfe:.3f}"
    return f"{param_str}  |  {metric_str}"


# ---------------------------------------------------------------------------
# Point-mode helpers
# ---------------------------------------------------------------------------

def parse_point_str(s: str) -> dict[str, float]:
    """
    Parse a point specification string into a parameter dict.

    Accepted formats (comma- or space-separated, any order):
        "R=3.5,A=60.0,CC=12,VC=6"
        "R=3.5 A=60.0 CC=12 VC=6"
        "R=3.5, A=60.0, CC=12, VC=6, T=1"

    T is optional and defaults to 0.  R, A, CC, VC are required.

    Returns
    -------
    dict mapping uppercase parameter names to float values.
    """
    import re
    result: dict[str, float] = {}
    for part in re.split(r'[,\s]+', s.strip()):
        part = part.strip()
        if not part:
            continue
        if '=' not in part:
            raise ValueError(f"Expected KEY=VALUE, got '{part}'.")
        k, v = part.split('=', 1)
        result[k.strip().upper()] = float(v.strip())
    for req in ('R', 'A', 'CC', 'VC'):
        if req not in result:
            raise ValueError(
                f"Missing required parameter '{req}'. "
                "Format: R=<float>,A=<float>,CC=<int>,VC=<int>"
            )
    return result


def _find_pd_row(df: pd.DataFrame, params: dict[str, float]) -> pd.Series:
    """
    Return the _PD.csv row that matches params by exact key lookup.

    If no row is found (e.g. the point was removed as an outlier), returns a
    synthetic row with NaN for SEA and CFE so the legend still shows the inputs.
    """
    r  = round(params['R'],  2)
    a  = round(params['A'],  2)
    cc = int(round(params['CC']))
    vc = int(round(params['VC']))
    t  = int(params.get('T', 0))

    mask = (
        (df['R'].round(2) == r) &
        (df['A'].round(2) == a) &
        (df['CC'].round(0).astype(int) == cc) &
        (df['VC'].round(0).astype(int) == vc) &
        (df['T'] == t)
    )
    hits = df[mask]
    if not hits.empty:
        return hits.iloc[0]
    return pd.Series({
        'R': params['R'], 'A': params['A'],
        'CC': params['CC'], 'VC': params['VC'],
        'T': t, 'SEA': float('nan'), 'CFE': float('nan'),
    })


def _nearest_pd_row(df: pd.DataFrame, params: dict[str, float]) -> tuple[pd.Series, float]:
    """
    Return (nearest_row, normalised_distance) in df relative to params.

    Each axis is scaled by its domain range so all four parameters contribute
    equally.  Used to give a helpful hint when an exact point lookup fails.
    """
    features = ['R', 'A', 'CC', 'VC']
    scales   = {k: (PARAM_DOMAINS[k][1] - PARAM_DOMAINS[k][0]) for k in features}
    clean    = df.dropna(subset=features)
    dists    = clean.apply(
        lambda row: sum(
            ((float(row[k]) - params.get(k, 0.0)) / scales[k]) ** 2
            for k in features
        ) ** 0.5,
        axis=1,
    )
    idx = dists.idxmin()
    return clean.loc[idx], float(dists[idx])


# ---------------------------------------------------------------------------
# Shared rendering core
# ---------------------------------------------------------------------------

def _render_plot(
    matched: list[tuple[pd.Series, str]],
    color_by: str | None,
    save: bool,
    title: str,
    output_name: str,
) -> None:
    """
    Draw, annotate, and save a superimposed FvD plot.

    Called by both plot_fvd_filtered and plot_fvd_points so all matplotlib
    logic lives in exactly one place.

    Parameters
    ----------
    matched     : list of (PD row, FvD csv path) pairs
    color_by    : input variable to drive line colour gradient, or None
    save        : True -> saves/, False -> temp/
    title       : figure title (formatted by the caller)
    output_name : filename stem for the PNG (no directory, no extension)
    """
    if color_by and color_by not in PARAM_DOMAINS:
        print(f"  Warning: --color-by '{color_by}' not in {list(PARAM_DOMAINS)}. Ignoring.")
        color_by = None

    if color_by:
        matched.sort(key=lambda x: float(x[0][color_by]))

    plt.close('all')
    n    = len(matched)
    figw = max(13, min(18, 10 + n * 0.05))
    fig, ax = plt.subplots(figsize=(figw, 7))

    for i, (row, fpath) in enumerate(matched):
        try:
            df_fvd = pd.read_csv(fpath, header=0)
            disp   = np.abs(df_fvd.iloc[:, 1].values.astype(float))
            force  = np.abs(df_fvd.iloc[:, 2].values.astype(float))
        except Exception as exc:
            print(f"  [warn] Could not load {os.path.basename(fpath)}: {exc}")
            continue

        color = (
            _var_to_color(float(row[color_by]), color_by)
            if color_by
            else _default_color(i, n)
        )
        label = _legend_label(row, color_by=color_by)
        ax.plot(disp, force, color=color, linewidth=2.0, alpha=0.72, label=label)

    # Colorbar (only when color_by is active)
    if color_by:
        lo, hi     = PARAM_DOMAINS[color_by]
        cmap_obj   = cm.get_cmap(_VAR_CMAPS[color_by])
        cmap_trunc = mcolors.LinearSegmentedColormap.from_list(
            f"{color_by}_trunc",
            cmap_obj(np.linspace(_CMAP_LO, _CMAP_HI, 256)),
        )
        sm = plt.cm.ScalarMappable(cmap=cmap_trunc, norm=plt.Normalize(vmin=lo, vmax=hi))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.015, shrink=0.88)
        cbar.set_label(f"{color_by}  (domain: {lo} - {hi})", fontsize=11)

    # Legend anchored below the x-axis — never overlaps the plot area
    lfsize       = max(5, min(9, int(100 / max(n, 1))))
    ncols_legend = max(2, min(5, -(-n // 8)))   # ceil(n/8), capped at 5 cols
    legend_title = f"{n} curves"
    if color_by:
        legend_title += f"   |   color scale: {color_by}"

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.07),
        ncol=ncols_legend,
        fontsize=lfsize,
        framealpha=0.92,
        title=legend_title,
        title_fontsize=8,
    )

    ax.set_title(title, fontsize=10, pad=10)
    ax.set_xlabel("Displacement (mm)", fontsize=11)
    ax.set_ylabel("Force (N)", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle='--', alpha=0.25)

    legend_rows = max(1, -(-n // ncols_legend))
    bottom_frac = min(0.45, 0.06 + legend_rows * 0.030 + 0.025)
    fig.tight_layout()
    fig.subplots_adjust(bottom=bottom_frac)

    out_dir  = os.path.join(_OUT_BASE, 'saves' if save else 'temp')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{output_name}.png")
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    print(f"[fvd_explorer] Saved -> {out_path}")
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------

def plot_fvd_filtered(
    folder_name: str,
    sea_min: float | None = None,
    sea_max: float | None = None,
    cfe_min: float | None = None,
    cfe_max: float | None = None,
    filter_mode: str = 'both',
    color_by: str | None = None,
    save: bool = False,
    max_lines: int = 80,
) -> None:
    """Filter the _PD.csv by SEA/CFE ranges and plot the matching FvD curves."""
    pd_file = _pd_path(folder_name)
    if not os.path.isfile(pd_file):
        print(f"[fvd_explorer] Error: processed data not found: {pd_file}")
        print("  Run --list-folders to see available datasets.")
        sys.exit(1)
    df = pd.read_csv(pd_file).dropna(subset=['R', 'A', 'CC', 'VC', 'SEA', 'CFE'])
    print(f"[fvd_explorer] Loaded {len(df)} rows from {os.path.basename(pd_file)}")

    try:
        fvd_index = build_fvd_index(folder_name)
    except FileNotFoundError as e:
        print(f"[fvd_explorer] Error: {e}")
        sys.exit(1)
    print(f"[fvd_explorer] Indexed {len(fvd_index)} FvD files")

    filter_mode = _auto_filter_mode(sea_min, sea_max, cfe_min, cfe_max, filter_mode)
    has_sea = sea_min is not None or sea_max is not None
    has_cfe = cfe_min is not None or cfe_max is not None
    if not has_sea and not has_cfe:
        print("[fvd_explorer] Warning: No filter bounds given -- all rows will be considered.")

    df_filtered = apply_filter(df, sea_min, sea_max, cfe_min, cfe_max, filter_mode)
    n_filtered  = len(df_filtered)
    print(f"[fvd_explorer] Filter ({filter_mode}): {n_filtered} / {len(df)} rows match")
    if df_filtered.empty:
        print("  No designs match. Try widening the range.")
        return
    if n_filtered > max_lines:
        print(f"  Capping at {max_lines} curves (top by SEA). Use --max-lines to change.")
        df_filtered = df_filtered.nlargest(max_lines, 'SEA')

    matched: list[tuple[pd.Series, str]] = []
    unmatched = 0
    for _, row in df_filtered.iterrows():
        key   = _make_key(row['R'], row['A'], row['CC'], row['VC'], row['T'])
        fpath = fvd_index.get(key)
        if fpath is None:
            unmatched += 1
        else:
            matched.append((row, fpath))
    if unmatched:
        print(f"  Warning: {unmatched} rows could not be matched to a FvD file (skipped)")
    if not matched:
        print("  No FvD files matched. Check folder name and data consistency.")
        return
    print(f"[fvd_explorer] Plotting {len(matched)} FvD curves...")

    filter_parts: list[str] = []
    if filter_mode in ('sea', 'both') and has_sea:
        lo_s = f"{sea_min:.3f}" if sea_min is not None else "-inf"
        hi_s = f"{sea_max:.3f}" if sea_max is not None else "+inf"
        filter_parts.append(f"SEA [{lo_s}, {hi_s}]")
    if filter_mode in ('cfe', 'both') and has_cfe:
        lo_s = f"{cfe_min:.3f}" if cfe_min is not None else "-inf"
        hi_s = f"{cfe_max:.3f}" if cfe_max is not None else "+inf"
        filter_parts.append(f"CFE [{lo_s}, {hi_s}]")
    filter_str = "  |  ".join(filter_parts) if filter_parts else "no filter"
    color_str  = f"   (colored by {color_by})" if color_by else ""
    title = (
        f"FvD Explorer: {folder_name}\n"
        f"Filter: {filter_str}  --  {len(matched)} curves{color_str}"
    )
    slug = (
        f"FvD_{folder_name}_{filter_str.replace(' ', '').replace('|', '_')}"
        .replace('[', '').replace(']', '').replace(',', '-').replace('/', '-')
    )
    if color_by:
        slug += f"_color{color_by}"

    _render_plot(matched, color_by, save, title, slug)


def plot_fvd_points(
    folder_name: str,
    point_strs: list[str],
    color_by: str | None = None,
    save: bool = False,
) -> None:
    """
    Plot FvD curves for an explicit list of design points.

    Each string in point_strs is parsed as comma- or space-separated key=value
    pairs, e.g. "R=3.5,A=60.0,CC=12,VC=6".  Points are matched to FvD files
    by exact parameter lookup.  SEA and CFE shown in the legend come from the
    _PD.csv when available; points absent from the CSV (e.g. removed outliers)
    show N/A for those fields.

    If a point is not found in the FvD index, the nearest neighbour in the
    dataset is printed as a hint so you can correct the values.

    Parameters
    ----------
    folder_name : dataset folder (must have both a _PD.csv and a FvD directory)
    point_strs  : one string per design, e.g. ["R=3.5,A=60.0,CC=12,VC=6"]
    color_by    : optional variable to colour lines by (R, A, CC, or VC)
    save        : write to saves/ (True) or temp/ (False)
    """
    pd_file = _pd_path(folder_name)
    if not os.path.isfile(pd_file):
        print(f"[fvd_explorer] Error: processed data not found: {pd_file}")
        sys.exit(1)
    df = pd.read_csv(pd_file)
    print(f"[fvd_explorer] Loaded {len(df)} rows from {os.path.basename(pd_file)}")

    try:
        fvd_index = build_fvd_index(folder_name)
    except FileNotFoundError as e:
        print(f"[fvd_explorer] Error: {e}")
        sys.exit(1)
    print(f"[fvd_explorer] Indexed {len(fvd_index)} FvD files")

    matched: list[tuple[pd.Series, str]] = []
    for s in point_strs:
        try:
            params = parse_point_str(s)
        except ValueError as exc:
            print(f"  [warn] Skipping '{s}': {exc}")
            continue

        key   = _make_key(params['R'], params['A'], params['CC'], params['VC'],
                          int(params.get('T', 0)))
        fpath = fvd_index.get(key)

        if fpath is None:
            row_near, dist = _nearest_pd_row(df, params)
            hint = (f"R={row_near['R']:.2f}, A={row_near['A']:.2f}, "
                    f"CC={int(row_near['CC'])}, VC={int(row_near['VC'])}")
            print(f"  [warn] No FvD file found for: {s}")
            print(f"         Nearest point in dataset: {hint}  (dist={dist:.3f})")
            continue

        row   = _find_pd_row(df, params)
        sea_s = f"{row['SEA']:.3f}" if not np.isnan(float(row['SEA'])) else "N/A"
        cfe_s = f"{row['CFE']:.3f}" if not np.isnan(float(row['CFE'])) else "N/A"
        print(f"  Resolved: {s}  ->  SEA={sea_s}  CFE={cfe_s}")
        matched.append((row, fpath))

    if not matched:
        print("[fvd_explorer] No points could be resolved. Check parameter values.")
        return
    print(f"[fvd_explorer] Plotting {len(matched)} point(s)...")

    color_str = f"   (colored by {color_by})" if color_by else ""
    title = (
        f"FvD Explorer: {folder_name}\n"
        f"{len(matched)} selected point(s){color_str}"
    )
    slug = f"FvD_{folder_name}_points{len(matched)}"
    if color_by:
        slug += f"_color{color_by}"

    _render_plot(matched, color_by, save, title, slug)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def list_folders() -> None:
    """Print all available dataset folders with their CSV counts and _PD status."""
    print("\nAvailable folders in Mesh_Conversion_Sample/:\n")
    print(f"  {'Folder':<55}  {'CSVs':>5}  {'_PD.csv':>8}")
    print("  " + "-" * 75)
    try:
        entries = sorted(os.listdir(_FVD_BASE))
    except FileNotFoundError:
        print(f"  Mesh_Conversion_Sample not found: {_FVD_BASE}")
        return

    for entry in entries:
        full = os.path.join(_FVD_BASE, entry)
        if not os.path.isdir(full):
            continue
        n_csv   = sum(1 for f in os.listdir(full) if f.endswith('.csv'))
        pd_ok   = os.path.isfile(_pd_path(entry))
        pd_mark = "yes" if pd_ok else "no"
        print(f"  {entry:<55}  {n_csv:>5}  {pd_mark:>8}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Superimposed FvD plots filtered by SEA and/or CFE ranges.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        '--folder',
        help='Dataset folder name inside Mesh_Conversion_Sample/ (e.g. FDData_SobS_OD40L50G3_Shell_Fixed)',
    )
    parser.add_argument('--sea-min',  type=float, default=None, metavar='N',
        help='Minimum SEA (inclusive)')
    parser.add_argument('--sea-max',  type=float, default=None, metavar='N',
        help='Maximum SEA (inclusive)')
    parser.add_argument('--cfe-min',  type=float, default=None, metavar='N',
        help='Minimum CFE (inclusive)')
    parser.add_argument('--cfe-max',  type=float, default=None, metavar='N',
        help='Maximum CFE (inclusive)')
    parser.add_argument(
        '--filter-mode',
        choices=['sea', 'cfe', 'both'],
        default='both',
        help=(
            'When both SEA and CFE ranges are given: '
            '"sea" ignores CFE args, "cfe" ignores SEA args, '
            '"both" (default) requires both to be satisfied simultaneously.'
        ),
    )
    parser.add_argument(
        '--color-by',
        choices=list(PARAM_DOMAINS),
        default=None,
        metavar='VAR',
        help=(
            'Color each FvD line by this input variable using a gradient '
            'across its full domain.  Choices: R, A, CC, VC.'
        ),
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save PNG to data_folder/output/saves/ instead of temp/',
    )
    parser.add_argument(
        '--max-lines',
        type=int,
        default=80,
        metavar='N',
        help='Maximum curves to draw (default: 80). Excess trimmed to top-N by SEA.',
    )
    parser.add_argument(
        '--point',
        action='append',
        dest='points',
        metavar='R=X,A=Y,CC=N,VC=M',
        help=(
            'Plot a specific design point by its exact parameter values. '
            'Repeat the flag for multiple points. '
            'Format: R=<float>,A=<float>,CC=<int>,VC=<int> '
            '(comma- or space-separated, T optional). '
            'Example: --point "R=3.5,A=60.0,CC=12,VC=6". '
            'When --point is used, --sea-min/max and --cfe-min/max are ignored.'
        ),
    )
    parser.add_argument(
        '--list-folders',
        action='store_true',
        help='Print available dataset folders and exit.',
    )

    args = parser.parse_args()

    if args.list_folders:
        list_folders()
        return

    if not args.folder:
        parser.error("--folder is required (or use --list-folders to see options).")

    # --point mode bypasses the SEA/CFE filter entirely
    if args.points:
        plot_fvd_points(
            folder_name=args.folder,
            point_strs=args.points,
            color_by=args.color_by,
            save=args.save,
        )
        return

    plot_fvd_filtered(
        folder_name=args.folder,
        sea_min=args.sea_min,
        sea_max=args.sea_max,
        cfe_min=args.cfe_min,
        cfe_max=args.cfe_max,
        filter_mode=args.filter_mode,
        color_by=args.color_by,
        save=args.save,
        max_lines=args.max_lines,
    )


if __name__ == '__main__':
    main()
