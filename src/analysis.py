# Standard Library
import os
import re

# 3rd Party Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# src imports
from data import DataInterface


# ---------------------------------------------------------------------------
# Geometry constants (used for volume and SEA calculation)
# ---------------------------------------------------------------------------

OD = 40   # Outer diameter (mm)
ID = 32   # Inner diameter (mm)
L  = 50   # Length (mm)


# ---------------------------------------------------------------------------
# Column classification
# ---------------------------------------------------------------------------

# Columns whose values are strictly discrete integers — these are visualised
# with a box / candlestick plot instead of a scatter plot.
DISCRETE_COLS: frozenset[str] = frozenset({"N", "CC", "VC", "T"})

# Regex for the Shell/Solid filename format:
#   N<n>(T)A(Solid|Shell)R<r>A<a>CC<cc>VC<vc>(!)_FD
# The trailing '!' before _FD marks N as a modified variant (stored as e.g. "0!").
_SOLID_SHELL_RE = re.compile(
    r'^N(\d+)(T)?A(?:Solid|Shell)R(\d+)A(\d+)CC(\d+)VC(\d+)(!?)_FD$'
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_param_string(s: str) -> dict[str, float | str]:
    """
    Parse a parameter string into a dict of parameter values.

    Shell/Solid format  (N<n>(T)A(Solid|Shell)R<r>A<a>CC<cc>VC<vc>(!)_FD)
    -----------------------------------------------------------------------
    * N is returned as a raw integer string (e.g. "6"); a trailing "!" before
      _FD appends "!" to the N string (e.g. "0!") to indicate a modified mesh
      variant.  "Solid"/"Shell" are consumed by the pattern and do not appear
      as columns.
    * T=1 when the prefix contains "TA"; T=0 otherwise.
    * R, A, CC, VC are divided by 100 as usual.

    Legacy AID format  (N600AID3504R201A8989CC2100VC400U_FD, …)
    -------------------------------------------------------------
    * First 8 characters are stripped (historical prefix), then:
    * A *key* is one or more alpha characters immediately followed by digits.
    * Trailing tokens not followed by digits (e.g. "U_FD") are ignored.
    * Each numeric value is divided by 100 before being stored.
    * If a key starts with 'T' and matches a known param (TAID→AID, TA→A),
      the 'T' prefix is stripped and T=1 is recorded; otherwise T=0.
    """
    m = _SOLID_SHELL_RE.match(s)
    if m:
        n_digits, t_flag, r_val, a_val, cc_val, vc_val, bang = m.groups()
        return {
            "N":  n_digits + ("!" if bang else ""),
            "R":  int(r_val) / 100,
            "A":  int(a_val) / 100,
            "CC": int(cc_val) / 100,
            "VC": int(vc_val) / 100,
            "T":  1 if t_flag else 0,
        }

    # ── legacy AID-format fallback ─────────────────────────────────────────
    s = s[8:]
    T_PARAMS = {"TAID": "AID", "TA": "A"}

    result: dict[str, float] = {}
    t_flag_val: int | None = None
    i = 0
    n = len(s)

    while i < n:
        key_start = i
        while i < n and (s[i].isalpha() or s[i] == '_'):
            i += 1
        key = s[key_start:i]

        val_start = i
        while i < n and s[i].isdigit():
            i += 1
        val_str = s[val_start:i]

        if key and val_str:
            if key in T_PARAMS:
                result[T_PARAMS[key]] = int(val_str) / 100
                t_flag_val = 1
            else:
                base = T_PARAMS.get(key)
                if base is None:
                    if key in T_PARAMS.values():
                        if t_flag_val is None:
                            t_flag_val = 0
                    result[key] = int(val_str) / 100
                else:
                    result[base] = int(val_str) / 100
                    t_flag_val = 1
        elif not key and not val_str:
            i += 1  # skip unrecognised character (e.g. '!') to avoid infinite loop

    if t_flag_val is not None:
        result["T"] = t_flag_val

    return result


# ---------------------------------------------------------------------------
# AUC & CFE from CSV
# ---------------------------------------------------------------------------

def _auc_and_cfe(csv_path: str) -> tuple[float, float]:
    """
    Return (AUC, CFE) for a single FvD CSV file.

    CSV format (3 columns, header row):
        Time, Displacement_U2_mm, ReactionForce_RF2_N

    Column index 0 (Time) is ignored.
    Column index 1 = Displacement, column index 2 = Force.

    AUC  = trapezoidal integral of |Force| over |Displacement|.
    CFE  = mean(|Force|) / max(|Force|)
    """
    df = pd.read_csv(csv_path, header=0)

    disp  = np.abs(df.iloc[:, 1].values.astype(float))
    force = np.abs(df.iloc[:, 2].values.astype(float))

    auc  = round(float(np.trapz(force, disp)), 4)
    peak = float(force.max())
    mean = float(force.mean())
    cfe  = round(mean / peak, 4) if peak != 0 else 0.0

    return auc, cfe


# ---------------------------------------------------------------------------
# Volume calculation
# ---------------------------------------------------------------------------

def compute_volume(R: float, CC: float, VC: float) -> float:
    """
    Compute the volume of a single design point using the formula:

        V = (1 - ((3√3/2 * R² * VC * CC) / (π * OD * L)))
              * (π/4 * (OD² - ID²) * L)

    Parameters
    ----------
    R   : corner radius      (parsed from filename, already divided by 100)
    CC  : cell count         (parsed from filename, already divided by 100)
    VC  : volume coefficient (parsed from filename, already divided by 100)

    Constants OD, ID, L are taken from the module-level globals.

    Returns
    -------
    Volume as a float (units: mm³).
    """
    base_volume     = (np.pi / 4.0) * (OD**2 - ID**2) * L
    cutout_fraction = (
        (3.0 * np.sqrt(3.0) / 2.0) * (R**2) * VC * CC
    ) / (np.pi * OD * L)
    return round((1.0 - cutout_fraction) * base_volume, 4)


# ---------------------------------------------------------------------------
# Processed-data path
# ---------------------------------------------------------------------------

# Columns written to (and read from) the _PD CSV file, in order.
PD_COLUMNS: list[str] = ["N", "R", "A", "CC", "VC", "T", "AUC", "CFE", "Volume", "SEA", "Obj"]

# Columns used as the unique key when deduplicating during a merge.
DUPE_KEY_COLS: list[str] = ["R", "A", "CC", "VC"]


def pd_path(folder_name: str) -> str:
    """
    Return the canonical path for the processed-data CSV:

        <data_directory>/1_param/<folder_name>_PD.csv

    This sits alongside Mesh_Conversion_Sample/, not inside it.
    Pass the result of _effective_folder_name() to include the mesh type.
    """
    return os.path.join(
        DataInterface.data_directory,
        "1_param",
        f"{folder_name}_PD.csv",
    )


def _detect_mesh_type(folder_name: str) -> str | None:
    """Return 'Solid', 'Shell', or None by inspecting filenames in the folder."""
    base = os.path.join(
        DataInterface.data_directory, "1_param", "Mesh_Conversion_Sample", folder_name
    )
    try:
        for f in os.listdir(base):
            if "Solid" in f:
                return "Solid"
            if "Shell" in f:
                return "Shell"
    except OSError:
        pass
    return None


def _effective_folder_name(folder_name: str) -> str:
    """
    Return folder_name with the detected mesh type appended when neither
    'Solid' nor 'Shell' already appears in folder_name.

    Example: "0v6v8v0!"  →  "0v6v8v0!_Solid"
             "FDData_SobS_OD40L50G3_Solid"  →  unchanged
    """
    if "Solid" in folder_name or "Shell" in folder_name:
        return folder_name
    mesh = _detect_mesh_type(folder_name)
    return f"{folder_name}_{mesh}" if mesh else folder_name


# ---------------------------------------------------------------------------
# Build segment — FD CSVs → processed DataFrame
# ---------------------------------------------------------------------------

def _build_raw(folder_name: str) -> pd.DataFrame:
    """
    Scan Mesh_Conversion_Sample/<folder_name>/ for every FD CSV, parse
    filenames for parameters, and compute AUC & CFE.  Returns a raw
    DataFrame (no Volume / SEA / Obj yet).
    """
    base = os.path.join(
        DataInterface.data_directory,
        "1_param",
        "Mesh_Conversion_Sample",
        folder_name,
    )

    if not os.path.isdir(base):
        raise FileNotFoundError(f"FD data directory not found: {base}")

    csv_files = sorted(f for f in os.listdir(base) if f.lower().endswith(".csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {base}")

    rows = []
    for fname in csv_files:
        stem   = os.path.splitext(fname)[0]
        params = parse_param_string(stem)

        csv_path = os.path.join(base, fname)
        try:
            auc, cfe = _auc_and_cfe(csv_path)
        except Exception as exc:
            print(f"  [WARN] Could not process {fname}: {exc}")
            auc, cfe = float("nan"), float("nan")

        row = {**params, "AUC": auc, "CFE": cfe}
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  Parsed {len(df)} FD files from '{folder_name}'.")
    return df


def build_processed_data(folder_name: str, k: float | None = None) -> pd.DataFrame:
    """
    Full build pipeline:  FD CSVs → Volume → SEA → Obj → _PD CSV.

    Steps
    -----
    1. Parse every FD CSV in Mesh_Conversion_Sample/<folder_name>/
    2. Compute Volume and SEA via compute_volume()
    3. Compute Obj = SEA - k * (1 - CFE),  k defaults to mean(SEA)
    4. Keep only PD_COLUMNS and write to <folder_name>_PD.csv next to
       Mesh_Conversion_Sample/

    Returns the processed DataFrame.
    """
    print(f"\n[BUILD] Processing FD data for '{folder_name}' …")

    df = _build_raw(folder_name)
    df = add_sea(df)
    df = add_objective(df, k=k)

    present = [c for c in PD_COLUMNS if c in df.columns]
    missing = [c for c in PD_COLUMNS if c not in df.columns]
    if missing:
        print(f"  [WARN] PD columns not produced and will be absent: {missing}")
    df = df[present]

    out = pd_path(_effective_folder_name(folder_name))
    df.to_csv(out, index=False)
    print(f"[BUILD] Saved processed data → {out}  ({len(df)} rows, {len(present)} cols)")
    return df


# ---------------------------------------------------------------------------
# Load or build
# ---------------------------------------------------------------------------

def load_or_build(folder_name: str, *, rebuild: bool = False) -> pd.DataFrame:
    """
    Return the processed DataFrame for folder_name.

    Logic
    -----
    * If rebuild=True, always regenerate from FD CSVs and overwrite the
      _PD file.
    * Otherwise, load the existing _PD CSV if it exists.
    * If no _PD file exists, build it automatically.

    Parameters
    ----------
    folder_name : str
        Name of the subdirectory under Mesh_Conversion_Sample/.
    rebuild : bool
        Pass True (or run the script with --rebuild) to force a full
        rebuild from FD data even when a _PD file already exists.
    """
    path = pd_path(_effective_folder_name(folder_name))

    if rebuild:
        print(f"[LOAD] --rebuild flag set; regenerating _PD file …")
        return build_processed_data(folder_name)

    if os.path.isfile(path):
        df = pd.read_csv(path)
        print(f"[LOAD] Loaded existing processed data from '{os.path.basename(path)}'  "
              f"({len(df)} rows)")
        return df

    print(f"[LOAD] No _PD file found for '{folder_name}'; building now …")
    return build_processed_data(folder_name)


# ---------------------------------------------------------------------------
# Merge a new FD folder into an existing _PD file
# ---------------------------------------------------------------------------

def merge_into_pd(
    source_folder: str,
    target_folder: str,
    *,
    overwrite_dupes: bool = False,
) -> pd.DataFrame:
    """
    Process FvD data from source_folder and append new rows into the
    _PD file for target_folder.

    Duplicate detection uses DUPE_KEY_COLS (R, A, CC, VC).  By default,
    rows already present in the target are skipped; pass
    overwrite_dupes=True to replace them instead.

    After merging, Obj is recomputed for the full combined dataset using
    k = mean(SEA) over all rows.

    Returns the updated combined DataFrame.
    """
    print(f"\n[MERGE] '{source_folder}'  →  '{target_folder}_PD.csv'")

    # 1. Build or load the source processed data
    source_df = load_or_build(source_folder)

    # 2. Load (or start empty) the target _PD
    target_path = pd_path(_effective_folder_name(target_folder))
    if os.path.isfile(target_path):
        target_df = pd.read_csv(target_path)
        print(f"[MERGE] Loaded target '{os.path.basename(target_path)}'  ({len(target_df)} rows)")
    else:
        print(f"[MERGE] No existing _PD for '{target_folder}' — starting from scratch.")
        target_df = pd.DataFrame(columns=PD_COLUMNS)

    # 3. Identify new-only rows (deduplicate on DUPE_KEY_COLS)
    key_cols = [c for c in DUPE_KEY_COLS
                if c in source_df.columns and c in target_df.columns]

    if target_df.empty or not key_cols:
        new_rows = source_df
        n_dupes  = 0
    else:
        # Left-join to detect which source rows already exist in target.
        # round(6) guards against floating-point noise in stored values.
        left      = source_df[key_cols].round(6).reset_index(drop=True)
        right     = target_df[key_cols].round(6).drop_duplicates()
        indicator = left.merge(right, on=key_cols, how="left", indicator=True)["_merge"]
        is_new    = (indicator == "left_only").values   # numpy bool array, positional

        n_dupes = int((~is_new).sum())
        if n_dupes:
            verb = "Replacing" if overwrite_dupes else "Skipping"
            print(f"[MERGE] {verb} {n_dupes} duplicate(s) matched on {key_cols}.")

        if overwrite_dupes and n_dupes:
            dupe_keys = source_df.loc[~is_new, key_cols].round(6)
            keep_mask = (
                target_df[key_cols].round(6)
                .merge(dupe_keys, on=key_cols, how="left", indicator=True)["_merge"]
                != "both"
            ).values
            target_df = target_df[keep_mask]
            new_rows  = source_df
        else:
            new_rows = source_df[is_new]

    n_new = len(new_rows)
    print(f"[MERGE] Appending {n_new} new row(s).")

    if n_new == 0:
        print("[MERGE] Nothing to add — target file unchanged.")
        return target_df

    # 4. Combine and recompute Obj with k = mean(SEA) over the full dataset
    combined = pd.concat([target_df, new_rows], ignore_index=True)
    if "SEA" in combined.columns and "CFE" in combined.columns:
        combined = add_objective(
            combined.drop(columns=["Obj"], errors="ignore"),
            k=float(combined["SEA"].mean()),
        )

    # 5. Restrict to canonical columns and save
    present  = [c for c in PD_COLUMNS if c in combined.columns]
    combined = combined[present]
    combined.to_csv(target_path, index=False)
    print(f"[MERGE] Saved → {target_path}  ({len(combined)} rows total)")
    return combined


# ---------------------------------------------------------------------------
# Objective column
# ---------------------------------------------------------------------------

def add_objective(df: pd.DataFrame, k: float | None = None) -> pd.DataFrame:
    """
    Append an 'Obj' column:  Obj = SEA - k * (1 - CFE)

    Requires 'SEA' and 'CFE' to already be present in df (call add_sea first).
    k defaults to the mean SEA of the dataset when not provided.
    """
    if "SEA" not in df.columns:
        raise ValueError("'SEA' column not found — call add_sea(df) before add_objective(df).")
    if k is None:
        k = float(df["SEA"].mean())
    df = df.copy()
    df["Obj"] = round(df["SEA"] - k * (1.0 - df["CFE"]), 4)
    print(f"  Objective computed with k = {k:.4f}")
    return df


# ---------------------------------------------------------------------------
# SEA column
# ---------------------------------------------------------------------------

def add_sea(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append 'Volume' and 'SEA' columns to the DataFrame.

        Volume  = compute_volume(R, CC, VC)   [mm³]
        SEA     = AUC / Volume                [N/mm²  ≡  MPa  when AUC is in N·mm]

    Rows missing R, CC, or VC receive NaN for both columns.
    Module-level constants OD, ID, L are printed so the user can verify them.
    """
    print(f"Computing SEA  (OD={OD} mm, ID={ID} mm, L={L} mm) …")

    df = df.copy()

    required = {"R", "CC", "VC"}
    missing  = required - set(df.columns)
    if missing:
        print(f"  [WARN] Column(s) {missing} not found — SEA will be NaN for all rows.")
        df["Volume"] = float("nan")
        df["SEA"]    = float("nan")
        return df

    volumes = np.where(
        df[["R", "CC", "VC"]].notna().all(axis=1),
        [
            compute_volume(r, cc, vc)
            for r, cc, vc in zip(df["R"], df["CC"], df["VC"])
        ],
        float("nan"),
    )

    df["Volume"] = volumes
    df["SEA"]    = round(df["AUC"] / df["Volume"], 4)

    n_nan = df["SEA"].isna().sum()
    if n_nan:
        print(f"  [WARN] {n_nan} row(s) produced NaN SEA (check R / CC / VC values).")

    print(f"SEA range: {df['SEA'].min():.4g}  –  {df['SEA'].max():.4g}")
    return df


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _save_fig(
    fig: plt.Figure,
    title: str,
    df: pd.DataFrame,
    save_file: bool,
    save_csv: bool,
) -> None:
    """Save figure (and optionally a CSV) then display it."""
    out_dir = os.path.join(
        DataInterface.data_directory,
        "output",
        "saves" if save_file else "temp",
    )
    os.makedirs(out_dir, exist_ok=True)
    safe     = title.replace(" ", "_").replace(":", "").replace("[", "").replace("]", "")
    out_path = os.path.join(out_dir, f"{safe}.png")

    if save_csv:
        csv_out = os.path.join(out_dir, f"{safe}.csv")
        export_cols = [c for c in df.columns if not c.startswith("_")]
        df[export_cols].to_csv(csv_out, index=False)
        print(f"CSV  saved → {csv_out}")

    fig.savefig(out_path, bbox_inches="tight")
    print(f"Plot saved → {out_path}")
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Box / candlestick plot  (used when x or y is a discrete column)
# ---------------------------------------------------------------------------

def box_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    folder_name: str,
    *,
    save_file: bool = False,
    save_csv: bool = False,
) -> None:
    """
    Draw a candlestick-style box plot when one axis is a discrete column
    (CC or VC).

    Layout rules
    ------------
    * The discrete column is always placed on the x-axis so each unique
      integer value gets its own candle.  If the *user* put the discrete
      column on y_col, the axes are silently swapped and a note is printed.
    * Each candle shows:
        - whiskers : full min → max range
        - box      : IQR  (Q1 → Q3)
        - median line
        - mean marker (triangle)
    * Individual data points are overlaid as a jittered strip for transparency.
    * color_col is not supported for box plots (grouping already encodes the
      discrete variable).
    """
    if x_col in DISCRETE_COLS and y_col in DISCRETE_COLS:
        raise ValueError(
            "Both axes are discrete columns — choose at least one continuous column."
        )
    if y_col in DISCRETE_COLS:
        print(f"  [INFO] Swapping axes: '{y_col}' is discrete and will be on the x-axis.")
        x_col, y_col = y_col, x_col

    groups      = sorted(df[x_col].dropna().unique())
    group_data  = [df.loc[df[x_col] == g, y_col].dropna().values for g in groups]
    positions   = list(range(len(groups)))

    plt.close("all")
    fig, ax = plt.subplots(figsize=(max(7, len(groups) * 1.2), 6))

    box_width = 0.35

    for pos, grp_vals in zip(positions, group_data):
        if len(grp_vals) == 0:
            continue

        q1, median, q3 = np.percentile(grp_vals, [25, 50, 75])
        mean_val        = grp_vals.mean()
        vmin, vmax      = grp_vals.min(), grp_vals.max()

        # Whisker (full range)
        ax.plot([pos, pos], [vmin, vmax],
                color="steelblue", linewidth=1.4, zorder=2)
        # Whisker caps
        for cap_y in (vmin, vmax):
            ax.plot([pos - box_width * 0.4, pos + box_width * 0.4],
                    [cap_y, cap_y],
                    color="steelblue", linewidth=1.4, zorder=2)

        # IQR box
        box_rect = plt.Rectangle(
            (pos - box_width / 2, q1),
            box_width, q3 - q1,
            facecolor="lightsteelblue", edgecolor="steelblue",
            linewidth=1.5, zorder=3,
        )
        ax.add_patch(box_rect)

        # Median line
        ax.plot([pos - box_width / 2, pos + box_width / 2],
                [median, median],
                color="navy", linewidth=2.0, zorder=4)

        # Mean marker
        ax.scatter([pos], [mean_val],
                   marker="^", color="darkred", s=60, zorder=5,
                   label="mean" if pos == 0 else "")

        # Jittered individual points
        jitter = np.random.uniform(-box_width * 0.3, box_width * 0.3,
                                   size=len(grp_vals))
        ax.scatter(pos + jitter, grp_vals,
                   color="steelblue", alpha=0.35, s=18, zorder=1)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{g:g}" if isinstance(g, (int, float)) else str(g) for g in groups])

    title = f"{folder_name}: {y_col} vs {x_col}  [box]"
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    from matplotlib.lines   import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Line2D([0], [0], color="navy",      linewidth=2,   label="Median"),
        Line2D([0], [0], marker="^",        color="darkred", linestyle="None",
               markersize=7,                label="Mean"),
        Patch(facecolor="lightsteelblue",   edgecolor="steelblue",
              label="IQR (Q1–Q3)"),
        Line2D([0], [0], color="steelblue", linewidth=1.4, label="Min / Max"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="best")

    _save_fig(fig, title, df, save_file, save_csv)


# ---------------------------------------------------------------------------
# Heatmap plot  (SEA vs CFE space, binned and coloured by mean A or R)
# ---------------------------------------------------------------------------

def heatmap_plot(
    df: pd.DataFrame,
    folder_name: str,
    color_col: str,
    *,
    n_bins: int = 10,
    save_file: bool = False,
    save_csv: bool = False,
) -> None:
    """
    Bin the (SEA, CFE) space into an n_bins × n_bins grid and colour each
    occupied cell by the mean value of color_col (typically 'A' or 'R').

    Parameters
    ----------
    df          : processed DataFrame, must contain SEA, CFE, and color_col.
    folder_name : used in the title and saved filename.
    color_col   : column whose mean drives the cell colour ('A' or 'R').
    n_bins      : number of bins along each axis (default 10).
    save_file   : if True, save to 'saves/' instead of 'temp/'.
    save_csv    : if True, also write the binned summary as a CSV.
    """
    required = {"SEA", "CFE", color_col}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Column(s) {missing} not found — cannot draw heatmap.")

    sub = df[["SEA", "CFE", color_col]].dropna()

    sea_vals   = sub["SEA"].values
    cfe_vals   = sub["CFE"].values
    color_vals = sub[color_col].values

    sea_edges = np.linspace(sea_vals.min(), sea_vals.max(), n_bins + 1)
    cfe_edges = np.linspace(cfe_vals.min(), cfe_vals.max(), n_bins + 1)

    # Build a 2-D grid of mean color_col values; NaN for empty bins
    grid = np.full((n_bins, n_bins), np.nan)
    for i in range(n_bins):
        for j in range(n_bins):
            mask = (
                (sea_vals >= sea_edges[i]) & (sea_vals < sea_edges[i + 1]) &
                (cfe_vals >= cfe_edges[j]) & (cfe_vals < cfe_edges[j + 1])
            )
            if mask.sum() > 0:
                grid[j, i] = color_vals[mask].mean()   # j=CFE row, i=SEA col

    plt.close("all")
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        extent=[sea_edges[0], sea_edges[-1], cfe_edges[0], cfe_edges[-1]],
        cmap="Reds",
        interpolation="nearest",
    )

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(f"mean {color_col}", fontsize=11)

    title = f"{folder_name}: CFE vs SEA  [heatmap, colour = mean {color_col}]"
    ax.set_title(title)
    ax.set_xlabel("SEA")
    ax.set_ylabel("CFE")
    ax.grid(False)

    _save_fig(fig, title, df, save_file, save_csv)


# ---------------------------------------------------------------------------
# Scatter plot  (default mode; delegates to box_plot or heatmap_plot as needed)
# ---------------------------------------------------------------------------

def scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    folder_name: str,
    *,
    color_col: str | None = None,
    mode: str = "scatter",
    save_file: bool = False,
    save_csv: bool = False,
) -> None:
    """
    Draw a plot of x_col vs y_col from df.

    mode
    ----
    "scatter"  (default)
        Standard scatter plot.  Automatically delegates to box_plot() when
        either axis is a discrete column (CC or VC).

    "heatmap"
        Bins the (SEA, CFE) space and colours each cell by the mean value of
        color_col (must be 'A' or 'R').  x_col and y_col are ignored in this
        mode — the axes are always SEA (x) and CFE (y).

    color_col (optional for scatter, required for heatmap)
        Scatter: maps point colours to this column's values using a red
        gradient; also shows a colorbar.
        Heatmap: the column whose mean determines each bin's colour.
    """
    # --- heatmap mode ---------------------------------------------------
    if mode == "heatmap":
        if color_col is None:
            raise ValueError(
                "heatmap mode requires color_col (e.g. 'A' or 'R')."
            )
        heatmap_plot(df, folder_name, color_col,
                     save_file=save_file, save_csv=save_csv)
        return

    # --- validate columns for scatter -----------------------------------
    check_cols = [x_col, y_col] + ([color_col] if color_col else [])
    missing = [c for c in check_cols if c not in df.columns]
    if missing:
        available = [c for c in df.columns if not c.startswith("_")]
        raise ValueError(
            f"Column(s) {missing} not found in dataset.\n"
            f"Available columns: {available}"
        )

    # --- auto-detect discrete axis and delegate -------------------------
    if x_col in DISCRETE_COLS or y_col in DISCRETE_COLS:
        if color_col:
            print("  [INFO] color_col is not supported for box plots — ignored.")
        box_plot(df, x_col, y_col, folder_name,
                 save_file=save_file, save_csv=save_csv)
        return

    # --- continuous scatter ---------------------------------------------
    x = df[x_col].values
    y = df[y_col].values

    plt.close("all")
    fig, ax = plt.subplots(figsize=(8, 6))

    if color_col is not None:
        # ---- color-mapped mode ----------------------------------------
        c_vals = df[color_col].values.astype(float)

        c_min, c_max = c_vals.min(), c_vals.max()
        norm = plt.Normalize(vmin=c_min, vmax=c_max)
        cmap = plt.cm.Reds

        sc = ax.scatter(
            x, y,
            c=c_vals,
            cmap=cmap,
            norm=norm,
            edgecolors="darkred",
            linewidths=0.5,
            alpha=0.45,           # semi-transparent so overlapping points show through
            s=80,
            zorder=3,
        )

        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(color_col, fontsize=11)

        unique_vals = np.unique(c_vals)
        if len(unique_vals) <= 15:
            cbar.set_ticks(unique_vals)
            cbar.set_ticklabels([f"{v:g}" for v in unique_vals])

    else:
        # ---- plain mode -----------------------------------------------
        ax.scatter(x, y, color="steelblue", edgecolors="navy",
                   alpha=0.40, s=70, zorder=3)

    title = f"{folder_name}: {y_col} vs {x_col}"
    if color_col:
        title += f"  [colour = {color_col}]"

    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, linestyle="--", alpha=0.4)

    _save_fig(fig, title, df, save_file, save_csv)


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------

def interactive(*, rebuild: bool = False) -> None:
    """
    Prompt the user to pick a subdirectory, then x/y columns, an optional
    visualization mode, and display the appropriate plot.

    Parameters
    ----------
    rebuild : bool
        When True the _PD file is always regenerated from FD data, even if
        one already exists.  Pass --rebuild on the command line to set this.
    """
    base_root = os.path.join(
        DataInterface.data_directory, "1_param", "Mesh_Conversion_Sample"
    )

    available = sorted(
        d for d in os.listdir(base_root)
        if os.path.isdir(os.path.join(base_root, d))
        and any(f.lower().endswith(".csv") for f in os.listdir(os.path.join(base_root, d)))
    )
    if not available:
        print(f"No subdirectories with CSV files found under {base_root}")
        return

    print("\nAvailable folders:")
    for i, a in enumerate(available, 1):
        tag = "  [PD exists]" if os.path.isfile(pd_path(_effective_folder_name(a))) else ""
        print(f"  [{i}] {a}{tag}")

    raw = input("\nEnter folder name or number from the list above: ").strip()

    folder_name: str
    if raw.isdigit():
        idx = int(raw) - 1
        if not (0 <= idx < len(available)):
            print(f"Number out of range — pick 1–{len(available)}.")
            return
        folder_name = available[idx]
    elif raw in available:
        folder_name = raw
    else:
        print(f"'{raw}' not found in the list above.")
        return

    print()
    try:
        df = load_or_build(folder_name, rebuild=rebuild)
    except FileNotFoundError as e:
        print(e)
        return

    param_cols = [c for c in df.columns if not c.startswith("_")]
    while True:
        print(f"\nAvailable columns: {param_cols}")

        mode_raw = input(
            "Visualization mode — scatter (default) / heatmap: "
        ).strip().lower()
        mode = mode_raw if mode_raw in ("scatter", "heatmap") else "scatter"

        if mode == "heatmap":
            color_raw = input(
                "Color column for heatmap (e.g. A or R): "
            ).strip()
            color_col = color_raw if color_raw else None
            x_col = y_col = "SEA"      # unused by heatmap; placeholders
        else:
            x_col = input("X-axis column: ").strip()
            y_col = input("Y-axis column: ").strip()
            color_raw = input(
                "Color column (leave blank for plain blue): "
            ).strip()
            color_col = color_raw if color_raw else None

        save     = input("Save to 'saves' directory? [y/N]: ").strip().lower() == "y"
        save_csv = input("Save data as CSV?              [y/N]: ").strip().lower() == "y"

        try:
            scatter_plot(df, x_col, y_col, folder_name,
                         color_col=color_col, mode=mode,
                         save_file=save, save_csv=save_csv)
        except ValueError as e:
            print(e)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    args = sys.argv[1:]

    if "--merge-into" in args:
        # Usage: python analysis.py --merge-into <target_folder> <source_folder> [--overwrite-dupes]
        idx = args.index("--merge-into")
        if idx + 2 >= len(args):
            print(
                "Usage: python analysis.py --merge-into <target_folder> <source_folder>"
                " [--overwrite-dupes]"
            )
            sys.exit(1)
        _target   = args[idx + 1]
        _source   = args[idx + 2]
        _overwrite = "--overwrite-dupes" in args
        merge_into_pd(_source, _target, overwrite_dupes=_overwrite)
    else:
        interactive(rebuild="--rebuild" in args)