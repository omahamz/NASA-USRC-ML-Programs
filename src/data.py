# Standard Library
import os
import shutil
import json

# 3rd Party Libraries
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# src imports
from operations import OperationsInterface as OI

class DataInterface:

    data_directory = os.path.join(os.path.dirname(__file__), '..', 'data_folder')

    @staticmethod
    def color_gradient(value, start_color=(1.0,0.5,0.5), end_color=(0.5,0.0,0.0), min_val=60, max_val=75, blue=False):
        if value == 190:
            return 'black'
        if blue or value > 100:
            start_color = (0.5,0.5,1.0)
            end_color = (0.0,0.0,0.5)
            min_val = 160
            max_val = 190
        t = (value - min_val) / (max_val - min_val)
        t = np.clip(t, 0.0, 1.0)
        return (1 - t) * np.array(start_color) + t * np.array(end_color)


    @staticmethod
    def wipe_temp() -> None:
        """
        Deletes all files and subdirectories inside the given folder.
        The folder itself is preserved.
        """
        folder_path = os.path.join(DataInterface.data_directory, 'output', 'temp')
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            return

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    
    @staticmethod
    def save_temp() -> None:
        """
        Moves all files from the 'temp' output directory to the 'saves' output directory.
        If 'saves' does not exist, it will be created.
        Existing files with the same name will be overwritten.
        """
        temp_dir = os.path.join(DataInterface.data_directory, 'output', 'temp')
        save_dir = os.path.join(DataInterface.data_directory, 'output', 'saves')

        if not os.path.exists(temp_dir):
            print(f"Temp folder not found: {temp_dir}")
            return
        os.makedirs(save_dir, exist_ok=True)

        moved_files = 0
        for filename in os.listdir(temp_dir):
            src_path = os.path.join(temp_dir, filename)
            dst_path = os.path.join(save_dir, filename)
            try:
                shutil.move(src_path, dst_path)
                moved_files += 1
            except Exception as e:
                print(f"Error moving {filename}: {e}")

        print(f"Moved {moved_files} file(s) from temp → saves.")


    @staticmethod
    def process_data(
        data_path: str | None = None,
        data_list: list | None = None,
        col_names: list[str] | None = None,
        cutoff: int = None,
        normalize: bool = False,
        x_col: str | None = None,
        y_col: str | None = None,
    ) -> pd.DataFrame:
        """
        Load and optionally pre-process tabular data.

        Parameters
        ----------
        data_path : str, optional
            Path to a whitespace- or comma-separated file.
        data_list : list, optional
            Raw data as a list of rows/columns; mutually exclusive with data_path.
        col_names : list[str], optional
            Column names to assign after loading.
        cutoff : int, optional
            Keep only rows where the first column's value <= cutoff.
        normalize : bool
            Min-max normalise every column when True.
        x_col : str, optional
            If provided together with y_col, the returned DataFrame contains
            only these two columns (in that order).
        y_col : str, optional
            See x_col.
        """
        if (data_path is None) == (data_list is None):
            raise ValueError("Provide exactly one of data_path or data_list.")
        col_names = col_names or []

        # Load data
        if data_path is not None:
            if data_path.endswith('.csv'):
                # CSV already has a header row — read it naturally, then rename
                df = pd.read_csv(data_path, sep=',', header=0, engine='python')
                if col_names:
                    df.columns = col_names[:len(df.columns)]
            else:
                # Legacy whitespace-separated files have no header
                df = pd.read_csv(data_path, sep=r'\s+', header=None, names=col_names or None, engine='python')
        else:
            df = pd.DataFrame(data_list, columns=col_names or None)

        # Apply cutoff if provided
        if cutoff is not None:
            df = df[df.iloc[:, 0] <= cutoff]

        if normalize:
            df = (df - df.min()) / (df.max() - df.min())

        # Slice to just the two columns of interest when requested
        if x_col is not None and y_col is not None:
            df = df[[x_col, y_col]].reset_index(drop=True)

        return df


    @staticmethod
    def display_graph(
        mode: OI.Mode,
        data: pd.DataFrame | list,
        *,
        json_path: str | None,
        params: list,
        labels: list[str] | None = None,
        save_file: bool = False,
        args: dict = {},
    ) -> None:
        """
        Displays a graph based on the provided data and configuration.

        Parameters
        ----------
        mode : OI.Mode
            Graph mode (FvD, AvC, …).
        data : DataFrame or list of DataFrames
            Single DataFrame for a solo plot; list for a superimposed multi-line plot.
        json_path : str or None
            Path to the JSON config file. Pass None when no config file is needed
            (e.g. mesh-comparison graphs that don't rely on param metadata).
        params : list
            Parameter values corresponding to each dataset (used for colouring /
            labelling when no explicit labels are supplied).
        labels : list[str], optional
            Explicit legend labels for each line in a multi-line plot.  When
            omitted the function falls back to the previous auto-label behaviour.
        save_file : bool
            Save to 'saves' directory when True, else 'temp'.
        args : dict
            Extra keyword arguments forwarded to ax.plot(), plus optional keys
            "xlabel", "ylabel", "title", and "grid" which are consumed here.
        """
        xlabel = args.pop("xlabel", None)
        ylabel = args.pop("ylabel", None)
        title = args.pop("title", None)
        want_grid = bool(args.pop("grid", False))

        is_df = isinstance(data, pd.DataFrame)
        is_list = isinstance(data, list) and all(isinstance(df, pd.DataFrame) for df in data)
        if not (is_df or is_list):
            raise ValueError("Data must be a DataFrame or list of DataFrames.")

        # Load JSON config only when a path is provided
        if json_path is not None:
            with open(json_path, 'r') as f:
                config = json.load(f)
            title_base = config.get("name", "Unnamed")
        else:
            config = {}
            title_base = ""

        plt.close('all')
        fig, ax = plt.subplots()

        if is_df:
            title = title or f"FvD: {'TA' if params >= 100 else 'A'}{params%100}"
            X, Y = data.columns[0], data.columns[1]
            ax.plot(data[X].values, data[Y].values, **args, color='black')
            title = title or f"FvD: {title_base}{params}"
            if mode == OI.Mode.AvC:
                ax.set_ylim(top=1)
                for x_val, y_val, param in zip(data[X].values, data[Y].values, params):
                    ax.annotate(f"T{param}", (x_val, y_val), textcoords="offset points", xytext=(10,0), ha='center', fontsize=8)
        else:
            # Mesh-size colour palette — cycles gracefully beyond 3 entries
            # default_colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
            n = len(data)
            default_colors = [
                (1.0 - 0.5 * (i / max(n - 1, 1)), 0.0, 0.0)
                for i in range(n)
            ]
            for i in range(n):
                df = data[i]
                X, Y = df.columns[0], df.columns[1]

                # Prefer explicit labels; fall back to legacy auto-label logic
                if labels is not None:
                    label = labels[i]
                    color = default_colors[i] #color = default_colors[i % len(default_colors)]
                else:
                    temp_dict = {1: ["T1", "red"], 2: ["T2", "green"], 3: ["T3", "blue"]}
                    if params[i] in temp_dict:
                        label, color = temp_dict[params[i]]
                    else:
                        color = DataInterface.color_gradient(params[i])
                        label = " | ".join(
                            config["params"][j] + str(params[i])
                            for j in range(config.get("n", 1))
                        ) if config else str(params[i])

                ax.plot(df[X].values, df[Y].values, label=label, color=color, **args)
            ax.legend()
            title = title or f"{title_base}: {Y} vs {X}"

        ax.set_title(title)
        ax.set_xlabel(xlabel or X)
        ax.set_ylabel(ylabel or Y)
        ax.set_ylim(bottom=0)

        if want_grid:
            ax.grid(True)

        out_dir = os.path.join(DataInterface.data_directory, 'output', 'saves' if save_file else 'temp')
        os.makedirs(out_dir, exist_ok=True)
        safe_title = title.replace(" ", "_").replace(":", "")
        out_path = os.path.join(out_dir, f"{safe_title}.png")
        fig.savefig(out_path, bbox_inches='tight')
        plt.show()
        plt.close(fig)