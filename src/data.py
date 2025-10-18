# Standard Libary
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
                    os.remove(file_path)       # remove file or symlink
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)   # remove directory recursively
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

        # Ensure both directories exist
        if not os.path.exists(temp_dir):
            print(f"Temp folder not found: {temp_dir}")
            return
        os.makedirs(save_dir, exist_ok=True)

        moved_files = 0
        for filename in os.listdir(temp_dir):
            src_path = os.path.join(temp_dir, filename)
            dst_path = os.path.join(save_dir, filename)
            try:
                # Move file (will overwrite if already exists)
                shutil.move(src_path, dst_path)
                moved_files += 1
            except Exception as e:
                print(f"Error moving {filename}: {e}")

        print(f"Moved {moved_files} file(s) from temp → saves.")


    @staticmethod
    def process_data(data_path: str | None = None, data_list: list | None = None, col_names: list[str] | None = None) -> pd.DataFrame:
        if (data_path is None) == (data_list is None):
            raise ValueError("Provide exactly one of data_path or data_list.")
        col_names = col_names or []

        if data_path is not None:
            return pd.read_csv(data_path, sep=r"\s+", header=None, names=col_names)
        else:
            return pd.DataFrame(data_list, columns=col_names or None)


    # Can display single or multiple graphs superimposed (list of DataFrames or list of list of DataFrames)
    @staticmethod

    # TODO:
    # Create TWO args passes one with optional feautres (ex: custom color)
    # Another with standard matplotlib features (ex: xlabel, ylabel, grid)
    # ^ Which will be used to easily unpack using **args
    def display_graph(mode: OI.Mode, data: pd.DataFrame | list, *, json_path: str, params: list, save_file: bool = False, args: dict = {}) -> None:
        """
        Displays a graph based on the provided data and configuration.
        """
        xlabel = args.pop("xlabel", None)
        ylabel = args.pop("ylabel", None)
        want_grid = bool(args.pop("grid", False))

        is_df = isinstance(data, pd.DataFrame)
        is_list = isinstance(data, list) and all(isinstance(df, pd.DataFrame) for df in data)
        if not (is_df or is_list):
            raise ValueError("Data must be a DataFrame or list of DataFrames.")

        with open(json_path, 'r') as f:
            config = json.load(f)
        title_base = config.get("name", "Unnamed")

        plt.close('all')
        fig, ax = plt.subplots()

        if is_df:
            X, Y = data.columns[0], data.columns[1]
            ax.plot(data[X].values, data[Y].values, **args)
            title = f"{title_base}: {Y} vs {X}"
            if mode == OI.Mode.AvC:
                ax.set_ylim(top=1)
                # Labeling eachpoint
                for x_val, y_val, param in zip(data[X].values, data[Y].values, params):
                    ax.annotate(f"{param}", (x_val, y_val), textcoords="offset points", xytext=(0, 10), ha='center')
        else:
            # one label for the whole param set
            label = " | ".join(f"{k}:{v}" for k, v in zip(config["params"], params))
            n = len(data)
            for i in range(n):
                df = data[i]
                X, Y = df.columns[0], df.columns[1]
                
                # color = np.random.rand(3,)
                color = (plt.cm.Reds((i + 1) / n) if mode == OI.Mode.FvD else plt.cm.Blues(i + 1 / n))

                label = " | ".join( config["params"][j]+": "+str(params[i]) for j in range(config["n"]) )
                ax.plot(df[X].values, df[Y].values, label=label, color=color, **args)
            ax.legend()
            title = f"{title_base}: {Y} vs {X}"

        ax.set_title(title)
        ax.set_xlabel(xlabel or X)
        ax.set_ylabel(ylabel or Y)
        ax.set_ylim(bottom=0)
    
        if want_grid:
            ax.grid(True)

        out_dir = os.path.join(DataInterface.data_directory, 'output', 'saves' if save_file else 'temp')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{title_base}_{str(mode).split('.')[1]}.png")
        fig.savefig(out_path, bbox_inches='tight')
        plt.show()
        plt.close(fig)

