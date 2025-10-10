import os
import json
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import pandas as pd

class DataInterface:

    data_directory: str = os.path.join(os.path.dirname(__file__), '..', 'data_folder')

    @staticmethod
    def process_data(data_path : str) -> list: # pandas DataFrame list
        """
        Reads a whitespace-delimited text file and returns a list of DataFrames.
        Each DataFrame corresponds to a separate dataset in the file.
        """
        data_frames = []
        with open(data_path, 'r') as f:
            df = pd.read_csv(f, delim_whitespace=True, header=None)
            data_frames.append(df)
        return data_frames

    def is_DF_list(data: list) -> bool:
        return isinstance(data, list) and all(isinstance(df, pd.DataFrame) for df in data)

    # Can display single or multiple graphs (list of DataFrames or list of list of DataFrames)
    @staticmethod
    def display_graph(data : list, json_path: str, save_file : bool = False) -> None:
        is_DF_list: bool = is_DF_list(data)
        if not is_DF_list:
            is_DF_2Dlist = isinstance(data, list) and all(is_DF_list(sublist) for sublist in data)
        if not (is_DF_list or is_DF_2Dlist):
            raise ValueError("Data must be a DataFrame or list of DataFrames.")
        
        config: any = {}
        with open(json_path, 'r') as f:
            config = json.load(f)

        if is_DF_list:
            data.plot(title=f"{config.get("name", "Unnamed Plot")}: Force vs Displacement", xlabel="Displacement (mm)", ylabel="Force (N)")
        elif is_DF_2Dlist:
            n: int = config["n"]
            for i in range(n):
                data[i].plot(title=f"{config.get("name", "Unnamed Plot")}: Force vs Displacement", xlabel="Displacement (mm)", ylabel="Force (N)", label= " | ".join([config["params"][i] for i in range(n)]))
    
        plt.savefig(f"{DataInterface.data_directory}output/{'saves' if save_file else 'temp'}/{data.Name}.png")
        plt.show()
        plt.clf()

        plt.figure()
plt.plot(auc_list, cfe_list, "o")
plt.xlabel("AUC (Nmm)")
plt.ylabel("CFE")
plt.ylim(0, 1) # Mutable
plt.title("Crushing Force Efficiency vs Area Under Curve (Nmm)")
plt.grid(True)