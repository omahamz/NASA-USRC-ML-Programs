# Standard Libary
import os
import glob
import re

# src imports
from data import DataInterface as DI
from operations import OperationsInterface as OI

data_directory = os.path.join(os.path.dirname(__file__), '..', 'data_folder')

def main():

  material_name = input("What Material <KOSE, KOSH, HEX>?: ")
  sub_directory = os.path.join(data_directory, "1_param", material_name)

  DI.wipe_temp()

  data_frames = { m : [] for m in OI.Mode } # Dictionary to hold data frames for each mode
  param_values = []

  ### Process each file in the sub_directory ###
  for file_path in os.listdir(sub_directory):
    file_path = os.path.join(sub_directory, file_path)

    if file_path.endswith('.txt'):
      param_values.append(int(re.search(rf"{material_name}_(\d+)", file_path).group(1))) # Extract param value from filename
      ### F vs. D ###
      processed_data = DI.process_data(data_path=file_path, col_names=["Displacement", "Force"] )
      data_frames[OI.Mode.FvD].append(processed_data)
      ### AUC and CFE ###
      auc = OI.auc_trapz(processed_data, x_col="Displacement", y_col="Force", x_max=6.0)
      cfe = OI.cfe(processed_data["Force"].tolist())
      data_frames[OI.Mode.AvC].append([auc, cfe])

  ### F vs. D ###
  DI.display_graph(OI.Mode.FvD,
    data_frames[OI.Mode.FvD], 
    json_path=f"{sub_directory}/{material_name}_params.json", 
    params=param_values, 
    save_file=True, 
    args = {"xlabel": "Displacement (mm)", "ylabel": "Force (N)", "grid": True})
  ### AUC and CFE ###
  processed_data = DI.process_data(data_list=data_frames[OI.Mode.AvC], col_names=["AUC", "CFE"])
  data_frames[OI.Mode.AvC] = processed_data
  DI.display_graph(OI.Mode.AvC,
    data_frames[OI.Mode.AvC], 
    json_path=f"{sub_directory}/{material_name}_params.json", 
    params=param_values, 
    save_file=True, 
    args = {"xlabel": "Area Under Curve (Nmm)", "ylabel": "Crushing Force Efficiency", "grid": True, "marker": "o", "linestyle": "--"})

if __name__ == "__main__":
  main()

