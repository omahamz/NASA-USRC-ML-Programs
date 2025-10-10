import os
import glob
import re

# src imports
import data
import operations

data_directory = os.path.join(os.path.dirname(__file__), '..', 'data_folder')

def main():

  material_name = "KOSE"
  sub_directory = os.path.join(data_directory, "1_param", material_name)
  axis_names = ["Displacement", "Force"] 

  data.DataInterface.wipe_temp()

  ### Process F vs. D ###
  data_frames = []
  param_values = []

  # Process each file in the sub_directory
  for file_path in os.listdir(sub_directory):
    file_path = os.path.join(sub_directory, file_path)

    print(file_path)
    with open(file_path, 'r'):
      if file_path.endswith('.txt'):
        processed_data = data.DataInterface.process_data(file_path, axis_names)
        data_frames.append(processed_data)
        param_values.append(int(re.search(rf"{material_name}_(\d+)", file_path).group(1))) # Extract param value from filename
    
  data.DataInterface.display_graph(data_frames, 
    json_path=f"{sub_directory}/{material_name}_params.json", 
    params=param_values, 
    save_file=False, 
    args = {"xlabel": "Displacement (mm)", "ylabel": "Force (N)", "grid": True})

  ### Process AUC vs CFE ###
  for file_path in os.listdir(sub_directory):
    file_path = os.path.join(sub_directory, file_path)

    with open(file_path, 'r'):
      if file_path.endswith('.txt'):
        processed_data = data.DataInterface.process_data(file_path, axis_names)
        auc = operations.OperationsInterface.auc_trapz(processed_data, x_col="Displacement", y_col="Force", x_max=6.0)
        cfe = operations.OperationsInterface.cfe(processed_data["Force"].tolist())
        param_value = int(re.search(rf"{material_name}_(\d+)", file_path).group(1))
        print("-"*30)
        print(f"{material_name}_{param_value}, AUC: {round(auc, 2)}, CFE: {round(cfe, 2)}")
        print("-"*30)


if __name__ == "__main__":
  main()

