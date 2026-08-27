# Standard Libary
import os
import glob
import re

# src imports
from data import DataInterface as DI
from operations import OperationsInterface as OI

data_directory = os.path.join(os.path.dirname(__file__), '..', 'data_folder')

def main():

  material_name = "temp2" #input("What Material <KOSE, KOSH, HEX>?: ")
  sub_directory = os.path.join(data_directory, "1_param/Mesh_Conversion_Sample", material_name)

  DI.wipe_temp()

  data_frames = { m : [] for m in OI.Mode } # Dictionary to hold data frames for each mode
  param_values = []
  corresponding_names = [] # Temporary solution
  ### Process each file in the sub_directory ###

  cutoffs = {60: 17.5, 160: 14, 175: 15, 190: 18}
  for file_path in os.listdir(sub_directory):
    file_path = os.path.join(sub_directory, file_path)

    if file_path.endswith('.txt'):
      #param_values.append(int(re.search(rf"{material_name}_(\d+)", file_path).group(1))) # (\d+)
      #param_values.append(int(re.search(r"A(\d+)", file_path).group(1))) # Extract param value from filename
      param_values.append(int(file_path[-5])) # Extract param value from filename
      if re.search(r"TA", file_path):
        param_values[-1] += 100
      ### F vs. D ###
      processed_data = DI.process_data(data_path=file_path, col_names=["Displacement", "Force"], normalize=False) #cutoff=cutoffs.get(param_values[-1], 20))
      data_frames[OI.Mode.FvD].append(processed_data)
      ### AUC and CFE ###
      auc = OI.auc_trapz(processed_data, x_col="Displacement", y_col="Force") # Convert to Nmm
      cfe = OI.cfe(processed_data)
      print("yuh", param_values)
      data_frames[OI.Mode.AvC].append([auc, cfe])
      corresponding_names = ["T1", "T2", "T3"] #corresponding_names.append(param_values[-1]) # Temporary solution

  # ### F vs. D ###
  # for i in range(len(data_frames[OI.Mode.FvD])):
  #   DI.display_graph(OI.Mode.FvD,
  #   data_frames[OI.Mode.FvD][i], 
  #   json_path=f"{sub_directory}/{material_name}_params.json", 
  #   params=param_values[i], 
  #   save_file=True, 
  #   args = {"xlabel": "Displacement (mm)", 
  #           "ylabel": "Force (N)", 
  #           "grid": False})
  # DI.display_graph(OI.Mode.FvD,
  #   data_frames[OI.Mode.FvD], 
  #   json_path=f"{sub_directory}/{material_name}_params.json", 
  #   params=param_values, 
  #   save_file=True, 
  #   args = {"xlabel": "Displacement (mm)", 
  #           "ylabel": "Force (N)", 
  #           "grid": False,
  #           "title": "FvD: HEX v Control Normalzied"})
  
  ### F vs. D TEMP ###
  # for i in range(len(data_frames[OI.Mode.FvD])):
  #   DI.display_graph(OI.Mode.FvD,
  #   data_frames[OI.Mode.FvD][i], 
  #   json_path=f"{sub_directory}/{material_name}_params.json", 
  #   params=param_values[i], 
  #   save_file=True, 
  #   args = {"xlabel": "Displacement (mm)", 
  #           "ylabel": "Force (N)", 
  #           "grid": False})
  DI.display_graph(OI.Mode.FvD,
    data_frames[OI.Mode.FvD], 
    json_path=f"{sub_directory}/{material_name}_params.json", 
    params=param_values, 
    save_file=True, 
    args = {"xlabel": "Displacement (mm)", 
            "ylabel": "Force (N)", 
            "grid": False,
            "title": "Angled Hex: FvD",})
    
  ### Single F vs. D ###
  
  ### SINGLE ###
  # DI.display_graph(OI.Mode.FvD,
  #   data_frames[OI.Mode.FvD][1], 
  #   json_path=f"{sub_directory}/{material_name}_params.json", 
  #   params=param_values, 
  #   save_file=False, 
  #   args = {"xlabel": "Displacement (mm)", 
  #           "ylabel": "Force (N)", 
  #           "grid": True,
  #           "title": "KOSE Twist Angle of 10 degrees",})
  
  ### AUC and CFE ###
  processed_data = DI.process_data(data_list=data_frames[OI.Mode.AvC], col_names=["AUC", "CFE"])
  data_frames[OI.Mode.AvC] = processed_data
  DI.display_graph(OI.Mode.AvC,
    data_frames[OI.Mode.AvC], 
    json_path=f"{sub_directory}/{material_name}_params.json", 
    params=param_values, 
    save_file=True, 
    args = {"xlabel": "Area Under Curve (Nmm)", 
            "ylabel": "Crushing Force Efficiency", 
            "grid": True, 
            "marker": "o", 
            "title": "Angled Hex: AUC vs CFE",
            "linestyle": "None"})#
  
  print(data_frames[OI.Mode.AvC])
  print(corresponding_names)

# In main.py — add this function

def simplify_name(filename: str) -> str:
    """
    Divides each number in the filename by 100 and removes 'U_FD' suffix.
    E.g. N600AID2840R280A5500CC1600VC500U_FD -> N6AID28.40R2.80A55CC16VC5
    """
    base = re.sub(r'U_FD$', '', filename)  # strip trailing U_FD
    def divide_match(m):
        val = int(m.group()) / 100
        # Format: drop trailing zeros but keep decimals if needed
        return f"{val:g}"
    return re.sub(r'\d+', divide_match, base)


def main2():
    """
    Processes files in Mesh_Conversion_Sample/FDData<N> subdirectories.
    For each unique filename across mesh sizes, plots a superimposed F vs D graph.
    """
    mesh_dir = os.path.join(data_directory, "1_param/Mesh_Conversion_Sample")

    # Find all FDData<N> subdirectories
    mesh_groups: dict[str, dict[int, str]] = {}  # {base_name: {mesh_size: filepath}}

    for entry in os.listdir(mesh_dir):
        match = re.fullmatch(r'FDData(\d+)', entry)
        if not match:
            continue
        mesh_size = int(match.group(1))
        sub_dir = os.path.join(mesh_dir, entry)

        for fname in os.listdir(sub_dir):
            if not fname.endswith('.csv'):
                continue
            base_name = os.path.splitext(fname)[0]  # strip .csv
            if base_name not in mesh_groups:
                mesh_groups[base_name] = {}
            mesh_groups[base_name][mesh_size] = os.path.join(sub_dir, fname)

    if not mesh_groups:
        print("No CSV files found in FDData subdirectories.")
        return

    DI.wipe_temp()

    # Color scheme for mesh sizes (up to 3)
    mesh_colors = {1: 'red', 2: 'green', 3: 'blue', 4: 'yellow'}  # fallback cycling

    for base_name, mesh_files in mesh_groups.items():
        simplified = simplify_name(base_name)
        data_frames = []
        labels = []
        mesh_sizes_sorted = sorted(mesh_files.keys())

        for mesh_size in mesh_sizes_sorted:
            fpath = mesh_files[mesh_size]
            df = DI.process_data(
                data_path=fpath,
                col_names=["Time", "Displacement", "Force"],
                x_col="Displacement",
                y_col="Force"
            )
            data_frames.append(df)
            labels.append(f"Mesh {mesh_size}")

        DI.display_graph(
            OI.Mode.FvD,
            data_frames,
            json_path=None,           # no param JSON needed here
            params=mesh_sizes_sorted,
            labels=labels,
            save_file=True,
            args={
                "xlabel": "Displacement (mm)",
                "ylabel": "Force (N)",
                "grid": False,
                "title": f"FvD: {simplified}",
            }
        )
    
if __name__ == "__main__":
  main2()

