import glob
import re
import numpy as np

data_folder_directory = "test_data/KOSE" # Mutable
angles_list, auc_list, cfe_list = [], [], []

plt.figure()


for file_path in glob.glob(data_folder_directory + "/*.txt"): # Only read txt

  twist_angle = int(re.search(r"KOSE(\d+)", file_path).group(1)) # Mutable
  angles_list.append(twist_angle)
  X, Y = [], []

  # Creating X & Y
  with open(file_path) as f:
    for line in f:
      x, y = re.split(r" +", line.strip()) # Mutable
      X.append(float(x))
      Y.append(float(y))

  # Calculating CFE
  median_force = sum(Y)/len(Y)
  peak_force = find_peak(Y)
  cfe_list.append(median_force / peak_force)

  # Calculating AUC
  auc = auc_trapz(X, Y)
  auc_list.append(auc)

  # Plotting each grpah
  g = int(np.clip(twist_angle * 7, 0, 255)) # map angle -> 0..255
  graph_color = f'#{g:02x}{g:02x}{g:02x}'
  plt.figure()
  plt.plot(X, Y, '-', label=str(twist_angle), color=graph_color) # Color on a Grayscale

  # Finalizing first plot
  plt.xlabel("Displacement (mm)", )
  plt.ylabel("Force (N)")
  plt.title(f"KOSH Twist Angle of {twist_angle} degrees")
  plt.grid(True)
  plt.show()

# Creating secondary plot
plt.figure()
plt.plot(auc_list, cfe_list, "o")
plt.xlabel("AUC (Nmm)")
plt.ylabel("CFE")
plt.ylim(0, 1) # Mutable
plt.title("Crushing Force Efficiency vs Area Under Curve (Nmm)")
plt.grid(True)

for i in range(len(cfe_list)):
  plt.text(auc_list[i], cfe_list[i], str(angles_list[i]))

plt.show()
