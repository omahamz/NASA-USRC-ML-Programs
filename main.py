import glob
import re
import matplotlib
import random
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import simpson

data_folder_directory = "test_data/HexFiles" # Mutable
angles_list, auc_list, cfe_list = [], [], []
color_list = [ f"#{255:02x}{0:02x}{0:02x}", f"#{0:02x}{255:02x}{0:02x}", f"#{0:02x}{0:02x}{255:02x}" ]

# Area Under Curve w Trapizoid method
# def auc_simps(x, y, x_max=None):
#     x, y = np.asarray(x), np.asarray(y)
#     order = np.argsort(x)
#     x, y = x[order], y[order]
#     if x_max is not None:
#         # clip to x_max (linear interpolate last point)
#         if x_max < x[-1]:
#             i = np.searchsorted(x, x_max)
#             x_clip = np.concatenate([x[:i], [x_max]])
#             y_clip = np.concatenate([y[:i],[np.interp(x_max, x[i-1:i+1], y[i-1:i+1])]])
#             x, y = x_clip, y_clip
#     return np.trapz(y, x)

def auc_simps(x, y, x_max=None):
    x, y = np.asarray(x), np.asarray(y)
    order = np.argsort(x)
    x, y = x[order], y[order]
    if x_max is not None:
        # clip to x_max (linear interpolate last point)
        if x_max < x[-1]:
            i = np.searchsorted(x, x_max)
            x_clip = np.concatenate([x[:i], [x_max]])
            y_clip = np.concatenate([y[:i],[np.interp(x_max, x[i-1:i+1], y[i-1:i+1])]])
            x, y = x_clip, y_clip
    if len(x) % 2 != 0:
        return simpson(y, x)
    else:
        return simpson(y[:-1], x[:-1]) + np.trapz(y[-2:], x[-2:])

# Peak Force
def find_peak(X):
  x_max = 0
  for x in X:
    if x > x_max:
      x_max = x
  return x_max

for file_path in glob.glob(data_folder_directory + "/*.txt"): # Only read txt

  twist_angle = int(re.search(r"HexTGV2_(\d+)", file_path).group(1)) # Mutable
  angles_list.append(twist_angle)
  X, Y = [], []

  # Creating X & Y
  with open(file_path) as f:
    for line in f:
      x, y = map(float, re.split(r"\s+", line.strip()))
      X.append(float(x))
      Y.append(float(y))

  # Calculating CFE
  median_force = sum(Y)/len(Y)
  peak_force = find_peak(Y)
  cfe_list.append(median_force / peak_force)

  # Calculating AUC
  auc = auc_simps(X, Y)
  auc_list.append(auc)

  # Plotting each grpah
  # plt.figure() # - Uncomment for non superimposed
  plt.plot(X, Y, '-', label=str("control" if twist_angle == 0 else "V2"), color=color_list[twist_angle])

# Finalizing first plot
  plt.xlabel("Displacement (mm)", )
  plt.ylabel("Force (N)")
  plt.title(f"HEXV2  | Force vs Displacement")
  plt.grid(True)

plt.legend()
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

print(f"AUC:\n {auc_list}")
print(f"CFE:\n {cfe_list}")

