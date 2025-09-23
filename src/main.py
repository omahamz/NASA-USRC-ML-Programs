import glob
import re
import matplotlib
import random
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import simpson

data_directory = "test_data/HexFiles" # Mutable
file_basename = "HexTGV2"
angles_list, auc_list, cfe_list, pf_list, af_list = [], [], [], [], []
color_list = [ f"#{255:02x}{0:02x}{0:02x}", f"#{0:02x}{255:02x}{0:02x}", f"#{0:02x}{0:02x}{255:02x}" ]

# Area Under Curve w Trapizoid method
def get_auc(x, y, x_max=None):
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
    return np.trapz(y, x)

# Area Under Curve w Simpson's method
# def get_auc(x, y, x_max=None):
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
#     if len(x) % 2 == 0:
#         # Add an extra point by estimating between the last two points
#         x_extra = x[-1] + (x[-1] - x[-2])
#         y_extra = y[-1] + (y[-1] - y[-2])
#         x = np.append(x, x_extra)
#         y = np.append(y, y_extra)
#     return simpson(y, x)

# Peak Force
def find_peak(X):
  x_max = 0
  for x in X:
    if x > x_max:
      x_max = x
  return x_max

for file_path in glob.glob(data_directory + "/*.txt"): # Only read txt

  twist_angle = int(re.search(rf"{file_basename}_(\d+)", file_path).group(1)) # Mutable
  angles_list.append(twist_angle)
  X, Y = [], []

  # Creating X & Y
  with open(file_path) as f:
    for line in f:
      x, y = map(float, re.split(r"\s+", line.strip()))
      X.append(float(x))
      Y.append(float(y))

  # Peak Force
  peak_force = find_peak(Y)
  pf_list.append(peak_force)

  # Calculating Values
  auc = get_auc(X, Y)
  auc_list.append(auc)
  avg_force = auc / max(X)
  af_list.append(avg_force)
  cfe_list.append(avg_force / peak_force)

  # Normalize
  Y = [y / peak_force for y in Y]

  # Plotting each grpah
  # plt.figure() # - Uncomment for non superimposed
  plt.plot(X, Y, '-', label=str("control" if twist_angle == 0 else "V2"), color=color_list[twist_angle])

# Finalizing first plot
  plt.xlabel("Displacement (mm)", )
  plt.ylabel("Force (N)")
  plt.title(f"HEXV2  | Force vs Displacement")
  plt.grid(True)

plt.legend()
plt.savefig(f"test_data/Graphs/{file_basename}_graph.png") # Mutable
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
print(f"PF:\n {pf_list}")
print(f"af:\n {af_list}")

