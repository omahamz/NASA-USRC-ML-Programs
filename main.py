import glob
import re
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import simpson

data_folder_directory = "/content/KOSHParsedTxtFiles/"
angles_list, auc_list, cfe_list = [], [], []

# Area Under Curve w Simpson's method
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
        return simpson(y[:-1], x[:-1]) + np.trapezoid(y[-2:], x[-2:]) # Changed np.trapz to np.trapezoid

# Peak Force
def find_peak(X):
  x_max = 0
  for x in X:
    if x > x_max:
      x_max = x
  return x_max


plt.figure(figsize=(10, 6)) # Added figure for clarity

for file_path in glob.glob(data_folder_directory + "/*.txt"): # Only read txt

  twist_angle = int(re.search(r"KOSH(\d+)parsed", file_path).group(1)) # Mutable
  angles_list.append(twist_angle)
  X, Y = [], []

  # Creating X & Y
  with open(file_path) as f:
    for line in f:
      x, y = line.split("\t") # Mutable
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
  plt.plot(X, Y, '-', label=f'{twist_angle} deg') # Added f-string for label and removed color for now


# Finalizing first plot
plt.xlabel("Displacement (mm)")
plt.ylabel("Force (N)")
plt.title("KOSH Twist Angle Force vs Displacement Graphs")
plt.legend() # Moved legend() outside the loop
plt.grid(True)
plt.show()

# Creating secondary plot
plt.figure()
plt.plot(angles_list, auc_list, "o")
plt.xlabel("Twist Angle (deg)")
plt.ylabel("AUC (Nmm)")
plt.title("KOSH Twist Angle vs AUC")
plt.grid(True)
plt.show()

# Creating tertiary plot
plt.figure()
plt.plot(angles_list, cfe_list, "o")
plt.xlabel("Twist Angle (deg)")
plt.ylabel("Crushing Force Efficiency")
plt.title("KOSH Twist Angle vs CFE")
plt.grid(True)
plt.show()