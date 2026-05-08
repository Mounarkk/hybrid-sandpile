#!/usr/bin/env python3
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure the output directory exists
os.makedirs("imgs/hybrid", exist_ok=True)

# 1. Run the experiment
# We start at 50% to see the transition towards the optimal split
print("Running dynamic trajectory experiment...")
# -v ocl_hybrid_dynamic : dynamic variant
# -a spirals           : sparse case
# -s 2048              : size
# -i 4000              : max iterations
# -c 50:1:0.1          : start_pct:bt_mult:alpha
cmd = "./run -k ssandPile -g -v ocl_hybrid_dynamic -a spirals -s 2048 -i 4000 -th 16 -tw 32 -c 60:1:0.1 -n -g"
subprocess.run(cmd.split())

if not os.path.exists("trajectory.csv"):
    print("Error: trajectory.csv was not generated. Did you recompile?")
    exit(1)

# 2. Plot the results
print("Generating trajectory plot...")
df = pd.read_csv("trajectory.csv", sep=';')

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="iteration", y="gpu_y_end", color="royalblue", linewidth=2.5)

# Initial split line
plt.axhline(y=df['gpu_y_end'].iloc[0], color='red', linestyle='--', label='Initial Split (50%)')

plt.title("Dynamic boundary adaptation over time", fontsize=16)
plt.xlabel("Iteration", fontsize=12)
plt.ylabel("gpu_y_end (rows allocated to GPU)", fontsize=12)
plt.legend()

plt.tight_layout()
plt.savefig("imgs/hybrid/dynamic_trajectory.png", dpi=300)
print("Plot saved to imgs/hybriddynamic_trajectory.png")
