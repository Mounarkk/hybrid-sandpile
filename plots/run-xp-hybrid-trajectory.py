#!/usr/bin/env python3
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure the output directory exists
os.makedirs("imgs/hybrid", exist_ok=True)

# Cases to test
cases = [
    {"label": "C1_Dense_4096", "s": 4096, "i": 1000, "a": "4partout"},
    {"label": "C2_Dense_1024", "s": 1024, "i": 4000, "a": "4partout"},
    {"label": "C3_Sparse_2048", "s": 2048, "i": 2000, "a": "spirals"}
]

# Alphas to compare
alphas = [0.05, 0.20, 0.50]

all_data = []

print("Running dynamic trajectory experiments...")

for case in cases:
    for alpha in alphas:
        print(f"  Case {case['label']} | alpha={alpha}...")
        
        # Initial split starts at 50% to see the adaptation
        # Config format: start_pct:bt_mult:alpha
        config = f"50:1:{alpha}"
        
        cmd = [
            "./run", "-k", "ssandPile", "-g", "-v", "ocl_hybrid_dynamic",
            "-a", case['a'], "-s", str(case['s']), "-i", str(case['i']),
            "-th", "16", "-tw", "32", "-c", config, "-n" 
        ]
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL)
        
        if os.path.exists("trajectory.csv"):
            df = pd.read_csv("trajectory.csv", sep=';')
            df['case'] = case['label']
            df['alpha'] = alpha
            all_data.append(df)
            os.remove("trajectory.csv")

if not all_data:
    print("Error: No data collected. Did you recompile?")
    exit(1)

# 2. Plot the results
print("Generating multi-case trajectory plot...")
final_df = pd.concat(all_data)

sns.set_theme(style="whitegrid")
# Create one column per case to see how alpha affects each one
g = sns.FacetGrid(final_df, col="case", hue="alpha", height=5, aspect=1.2, sharey=False, palette="viridis")
g.map(sns.lineplot, "iteration", "gpu_y_end", linewidth=2)

# Add horizontal line for the initial 50% split on each plot
for ax in g.axes.flat:
    # Get dimension from the title
    title = ax.get_title()
    dim = int(title.split('_')[-1])
    ax.axhline(y=dim // 2, color='red', linestyle='--', alpha=0.6, label='Initial (50%)')

g.add_legend(title="Alpha (Smoothing)")
g.set_axis_labels("Iteration (Super-steps)", "gpu_y_end (GPU Rows)")
g.set_titles("{col_name}")

plt.subplots_adjust(top=0.85)
g.fig.suptitle("Dynamic Boundary Adaptation: Impact of Alpha across Scenarios", fontsize=16)

plot_path = "imgs/hybrid/dynamic_trajectory_comparison.png"
plt.savefig(plot_path, dpi=300)
print(f"Plot saved to {plot_path}")
