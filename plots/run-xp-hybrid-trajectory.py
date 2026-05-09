#!/usr/bin/env python3
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure the output directory exists
os.makedirs("imgs/hybrid", exist_ok=True)

# We define scenarios: (Case, Start Percentage)
# C1 starts low (40%) to see it go UP
# C3 starts high (90%) to see it go DOWN
scenarios = [
    {"label": "C1_Dense_4096", "s": 4096, "i": 1000, "a": "4partout", "start": 40},
    {"label": "C3_Sparse_2048", "s": 2048, "i": 2000, "a": "spirals", "start": 90}
]

alphas = [0.05, 0.20, 0.50]
all_data = []

# Prepare environment variables
my_env = os.environ.copy()
my_env["OMP_NUM_THREADS"] = "24"
my_env["OMP_SCHEDULE"] = "static"
my_env["OMP_PLACES"] = "cores"

print("Running dynamic trajectory experiments...")

for sc in scenarios:
    for alpha in alphas:
        print(f"  {sc['label']} | Start={sc['start']}% | alpha={alpha}...")
        
        config = f"{sc['start']}:1:{alpha}"
        cmd = [
            "./run", "-k", "ssandPile", "-g", "-v", "ocl_hybrid_dynamic",
            "-a", sc['a'], "-s", str(sc['s']), "-i", str(sc['i']),
            "-th", "16", "-tw", "32", "-c", config, "-n" 
        ]
        
        # Run with the specific OpenMP environment
        subprocess.run(cmd, env=my_env, stdout=subprocess.DEVNULL)
        
        if os.path.exists("trajectory.csv"):
            df = pd.read_csv("trajectory.csv", sep=';')
            df['scenario'] = f"{sc['label']} (Start {sc['start']}%)"
            df['alpha'] = alpha
            all_data.append(df)
            os.remove("trajectory.csv")

if not all_data:
    print("Error: No data collected. Did you recompile?")
    exit(1)

# 2. Plot the results
print("Generating multi-scenario trajectory plot...")
final_df = pd.concat(all_data)

sns.set_theme(style="whitegrid")
g = sns.FacetGrid(final_df, col="scenario", hue="alpha", height=5, aspect=1.2, sharey=False, palette="viridis")
g.map(sns.lineplot, "iteration", "gpu_y_end", linewidth=2)

# Add horizontal line for the initial split on each plot
for ax, (name, sub_df) in zip(g.axes.flat, final_df.groupby('scenario', sort=False)):
    # Extract starting percentage from name
    start_pct = int(name.split("Start ")[1].split("%")[0])
    # Extract DIM from name
    dim = int(name.split("_")[-1].split(" ")[0])
    initial_y = (dim * start_pct) // 100
    ax.axhline(y=initial_y, color='red', linestyle='--', alpha=0.6, label=f'Initial ({start_pct}%)')

g.add_legend(title="Alpha (Smoothing)")
g.set_axis_labels("Iteration (Super-steps)", "gpu_y_end (GPU Rows)")
g.set_titles("{col_name}")

plt.subplots_adjust(top=0.85)
g.fig.suptitle("Dynamic Load Balancing: UP and DOWN trajectories", fontsize=16)

plot_path = "imgs/hybrid/dynamic_trajectory_comparison.png"
plt.savefig(plot_path, dpi=300)
print(f"Plot saved to {plot_path}")
