#!/usr/bin/env python3
from expTools import *

output_file = "hybrid_speedup_perf.csv"

# Threads to test
thread_counts = [1, 2, 4, 6, 8, 12, 16, 20, 24]
# Schedules to test
schedules = ["static", "dynamic", "guided"]

# Common options
common_options = {
    "-k": ["ssandPile"],
    "-v": ["ocl_hybrid_thick"],
    "-c": ["60:1"],  # 60% GPU, 1xTILE_H border thickness
    "-g": [""],
    "-th": [16],
    "-tw": [32],
    "-of": [output_file],
}

ompICV = {
  "OMP_SCHEDULE": schedules,
  "OMP_NUM_THREADS": thread_counts,
  "OMP_PLACES":       ["cores"],
}

nbruns = 20

# --- Case C1: Dense 4096 ---
easypapOptions_C1 = common_options.copy()
easypapOptions_C1.update({
    "-a": ["4partout"],
    "-s": [4096],
    "-i": ["1000"],
    "--label": ["C1_Dense_4096"]
})
execute("./run ", ompICV, easypapOptions_C1, nbruns, verbose=False, easyPath=".")

# --- Case C2: Dense 1024 ---
easypapOptions_C2 = common_options.copy()
easypapOptions_C2.update({
    "-a": ["4partout"],
    "-s": [1024],
    "-i": ["4000"],
    "--label": ["C2_Dense_1024"]
})
execute("./run ", ompICV, easypapOptions_C2, nbruns, verbose=False, easyPath=".")

# --- Case C3: Sparse 2048 ---
easypapOptions_C3 = common_options.copy()
easypapOptions_C3.update({
    "-a": ["spirals"],
    "-s": [2048],
    "-i": ["2000"],
    "--label": ["C3_Sparse_2048"]
})
execute("./run ", ompICV, easypapOptions_C3, nbruns, verbose=False, easyPath=".")
