#!/usr/bin/env python3
from expTools import *

output_file = "hybrid_bt_perf.csv"

# We sweep the multiplier (1 to 5) with a fixed GPU split
bt_configs = ["60:1", "60:2", "60:3", "60:4", "60:5"]

# Common options
common_options = {
    "-k": ["ssandPile"],
    "-v": ["ocl_hybrid_thick"],
    "-g": [""],
    "-th": [16],
    "-tw": [32],
    "-c": bt_configs,
    "-of": [output_file],
}

ompICV = {
  "OMP_SCHEDULE": ["static"],
  "OMP_NUM_THREADS":  ["16"],
  "OMP_PLACES":       ["cores"],
}

nbruns = 5

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
