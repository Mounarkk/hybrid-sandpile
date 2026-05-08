#!/usr/bin/env python3
from expTools import *

output_file = "hybrid_bt_perf.csv"

# We use the syntax gpu_pct:border_thickness_mult
bt_configs = ["60:1", "60:2", "60:3", "60:4", "60:5"]

easypapOptions = {
    "-k": ["ssandPile"],
    "-g": [""],
    "-v": ["ocl_hybrid_thick"],
    "-a": ["4partout"],
    "-s": [4096],
    "-th": [16],
    "-tw": [32],
    "-i": ["1000"],
    "-c": bt_configs,
    "-of": [output_file],
}

ompICV = {
  "OMP_SCHEDULE": ["static"],
  "OMP_NUM_THREADS":  ["16"],
  "OMP_PLACES":       ["cores"],
}

nbruns = 3

execute("./run ", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")
