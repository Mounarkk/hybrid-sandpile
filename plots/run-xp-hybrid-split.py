#!/usr/bin/env python3
from expTools import *

output_file = "hybrid_split_perf.csv"

split_percentages = [str(x) for x in range(10, 100, 10)]

easypapOptions = {
    "-k": ["ssandPile"],
    "-g": [""],
    "-v": ["ocl_hybrid"],
    "-a": ["spirals"],
    "-s": [2048],
    "-th": [16],
    "-tw": [32],
    "-i": ["2000"],
    "-c": split_percentages,
    "-of": [output_file],
}

ompICV = {
  "OMP_SCHEDULE": ["static"],
  "OMP_NUM_THREADS":  ["16"],
  "OMP_PLACES":       ["cores"],
}

nbruns = 20

execute("./run ", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")
