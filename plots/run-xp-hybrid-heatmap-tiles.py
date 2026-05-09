#!/usr/bin/env python3
from expTools import *
import os

output_file = "hybrid_tiles_heatmap.csv"

# Powers of two to test
sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

# Hardware limit for workgroup size (TILE_W * TILE_H)
MAX_WORKGROUP_SIZE = 1024

def run_case(label, dim, iterations, algo):
    print(f"--- Running Heatmap for {label} ---")
    
    configs = []
    
    for tw in sizes:
        for th in sizes:
            # 1. Check OpenCL hardware limit
            if tw * th > MAX_WORKGROUP_SIZE:
                continue
            
            # 2. Check dimension limits
            if tw > dim or th > dim:
                continue
                
            # 3. Check split feasibility (60% split)
            # We want at least one tile row for GPU and one for CPU
            gpu_y_end = (dim * 60 // 100) // th * th
            if gpu_y_end <= 0 or gpu_y_end >= dim:
                continue
            
            configs.append((tw, th))

    if not configs:
        return

    # Extract zipped tw and th lists
    tw_list, th_list = zip(*configs)

    # Execute the sweep
    common_options = {
        "-k": ["ssandPile"],
        "-g": [""],
        "-v": ["ocl_hybrid_thick"],
        "-c": ["60:1"],
        "-th": list(th_list),
        "-tw": list(tw_list),
        "-s": [dim],
        "-i": [iterations],
        "-a": [algo],
        "--label": [label],
        "-of": [output_file],
    }
    
    ompICV = {
        "OMP_NUM_THREADS": [24],
        "OMP_SCHEDULE": ["static"],
        "OMP_PLACES": ["threads", "cores", "sockets"],
    }

    # To be safe and avoid the cross-product of lists,
    # we iterate and call execute for each valid pair and each place.
    for tw, th in configs:
        for place in ompICV["OMP_PLACES"]:
            opts = common_options.copy()
            opts["-tw"] = [tw]
            opts["-th"] = [th]
            icv = ompICV.copy()
            icv["OMP_PLACES"] = [place]
            execute("./run ", icv, opts, nbruns=1, verbose=False, easyPath=".")

# Clear old results
if os.path.exists(output_file):
    os.remove(output_file)

# Run for all three cases
run_case("C1_Dense_4096", 4096, 1000, "4partout")
run_case("C2_Dense_1024", 1024, 4000, "4partout")
run_case("C3_Sparse_2048", 2048, 2000, "spirals")
