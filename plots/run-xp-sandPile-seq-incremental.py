#!/usr/bin/env python3
from expTools import *

output_file = "sandpile_seq_incremental.csv"

# Experience setup (Time measurement on varied tiling optimizations)
easypapOptions = {
    "-k": ["ssandPile", "asandPile"],  
    "-v": ["seq"],                     
    "-s": [64],    # Une seule taille suffisante pour démontrer l'évolution
    # La liste stricte de nos variantes du moins bon au meilleur
    "-wt": ["default", "opt1", "opt2", "opt3", "opt"],         
    "-of": [output_file],              
}

# Force 1 thread as it's pure sequential benchmarking
ompICV = {"OMP_NUM_THREADS": [1]}

# Moyenne sur 3 executions
nbruns = 10

# Execution
execute("./run ", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".", filter=lambda comb: True)

print(f"Data saved to {output_file}")
print("-" * 60)
print("COMMANDES POUR GENERER LES GRAPHIQUES (NUAGES DE POINTS CATPLOT):")
print(f"python3 plots/easyplot.py -if {output_file} -k ssandPile -y time -x tiling --plottype catplot --kind point")
print(f"python3 plots/easyplot.py -if {output_file} -k asandPile -y time -x tiling --plottype catplot --kind point")
print("-" * 60)