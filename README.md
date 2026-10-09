# Hybrid CPU/GPU sandpile

A parallel **abelian sandpile** solver written as an [EasyPAP][easypap] kernel,
for the *Programmation des Architectures Parallèles* course of the Master
Informatique at the Université de Bordeaux (2025-2026).

The kernel starts from a sequential tiled loop and ends on a hybrid execution
where the GPU and the 24 CPU cores each own a horizontal slice of the grid,
exchange their shared border, and renegotiate the split at runtime from the
time the GPU spends waiting.

This repository holds the **synchronous kernel** (`ssandPile`) and the
OpenMP/OpenCL work built on it, pulled out of the course repository with its
history.  It is not standalone: EasyPAP provides the window, the timing
harness, the tiling and the OpenCL boilerplate, so the files here are meant to
be dropped into a checkout of it.

## The problem

Every cell of a square grid holds a number of grains.  A cell holding four or
more is unstable: it gives one grain to each of its four neighbours and keeps
the rest.  Applying this until no cell is unstable yields the same final
configuration whatever order the topplings are applied in, which is what makes
the rule interesting to parallelise.

The synchronous variant computes each new state from the previous one, so the
whole grid can be updated at once.  Two properties drive every optimisation
below: activity is sparse and clustered, and it spreads by at most one cell per
iteration.

## Layout

```
kernel/c/ssandPile.c         synchronous kernel, every CPU and hybrid variant
kernel/c/sandPile_common.c   quadtree and bitset used to track active tiles
kernel/c/sandPile_common.h
kernel/ocl/ssandPile.cl      OpenCL kernels, including the GPU half of the split
plots/run-xp-hybrid-*.py     the four hybrid experiments
plots/run-xp-sandPile-seq-incremental.py
plots/expTools.py            EasyPAP experiment helpers (upstream)
plots/graphTools.py
results/                     measured output of the hybrid experiments
etape-3/evaluation/          configurations submitted for the graded runs
etape-{2,3}/rapport.pdf      the two written reports
```

## Build and run

EasyPAP is not vendored here.  Clone it, drop these files in, and build:

```sh
git clone https://gitlab.com/gforgeron/easypap-se.git
cd easypap-se
cp -r /path/to/hybrid-sandpile/kernel/c/*.c   kernel/c/
cp -r /path/to/hybrid-sandpile/kernel/c/*.h   kernel/c/
cp    /path/to/hybrid-sandpile/kernel/ocl/ssandPile.cl kernel/ocl/
cp    /path/to/hybrid-sandpile/plots/run-xp-*.py       plots/
make -j
```

`sandPile_common.h` only adds declarations, so the `asandPile` kernel shipped
by EasyPAP still compiles next to these files.

A few runs, from the EasyPAP root:

```sh
# sequential reference
./run -k ssandPile -v seq -s 1024 -a spirals -n

# lazy OpenMP, 24 threads pinned to cores
OMP_NUM_THREADS=24 OMP_PLACES=cores OMP_PROC_BIND=close \
    ./run -k ssandPile -v omp_lazy_border -wt opt_avx -th 8 -tw 16 \
          -s 2048 -a spirals -n

# GPU only
./run -k ssandPile -g -v ocl_multi_m5_static -s 4096 -th 16 -tw 32 -i 1000 -n

# hybrid, GPU takes 60% of the rows, border 4 tile rows thick
./run -k ssandPile -g -v ocl_hybrid_thick -c 60:4 -s 4096 -a 4partout \
      -th 16 -tw 32 -i 1000 -n

# dynamic rebalancing, starting at 60%, four-tile border, alpha = 0.2
./run -k ssandPile -g -v ocl_hybrid_dynamic -c 60:4:0.2 -s 4096 -a 4partout \
      -th 16 -tw 32 -i 1000 -n
```

Drop `-n` to watch it run in the EasyPAP window.

## Variants

| `-v` | What it does |
|---|---|
| `seq`, `tiled` | sequential reference, then tiled traversal |
| `omp_tiled`, `omp_taskloop` | OpenMP over tiles, with a `for` and with tasks |
| `lazy`, `omp_lazy` | skip tiles with no activity, tracked in a quadtree and a bitset |
| `omp_lazy_border` | only wake a neighbour tile when activity reaches the border |
| `*_avx`, `opt_avx` | AVX2 tile kernels, lateral and vertical traversal |
| `ocl` | straight OpenCL port, one work item per cell |
| `ocl_multi_m4_loop` | several iterations per kernel launch |
| `ocl_multi_m5_static` | fastest GPU variant, local-memory tile with a halo |
| `ocl_multi_m5_static_quad` | the same, restricted to the active quadrant |
| `ocl_multi_m6_robust` | guards the cases where a tile does not divide the grid |
| `ocl_multi_m8_static` | wider work per item, fewer launches |
| `ocl_hybrid` | CPU and GPU split, border exchanged every iteration |
| `ocl_hybrid_thick` | thick border, exchanged every *k* iterations instead |
| `ocl_hybrid_dynamic` | moves the split at runtime from the GPU wait time |

## The hybrid split

The grid is cut horizontally.  The GPU owns rows `[0, gpu_y_end)`, the CPU owns
`[gpu_y_end, DIM)`, and `gpu_y_end` is rounded down to a whole number of tile
rows.  Both sides need the row the other one owns next to the cut, so one row
is copied each way after every iteration, which costs two small transfers and a
synchronisation.

Exchanging a single row per iteration makes those transfers dominate.
`ocl_hybrid_thick` instead gives each side a border of `k` tile rows and lets
both compute `k` iterations before exchanging, trading redundant work in the
overlap against a *k*-fold drop in transfers.  The second field of `-c` is that
`k` as a multiple of the tile height, so `-c 60:4 -th 16` means a 64-row border
and 64 iterations between exchanges.  Activity spreading by one cell per
iteration is what makes the redundant computation correct.

A fixed split is only right for one grid, one pattern and one machine.
`ocl_hybrid_dynamic` measures, over each batch of `k` iterations, the total
time and the time the CPU side was busy; the difference is how long the GPU sat
idle.  That signal is noisy, so it is smoothed with an exponentially weighted
moving average,

```
ewma = alpha * gpu_wait + (1 - alpha) * ewma
```

Above 500 us of smoothed idle time the GPU gives up a tile row, below 100 us it
takes one, and in between the split is left alone, so it settles instead of
oscillating.  `alpha` is the third field of `-c`, so `-c 60:4:0.2` starts the
GPU on 60% of the rows, uses a four-tile border and reacts with
`alpha = 0.2`.  Ownership is
checked on both sides before a cell is written, otherwise the overlap lets the
two halves disagree and the result stops matching the sequential reference.

## Experiments

Each script drives EasyPAP over a parameter sweep and appends to a CSV with the
usual EasyPAP columns, which `results/` keeps alongside the figure drawn from
it.

| Script | Question it answers | Output |
|---|---|---|
| `run-xp-hybrid-split.py` | Where is the best fixed split?  Sweeps the GPU share from 10% to 90%, 20 runs each. | `hybrid_split_perf.{csv,pdf}` |
| `run-xp-hybrid-bt.py` | How much does the thick border buy, and when does the redundant work cancel it? | `hybrid_bt_perf.{csv,pdf}` |
| `run-xp-hybrid-speedup.py` | Speedup of the hybrid against the GPU alone and the CPU alone, over thread counts and schedules. | `hybrid_speedup_perf.{csv,pdf}` |
| `run-xp-hybrid-heatmap-tiles.py` | Which tile geometries are usable at all, given the 1024 work-item limit and the split, and which are fast? | `hybrid_tiles_heatmap.{csv,pdf}` |
| `run-xp-hybrid-trajectory.py` | Does the dynamic split converge, from both below and above, and how does `alpha` change the path? | `imgs/hybrid/dynamic_trajectory_comparison.png` |

The trajectory figure is not committed: the run writes `trajectory.csv` in the
EasyPAP root and the script redraws it, so it regenerates in one command.

## Tuned configurations

The configurations submitted for the graded runs of the third stage, one per
grid size.  `etape-3/evaluation/` holds them verbatim.

| Size | Configuration |
|---|---|
| 1024 | `omp_lazy_border` with `opt_avx` tiles, 8x16, 4 threads on cores |
| 2048 | the same, at 2048 |
| 4096 | `ocl_multi_m5_static_quad`, 16x32 tiles, on the GPU |

Four threads winning at 1024 and 2048 is not a mistake.  Once the lazy kernel
has removed the stable tiles there is little work left, and spreading it over
24 hybrid cores costs more in synchronisation and in E-core latency than it
returns.

The two remaining sizes of the graded set, 512 and 8192, were won by the
asynchronous kernel, which is not part of this repository.

## Hardware

Room 104 of the university, reported by EasyPAP as the `patrin` machine,
GCC 12:

- 8 P-cores and 16 E-cores, AVX2, so every CPU number here is on a
  deliberately asymmetric machine
- NVIDIA RTX 4000 Ada

The P-core and E-core split matters for the measurements.  Thread placement
(`OMP_PLACES`, `OMP_PROC_BIND`) changes the results more than the schedule
does, and the dynamic split reacts to it on its own.

## Reports

Two reports, in French, written over the course of the project:

- [`etape-2/rapport.pdf`](etape-2/rapport.pdf): lazy evaluation, the quadtree
  and the bitset, thread placement on the P-core/E-core split
- [`etape-3/rapport.pdf`](etape-3/rapport.pdf): AVX2 traversals, the OpenCL
  port, the GPU memory hierarchy and the kernel variants

## Division of work

The project was done in a pair with **Mathis Foussac**, and the history here is
the real one, so both names appear on it.

- The OpenCL kernels, every `ocl_*` variant and the whole hybrid CPU/GPU work,
  including the dynamic split, are mine.
- The lazy evaluation of the second stage and the AVX2 tile kernels are
  Mathis's.  The hybrid CPU side calls them, which is why they are here.
- The OpenMP work of the first stage, the shared quadtree and bitset, and the
  two reports were written together.

Mathis also wrote an MPI + OpenMP version of the kernel for the last stage,
on his own branch.  It is not part of this repository.

## Acknowledgements

[EasyPAP][easypap] is the teaching framework of Raymond Namyst and
Pierre-André Wacrenier, who also wrote the kernel skeletons and the experiment
helpers kept here.  It is distributed under the BSD 3-Clause licence, which
`LICENSE` and `AUTHORS` reproduce.

[easypap]: https://gitlab.com/gforgeron/easypap-se
