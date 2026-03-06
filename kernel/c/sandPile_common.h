#ifndef SANDPILE_COMMON_H
#define SANDPILE_COMMON_H

#include "easypap.h"
#include <stdbool.h>

// Type definition
typedef unsigned int TYPE;

// Global variables
extern TYPE *restrict TABLE;
extern int in;
extern int out;

// Table access macros and functions
static inline TYPE *atable_cell (TYPE *restrict i, int y, int x)
{
  return i + y * DIM + x;
}

#define atable(y, x) (*atable_cell (TABLE, (y), (x)))

static inline TYPE *table_cell (TYPE *restrict i, int step, int y, int x)
{
  return DIM * DIM * step + i + y * DIM + x;
}

#define table(step, y, x) (*table_cell (TABLE, (step), (y), (x)))

// Common functions
void swap_tables (void);
void sandPile_refresh_img (void);
void sandPile_draw_4partout (void);
void sandPile_draw_DIM (void);
void sandPile_draw_alea (void);
void sandPile_draw_big (void);
void sandPile_draw_spirals (void);

// Debug facilities
void sandPile_config (char *param);
void sandPile_debug (int x, int y);

// Macro to create kernel-specific aliases for common functions
#define SANDPILE_ALIAS(kernel, fun)                                            \
  void kernel##_##fun (void)                                                   \
  {                                                                            \
    sandPile_##fun ();                                                         \
  }

// Macro to create kernel-specific draw function
#define SANDPILE_DRAW_ALIAS(kernel)                                            \
  void kernel##_draw (char *param)                                             \
  {                                                                            \
    hooks_draw_helper (param, kernel##_draw_4partout);                         \
  }

#define SANDPILE_STR(x) #x

#ifdef __GNUC__
#define SANDPILE_UNROLL_LOOP(x) _Pragma (SANDPILE_STR (GCC unroll x))
#elif __clang__
#define SANDPILE_UNROLL_LOOP(x) _Pragma (SANDPILE_STR (unroll x))
#endif

// =========================================================================
// LAZY EVALUATION QUADTREE
// =========================================================================
//
// Flat-array quadtree over the tile grid for lazy evaluation. Each node is
// a single byte (CLEAN=0, DIRTY=1). The tree is stored level-by-level:
//
//   Level 0 (root):  1×1
//   Level 1:         2×2
//   Level k:         2^k × 2^k
//   Level depth-1:   n×n  (leaves = tiles)
//
// Index of node (level, ty, tx) = level_offset(level) + ty * 2^level + tx
// where level_offset(k) = (4^k - 1) / 3.
//
// Two arrays (cur / nxt) are used for double-buffered dirty tracking:
//   - cur: tiles to compute THIS iteration
//   - nxt: tiles to compute NEXT iteration (marked when a tile changes)
// After each iteration, swap cur/nxt and memset nxt to CLEAN.

#define QT_CLEAN 0
#define QT_DIRTY 1

typedef struct
{
  int n;              // padded dimension (next power of 2 from max(ntw,nth))
  int ntw, nth;       // actual tile grid dimensions
  int depth;          // number of levels (leaf level = depth - 1)
  int total_nodes;    // total nodes across all levels
  unsigned char *cur; // current iteration dirty flags
  unsigned char *nxt; // next iteration dirty flags
} LazyQT;

// Global quadtree instance (shared by both kernels)
extern LazyQT lazy_qt;

// Tile computation function pointer type (for quadtree traversal)
typedef int (*qt_tile_fn) (int x, int y, int width, int height);

// Lifecycle
void qt_init (void);
void qt_destroy (void);

// Marking
void qt_mark_all_dirty (void);
void qt_mark_one_dirty_nxt (int tx, int ty);
void qt_mark_dirty_with_neighbors (int tx, int ty);

// Iteration management
void qt_swap_and_clear (void);

// Traversal, returns 1 if any tile changed, 0 if fully stable
int qt_compute_iteration (qt_tile_fn tile_func);

// Inline helpers

static inline int qt_level_offset (int level)
{
  return ((1 << (2 * level)) - 1) / 3;
}

static inline int qt_level_width (int level)
{
  return 1 << level;
}

static inline int qt_node_index (int level, int ty, int tx)
{
  return qt_level_offset (level) + ty * qt_level_width (level) + tx;
}

static inline int qt_is_leaf_dirty (int tx, int ty)
{
  return lazy_qt.cur [qt_node_index (lazy_qt.depth - 1, ty, tx)] == QT_DIRTY;
}

// =========================================================================
// TILESETS FOR LAZY EVALUATION
// =========================================================================
// USAGE
//
// Meant to be used with the two provided tilsets.
// Both the bitsets must be initialized, they are identified by the vars
//   current_tile_set, next_tile_set
//
// One of the bitset must be defined as the current, and the other as the next
//
// The current tileset must be initialized as a full tileset
// The other must be initialized as an empty tileset empty one
// Every mark must be made to the next tileset
// At the end of an iteration, the sets should be switch
//
// =========================================================================
// DESCRIPTION
//
// TilesPerRow is the number of tiles per row
// TilesPerCol is the number of tiles per column
// K is the number of elements one bitset can hold (64 here)
//
// We use arrays of 64 bits integers to act as the rows of tiles.
// For each row, we allocate N bitset integers
//
// Then, one row of tile will need upper(K / TilesPerRow) bitsets
// The total number of bitset allocated is :
//   upper(K / TilesPerRow) * tilesPerCol
//
// Since the last bitset of a row may contain invalid indices
// The truncate function should be called when switching and allocating
//
// The bitsets act more like an iterator over the tiles.
// Getting the next tile is then akin to popping it.
typedef struct _tileset tileset;

typedef struct
{
  unsigned long tx, ty;
} tile;

extern tileset *restrict TILESET;

tileset *tileset_init (const unsigned tiles_per_row, const unsigned nb_rows);

void tileset_finalize (tileset *tileset);

const tile *tileset_flush_tiles (tileset *tileset);

const unsigned tileset_nb_tiles (const tileset *tileset);

void tileset_mark_at (tileset *tileset, const int change, const tile t);

void tileset_mark_full (tileset *tileset);

void tileset_mark_empty (tileset *tileset);

void tileset_trunc (tileset *tileset);

unsigned long tileset_get_total_tiles (tileset *tileset);

int tileset_merge (tileset *tile, tileset **others, unsigned nb_others);
int tileset_merge_omp (tileset *tile, tileset **others, unsigned nb_others);

#endif // SANDPILE_COMMON_H
