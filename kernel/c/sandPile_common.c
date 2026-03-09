#include "sandPile_common.h"
#include <stdalign.h>
#include <sys/mman.h>
#include <unistd.h>

// Global variables
TYPE *restrict TABLE = NULL;
int in               = 0;
int out              = 1;

// Swap tables for synchronous version
void swap_tables (void)
{
  int tmp = in;
  in      = out;
  out     = tmp;
}

// Refresh image function (common to both kernels)
void sandPile_refresh_img (void)
{
  for (int i = 1; i < DIM - 1; i++)
    for (int j = 1; j < DIM - 1; j++) {
      uint32_t c = table (in, i, j);
      uint8_t r = 0, g = 0, b = 0;

      if (c == 1) {
        g = 255;
        b = 199;
      } else if (c == 2) {
        b = 255;
        r = 92;
        g = 184;
      } else if (c == 3) {
        r = 255;
        g = 92;
        b = 92;
      } else if (c == 4)
        r = g = b = 255;
      else if (c > 4)
        r = b = 255 - (128 * 5.0f / (float)c);

      cur_img (i, j) = ezv_rgb (r, g, b);
    }
}

// Debug facilities
static int debug_hud = -1;

void sandPile_config (char *param)
{
  if (picking_enabled) {
    debug_hud = ezv_hud_alloc (ctx [0]);
    ezv_hud_on (ctx [0], debug_hud);
  }
}

void sandPile_debug (int x, int y)
{
  if (x == -1 || y == -1)
    ezv_hud_off (ctx [0], debug_hud);
  else {
    ezv_hud_on (ctx [0], debug_hud);
    ezv_hud_set (ctx [0], debug_hud, "#grains: %d", table (in, y, x));
  }
}

/////////////////////////////  Initial Configurations

static inline void set_cell (int y, int x, unsigned v)
{
  atable (y, x) = v;
  if (gpu_used)
    cur_img (y, x) = v;
}

void sandPile_draw_4partout (void)
{
  for (int i = 1; i < DIM - 1; i++)
    for (int j = 1; j < DIM - 1; j++)
      set_cell (i, j, 4);
}

void sandPile_draw_DIM (void)
{
  for (int i = DIM / 4; i < DIM - 1; i += DIM / 4)
    for (int j = DIM / 4; j < DIM - 1; j += DIM / 4)
      set_cell (i, j, i * j / 4);
}

// Deterministic function to generate pseudo-random configurations
// independently of the call context

static unsigned long seed = 123456789;

static unsigned long pseudo_random ()
{
  unsigned long a = 1664525;
  unsigned long c = 1013904223;
  unsigned long m = 4294967296;

  seed = (a * seed + c) % m;
  seed ^= (seed >> 21);
  seed ^= (seed << 35);
  seed ^= (seed >> 4);
  seed *= 2685821657736338717ULL;
  return seed;
}

void sandPile_draw_alea (void)
{
  for (int i = 0; i < DIM >> 3; i++)
    set_cell (1 + pseudo_random () % (DIM - 2),
              1 + pseudo_random () % (DIM - 2),
              1000 + (pseudo_random () % (4000)));
}

void sandPile_draw_big (void)
{
  const int i = DIM / 2;
  set_cell (i, i, 100000);
}

static void one_spiral (int x, int y, int step, int turns)
{
  int i = x, j = y, t;

  for (t = 1; t <= turns; t++) {
    for (; i < x + t * step; i++)
      set_cell (i, j, 3);
    for (; j < y + t * step + 1; j++)
      set_cell (i, j, 3);
    for (; i > x - t * step - 1; i--)
      set_cell (i, j, 3);
    for (; j > y - t * step - 1; j--)
      set_cell (i, j, 3);
  }
  set_cell (i, j, 4);

  for (int i = -2; i < 3; i++)
    for (int j = -2; j < 3; j++)
      set_cell (i + x, j + y, 3);
}

static void many_spirals (int xdebut, int xfin, int ydebut, int yfin, int step,
                          int turns)
{
  int i, j;
  int size = turns * step + 2;

  for (i = xdebut + size; i < xfin - size; i += 2 * size)
    for (j = ydebut + size; j < yfin - size; j += 2 * size)
      one_spiral (i, j, step, turns);
}

static void spiral (unsigned twists)
{
  many_spirals (1, DIM - 2, 1, DIM - 2, 2, twists);
}

void sandPile_draw_spirals (void)
{
  spiral (DIM / 32);
}

// =========================================================================
// QUADTREE implementation
// =========================================================================

LazyQT lazy_qt = {0};

// Compute the smallest power of 2 >= v
static int next_pow2 (int v)
{
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return v + 1;
}

void qt_init (void)
{
  lazy_qt.ntw = DIM / TILE_W;
  lazy_qt.nth = DIM / TILE_H;

  int max_dim = lazy_qt.ntw > lazy_qt.nth ? lazy_qt.ntw : lazy_qt.nth;
  lazy_qt.n   = next_pow2 (max_dim);

  // depth = log2(n) + 1 : root is level 0, leaves are level depth-1
  lazy_qt.depth = 0;
  for (int v = lazy_qt.n; v > 1; v >>= 1)
    lazy_qt.depth++;
  lazy_qt.depth++; // +1 for the root level

  // total_nodes = sum of 4^k for k = 0..depth-1 = (4^depth - 1) / 3
  lazy_qt.total_nodes = ((1 << (2 * lazy_qt.depth)) - 1) / 3;

  lazy_qt.cur = calloc (lazy_qt.total_nodes, sizeof (unsigned char));
  lazy_qt.nxt = calloc (lazy_qt.total_nodes, sizeof (unsigned char));

  PRINT_DEBUG ('u',
               "QuadTree: %dx%d tiles (padded to %d), depth=%d, "
               "total_nodes=%d (%d bytes)\n",
               lazy_qt.ntw, lazy_qt.nth, lazy_qt.n, lazy_qt.depth,
               lazy_qt.total_nodes, lazy_qt.total_nodes * 2);
}

void qt_destroy (void)
{
  free (lazy_qt.cur);
  free (lazy_qt.nxt);
  lazy_qt.cur = NULL;
  lazy_qt.nxt = NULL;
}

void qt_mark_all_dirty (void)
{
  memset (lazy_qt.cur, QT_DIRTY, lazy_qt.total_nodes);
}

// Mark a single tile dirty in the nxt array and propagate up to root.
// The early exit when a parent is already dirty makes this O(1) amortized.
void qt_mark_one_dirty_nxt (int tx, int ty)
{
  int leaf_level                                   = lazy_qt.depth - 1;
  lazy_qt.nxt [qt_node_index (leaf_level, ty, tx)] = QT_DIRTY;

  int cx = tx, cy = ty;
  for (int l = leaf_level - 1; l >= 0; l--) {
    cx >>= 1;
    cy >>= 1;
    int idx = qt_node_index (l, cy, cx);
    if (lazy_qt.nxt [idx] == QT_DIRTY)
      break; // parent already dirty, ancestors are too
    lazy_qt.nxt [idx] = QT_DIRTY;
  }
}

// Mark a tile AND its 4-connected neighbors dirty in the nxt tree.
// Called when a tile had at least one unstable cell.
void qt_mark_dirty_with_neighbors (int tx, int ty)
{
  qt_mark_one_dirty_nxt (tx, ty);
  if (tx > 0)
    qt_mark_one_dirty_nxt (tx - 1, ty);
  if (tx < lazy_qt.ntw - 1)
    qt_mark_one_dirty_nxt (tx + 1, ty);
  if (ty > 0)
    qt_mark_one_dirty_nxt (tx, ty - 1);
  if (ty < lazy_qt.nth - 1)
    qt_mark_one_dirty_nxt (tx, ty + 1);
}

void qt_swap_and_clear (void)
{
  unsigned char *tmp = lazy_qt.cur;
  lazy_qt.cur        = lazy_qt.nxt;
  lazy_qt.nxt        = tmp;
  memset (lazy_qt.nxt, QT_CLEAN, lazy_qt.total_nodes);
}

// ---- Recursive top-down traversal ----
//
// Walks the cur tree. At each node :
//   - If CLEAN skip entire subtree (the big win)
//   - If DIRTY and at leaf level compute the tile
//   - If DIRTY and internal recurse into 4 children

static int qt_traverse (int level, int ty, int tx, qt_tile_fn tile_func)
{
  int idx = qt_node_index (level, ty, tx);

  if (lazy_qt.cur [idx] == QT_CLEAN)
    return 0; // skip entire subtree

  int leaf_level = lazy_qt.depth - 1;

  if (level == leaf_level) {
    // Leaf node, corresponds to tile (tx, ty)
    if (tx >= lazy_qt.ntw || ty >= lazy_qt.nth)
      return 0; // padding tile (outside actual grid)

    // Convert tile coords to pixel coords, handling borders
    int px    = tx * TILE_W;
    int py    = ty * TILE_H;
    int x0    = (px == 0);
    int y0    = (py == 0);
    int x_end = (px + TILE_W == DIM);
    int y_end = (py + TILE_H == DIM);

    int changed =
        tile_func (px + x0, py + y0, TILE_W - x_end - x0, TILE_H - y_end - y0);

    if (changed)
      qt_mark_dirty_with_neighbors (tx, ty);

    return changed;
  }

  // Internal node, recurse into 4 children
  int change = 0;
  change |= qt_traverse (level + 1, 2 * ty, 2 * tx, tile_func);
  change |= qt_traverse (level + 1, 2 * ty, 2 * tx + 1, tile_func);
  change |= qt_traverse (level + 1, 2 * ty + 1, 2 * tx, tile_func);
  change |= qt_traverse (level + 1, 2 * ty + 1, 2 * tx + 1, tile_func);
  return change;
}

int qt_compute_iteration (qt_tile_fn tile_func)
{
  return qt_traverse (0, 0, 0, tile_func);
}

// =========================================================================
// TILESETS FOR LAZY EVALUATION
// =========================================================================
//

/*
 * ===== Memory layout configuration =====
 * Those were placed to evaluate how memory layout could affect performance
 */
#define TILESET_MEM_PADDING 64 /* Padding between the allocated memory */
#define TILESET_MEM_ALIGN 1    /* Try to align the values of the struct */

#if TILESET_MEM_ALIGN == 1
#define TILESET_MEM_ALIGN_TO(addr, align) (((addr) + align - 1) & ~(align - 1))
#define TILESET_MEM_GET_ALIGN(type) (alignof (type))
#else
#define TILESET_MEM_ALIGN_TO(addr, align) (addr + 0 * align)
#define TILESET_MEM_GET_ALIGN(type) (0)
#endif

/* ===== Global variables ===== */
tileset_t TILESET      = NULL;
char *restrict PER_SET = NULL;

/* ===== Tileset functions ===== */

tileset_t tileset_init (const unsigned tiles_per_row, const unsigned nb_rows)
{
  unsigned tiles_left, sets_per_row, total_nb_sets;
  bitset mask, trunc_mask;

  tiles_left    = BITSET_SIZE_MOD (tiles_per_row);
  sets_per_row  = BITSET_SIZE_DIV (tiles_per_row) + 1 * (tiles_left != 0);
  total_nb_sets = sets_per_row * nb_rows;
  mask          = bitset_at (tiles_left - 1) - 1;
  trunc_mask    = ~(mask * (tiles_left != 0));

  unsigned total_size = sizeof (struct _tileset);
  total_size += TILESET_MEM_PADDING;
  total_size += sizeof (bitset) * total_nb_sets + alignof (bitset);

  /*  Allocate as one big continuous chunk of memory */
  tileset_t tileset = calloc (1, total_size);
  if (tileset == NULL)
    return NULL;

  tileset->trunc_mask    = trunc_mask;
  tileset->sets_per_row  = sets_per_row;
  tileset->tiles_per_row = tiles_per_row;
  tileset->nb_rows       = nb_rows;
  tileset->total_nb_sets = total_nb_sets;

  int bitset_align = TILESET_MEM_GET_ALIGN (bitset);

  uintptr_t addr = ((uintptr_t)(tileset + 1)) + TILESET_MEM_PADDING;
  tileset->sets  = (bitset *)TILESET_MEM_ALIGN_TO (addr, bitset_align);

  return tileset;
}

void tileset_finalize (tileset_t tileset)
{
  if (tileset == NULL)
    return;

  free ((void *)tileset);
}

void tileset_mark_at (tileset_t tileset, int change, const tile t)
{
  unsigned long long tx = t.tx;
  unsigned long long ty = t.ty;
  unsigned long index   = ty * tileset->sets_per_row + (BITSET_SIZE_DIV (tx));
  bitset bit            = bitset_at (BITSET_SIZE_MOD (tx));

  PER_SET [index] += change * ((tileset->sets [index] & bit) == 0);
  tileset->sets [index] |= bit * change;
}

void tileset_mark_at_no_count (tileset_t tileset, int change, const tile t)
{
  unsigned long long tx = t.tx;
  unsigned long long ty = t.ty;
  unsigned long index   = ty * tileset->sets_per_row + (BITSET_SIZE_DIV (tx));
  bitset bit            = bitset_at (BITSET_SIZE_MOD (tx));

  tileset->sets [index] |= bit * change;
}

void tileset_mark_full (tileset_t tileset)
{
  unsigned total = tileset->total_nb_sets;
  for (unsigned i = 0; i < total; i++) {
    bitset set        = (~((bitset)0)) & tileset->trunc_mask;
    tileset->sets [i] = set;
    PER_SET [i]       = __builtin_popcountll (set);
  }
}

void tileset_mark_empty (tileset_t tileset)
{
  unsigned total = tileset->total_nb_sets;
  for (unsigned i = 0; i < total; i++)
    tileset->sets [i] = 0;
}

void tileset_trunc (tileset_t tileset)
{
  unsigned per_row = tileset->sets_per_row;
  for (unsigned i = per_row - 1; i < tileset->total_nb_sets; i += per_row)
    tileset->sets [i] &= tileset->trunc_mask;
}

unsigned long tileset_get_total_tiles (tileset_t tileset)
{
  return tileset->tiles_per_row * tileset->nb_rows;
}

int tileset_merge (tileset_t tile, tileset_t *restrict others,
                   unsigned nb_others)
{
  for (unsigned i = 0; i < nb_others; i++) {
    for (unsigned j = 0; j < tile->total_nb_sets; j++)
      tile->sets [j] |= others [i]->sets [j];
  }
  int change = 0;
  for (unsigned i = 0; i < tile->total_nb_sets; i++) {
    PER_SET [i] = __builtin_popcountll (tile->sets [i]);
    change |= (PER_SET [i] != 0);
  }

  return change;
}

int tileset_merge_omp (tileset_t tile, tileset_t *restrict others,
                       unsigned nb_others)
{

#pragma omp parallel for schedule(static) shared(others)
  for (unsigned j = 0; j < tile->total_nb_sets; j++)
    for (unsigned i = 0; i < nb_others; i++) {
      tile->sets [j] |= others [i]->sets [j];
      others [i]->sets [j] = 0;
    }

  int change = 0;
  for (unsigned i = 0; i < tile->total_nb_sets; i++) {
    PER_SET [i] = __builtin_popcountll (tile->sets [i]);
    change |= PER_SET [i];
  }

  return change;
}
