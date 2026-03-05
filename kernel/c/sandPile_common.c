#include "sandPile_common.h"
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

#define TILES_PER_SET 64
#define TILES_PER_SET_MOD(x) (x & (TILES_PER_SET - 1))
#define TILES_PER_SET_DIV(x) (x >> 6)
#define TILES_PER_SET_MUL(x) (x << 6)

typedef uint64_t tileset;

unsigned current_tile_set, next_tile_set;
tile_bitset tile_sets [2];

struct _tile_bitset
{
  unsigned sets_per_row, tile_per_row, nb_rows, last_found, total_nb_sets;
  tileset *restrict values, trunc_mask;
};

static inline const tileset tile_bitset_at (const unsigned index)
{
  return (((tileset)1) << (TILES_PER_SET - index - 1)) *
         (index < TILES_PER_SET);
}

static inline void tile_bitset_print (const tileset set)
{
  printf ("{");
  for (int i = 0; i < TILES_PER_SET - 1; i++)
    printf ("%c, ", set & tile_bitset_at (i) ? '1' : '0');
  printf ("%c}\n", set & tile_bitset_at (TILES_PER_SET - 1) ? '1' : '0');
}

tile_bitset tile_bitset_init (const unsigned nb_tiles_w,
                              const unsigned nb_tiles_h)
{
  tile_bitset bitset;
  bitset = malloc (sizeof (*bitset));
  if (bitset == NULL)
    return NULL;

  unsigned tiles_left = TILES_PER_SET_MOD (nb_tiles_w);
  unsigned sets_per_row =
      TILES_PER_SET_DIV (nb_tiles_w) + 1 * (tiles_left != 0);

  bitset->values = malloc (sizeof (tileset) * nb_tiles_h * sets_per_row);
  if (bitset->values == NULL) {
    free (bitset);
    return NULL;
  }

  bitset->nb_rows       = nb_tiles_h;
  bitset->tile_per_row  = nb_tiles_w;
  bitset->sets_per_row  = sets_per_row;
  bitset->last_found    = 0;
  bitset->total_nb_sets = bitset->sets_per_row * bitset->nb_rows;

  tileset mask = 0;
  if (tiles_left != 0)
    mask = tile_bitset_at (tiles_left - 1) - 1;
  bitset->trunc_mask = ~mask;

  for (int i = 0; i < bitset->nb_rows * bitset->sets_per_row; i++)
    __builtin_prefetch (&bitset->values [i], 1, 3);

  return bitset;
}

const unsigned long long tile_bitset_next_tile (tile_bitset bitset)
{
  for (unsigned i = bitset->last_found; i < bitset->total_nb_sets; i++) {
    if (bitset->values [i] == 0)
      continue;

    unsigned pos = __builtin_clzll (bitset->values [i]);
    bitset->values [i] &= ~tile_bitset_at (pos);

    bitset->last_found = i;
    return bitset->tile_per_row * (i / bitset->sets_per_row) +
           TILES_PER_SET_MUL (i % bitset->sets_per_row) + pos;
  }

  return -1;
}

const unsigned tile_bitset_nb_tiles (const tile_bitset bitset)
{
  unsigned nb = 0;

  for (unsigned i = 0; i < bitset->total_nb_sets; i++)
    nb += __builtin_popcountl (bitset->values [i]);

  return nb;
}

void tile_bitset_mark_at (tile_bitset bitset, int change, unsigned ty,
                          unsigned tx)
{
  unsigned index = ty * bitset->sets_per_row + (TILES_PER_SET_DIV (tx));
  bitset->values [index] |= (tile_bitset_at (TILES_PER_SET_MOD (tx)) * change);
}

void tile_bitset_mark_full (tile_bitset bitset)
{
  unsigned total = bitset->total_nb_sets;
  for (unsigned i = 0; i < total; i++)
    bitset->values [i] = ~((tileset)0);
}

void tile_bitset_mark_empty (tile_bitset bitset)
{
  for (unsigned i = 0; i < bitset->total_nb_sets; i++)
    bitset->values [i] = (tileset)0;
}

void tile_bitset_trunc (tile_bitset bitset)
{
  unsigned per_row = bitset->sets_per_row;
  for (unsigned i = per_row - 1; i < bitset->total_nb_sets; i += per_row)
    bitset->values [i] &= bitset->trunc_mask;
}

void tile_bitset_switch ()
{
  tile_bitset_trunc (tile_sets [next_tile_set]);

  next_tile_set                            = current_tile_set;
  current_tile_set                         = (current_tile_set + 1) & 1;
  tile_sets [current_tile_set]->last_found = 0;
}
