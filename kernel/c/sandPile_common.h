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

// =========================================================================
// PROJECT ADDITIONS
// =========================================================================

#define SANDPILE_STR(x) #x

#ifdef __GNUC__
#define SANDPILE_UNROLL_LOOP(x) _Pragma (SANDPILE_STR (GCC unroll x))
#elif __clang__
#define SANDPILE_UNROLL_LOOP(x) _Pragma (SANDPILE_STR (unroll x))
#endif

#define SANDPILE_BORDER_SELF (1)
#define SANDPILE_BORDER_UP (1 << 1)
#define SANDPILE_BORDER_DOWN (1 << 2)
#define SANDPILE_BORDER_LEFT (1 << 3)
#define SANDPILE_BORDER_RIGHT (1 << 4)

#define SANDPILE_BORDER_GET_SELF(x) (((x) & SANDPILE_BORDER_SELF) != 0)
#define SANDPILE_BORDER_GET_UP(x) (((x) & SANDPILE_BORDER_UP) != 0)
#define SANDPILE_BORDER_GET_DOWN(x) (((x) & SANDPILE_BORDER_DOWN) != 0)
#define SANDPILE_BORDER_GET_LEFT(x) (((x) & SANDPILE_BORDER_LEFT) != 0)
#define SANDPILE_BORDER_GET_RIGHT(x) (((x) & SANDPILE_BORDER_RIGHT) != 0)

#define SANDPILE_BORDER_SET_SELF(x) (SANDPILE_BORDER_SELF * ((x) != 0))
#define SANDPILE_BORDER_SET_UP(x, set)                                         \
  (SANDPILE_BORDER_UP * ((x) != 0) * ((set) != 0))
#define SANDPILE_BORDER_SET_DOWN(x, set)                                       \
  (SANDPILE_BORDER_DOWN * ((x) != 0) * ((set) != 0))
#define SANDPILE_BORDER_SET_LEFT(x, set)                                       \
  (SANDPILE_BORDER_LEFT * ((x) != 0) * ((set) != 0))
#define SANDPILE_BORDER_SET_RIGHT(x, set)                                      \
  (SANDPILE_BORDER_RIGHT * ((x) != 0) * ((set) != 0))

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

/*
 * =========================================================================
 * BITSET BASE TYPE
 * =========================================================================
 */

#define BITSET_SIZE 64

#if BITSET_SIZE == 8

#define BITSET_POW 3
typedef uint8_t bitset;

#elif BITSET_SIZE == 16

#define BITSET_POW 4
typedef uint16_t bitset;

#elif BITSET_SIZE == 32

#define BITSET_POW 5
typedef uint32_t bitset;

#elif BITSET_SIZE == 64

#define BITSET_POW 6
typedef uint64_t bitset;

#endif

#define BITSET_SIZE_MOD(x) ((x) & (BITSET_SIZE - 1))
#define BITSET_SIZE_DIV(x) ((x) >> BITSET_POW)
#define BITSET_SIZE_MUL(x) ((x) << BITSET_POW)

#define bitset_clz(set) (__builtin_clzll (set) - (64 - BITSET_SIZE))
#define bitset_count(set) (__builtin_popcountll (set))

static inline const bitset bitset_at (const unsigned at)
{
  unsigned index = BITSET_SIZE - at - 1;
  return ((bitset)1) << index;
}

/* Debug function */
static inline void bitset_print (const bitset bitset)
{
  printf ("{");
  for (int i = 0; i < BITSET_SIZE - 1; i++)
    printf ("%c,", bitset & bitset_at (i) ? '1' : '0');
  printf ("%c}\n", bitset & bitset_at (BITSET_SIZE - 1) ? '1' : '0');
}

/*
 * =========================================================================
 * BITSET BASED TILESET FOR LAZY EVALUATION
 * =========================================================================
 */

struct _tileset
{
  bitset *restrict sets;
};

typedef struct _tileset *restrict tileset_t;

#define tileset_at(tileset, index) ((tileset)->sets [index])

typedef struct
{
  unsigned long tx, ty;
} tile;

extern tileset_t TILESET;
extern char *restrict PER_SET;
extern bitset TRUNC_MASK;
extern unsigned SETS_PER_ROW, TILES_PER_ROW, NB_ROWS, TOTAL_NB_SETS;

tileset_t tileset_init (const unsigned tiles_per_row, const unsigned nb_rows);
void tileset_finalize (tileset_t tileset);
void tileset_mark_full (tileset_t tileset);
void tileset_trunc (tileset_t tileset);

static inline void tileset_mark_empty (tileset_t tileset)
{
  unsigned total = TOTAL_NB_SETS;
  for (unsigned i = 0; i < total; i++)
    tileset_at (tileset, i) = 0;
}

static inline unsigned long tileset_get_total_tiles (tileset_t tileset)
{
  return TILES_PER_ROW * NB_ROWS;
}

static inline int tileset_count (tileset_t tileset)
{
  int left = 0;

  for (unsigned i = 0; i < TOTAL_NB_SETS; i++) {
    PER_SET [i] = bitset_count (tileset_at (tileset, i));
    left |= (PER_SET [i] != 0);
  }

  return left;
}

static inline int tileset_merge (tileset_t tile, tileset_t *restrict others,
                                 unsigned nb_others)
{

  for (unsigned i = 0; i < nb_others; i++)
    for (unsigned j = 0; j < TOTAL_NB_SETS; j++)
      tileset_at (tile, j) |= tileset_at (others [i], j);

  return tileset_count (tile);
}

static inline int tileset_merge_omp (tileset_t tile, tileset_t *restrict others,
                                 unsigned nb_others)
{

  int diff;
  #pragma omp parallel 
  #pragma omp single
  #pragma omp taskloop reduction(| : diff) 
  for (unsigned k = 0; k < TOTAL_NB_SETS; k++) {
    tileset_at (tile, k) = 0;

    for (unsigned j = 0; j < nb_others; j++) {
        tileset_at (tile, k) |= tileset_at (others[j], k);
	diff |= tileset_at (tile, k);
    }
  }

  // return tileset_count (tile);
  return diff;
}

static inline void tileset_mark_at (tileset_t tileset, int change, const tile t)
{
  unsigned long long tx = t.tx;
  unsigned long long ty = t.ty;
  unsigned long index   = ty * SETS_PER_ROW + (BITSET_SIZE_DIV (tx));
  bitset bit            = bitset_at (BITSET_SIZE_MOD (tx));

  tileset_at (tileset, index) |= bit * change;
}

#if __AVX2__ == 1

#include <immintrin.h>

inline __m256i rotate_right (__m256i v)
{
  const __m256i scheme = _mm256_setr_epi32 (7, 0, 1, 2, 3, 4, 5, 6);
  return _mm256_permutevar8x32_epi32 (v, scheme);
}

inline __m256i rotate_left (__m256i v)
{
  const __m256i scheme = _mm256_setr_epi32 (1, 2, 3, 4, 5, 6, 7, 0);
  return _mm256_permutevar8x32_epi32 (v, scheme);
}

static inline int tileset_merge_avx (tileset_t tile, tileset_t *restrict others,
                                 unsigned nb_others)
{

  for (unsigned i = 0; i < nb_others; i++)
    SANDPILE_UNROLL_LOOP(4)
    for (unsigned j = 0; j < TOTAL_NB_SETS; j += AVX_VEC_SIZE_DOUBLE) {
      __m256i * tile_at = (__m256i * )(&tileset_at(tile, j));

      __m256i t = _mm256_load_si256(tile_at);
      __m256i other = _mm256_load_si256((__m256i * )(&tileset_at(others[i], j)));

      __m256i res = _mm256_or_si256(t, other);

     _mm256_store_si256(tile_at, res);
    }

  return tileset_count (tile);
}

#endif

#endif // SANDPILE_COMMON_H
