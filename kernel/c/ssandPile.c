#include "sandPile_common.h"

#include <omp.h>
#include <stdbool.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

int gpu_batch_size = 512;

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
///////////////////////////// Synchronous Kernel
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

// Use ALIAS macros to create wrapper functions for common functions
SANDPILE_ALIAS (ssandPile, refresh_img);
SANDPILE_ALIAS (ssandPile, draw_4partout);
SANDPILE_ALIAS (ssandPile, draw_DIM);
SANDPILE_ALIAS (ssandPile, draw_alea);
SANDPILE_ALIAS (ssandPile, draw_big);
SANDPILE_ALIAS (ssandPile, draw_spirals);
SANDPILE_DRAW_ALIAS (ssandPile);

// Debug facilities

void ssandPile_config (char *param)
{
  sandPile_config (param);
}

void ssandPile_debug (int x, int y)
{
  sandPile_debug (x, y);
}

void ssandPile_init (void)
{
  if (config_param != NULL) {
    gpu_batch_size = atoi (config_param);
    PRINT_DEBUG ('u', "Using GPU BATCH_SIZE = %d\n", gpu_batch_size);
  }

  if (TABLE == NULL) {
    const unsigned size = 2 * DIM * DIM * sizeof (TYPE);

    PRINT_DEBUG ('u', "Memory footprint = 1 x %d bytes\n", size);

    TABLE = ezp_alloc (size);
  }

  num_threads = omp_get_max_threads ();
}

void ssandPile_finalize (void)
{
  const unsigned size = 2 * DIM * DIM * sizeof (TYPE);

  ezp_free (TABLE, size);
}

int ssandPile_do_tile_default (int x, int y, int width, int height)
{
  int diff = 0;

  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++) {
      table (out, i, j) = table (in, i, j) % 4;
      table (out, i, j) += table (in, i + 1, j) / 4;
      table (out, i, j) += table (in, i - 1, j) / 4;
      table (out, i, j) += table (in, i, j + 1) / 4;
      table (out, i, j) += table (in, i, j - 1) / 4;
      if (table (out, i, j) != table (in, i, j))
        diff = 1;
    }

  return diff;
}

/* Omptimized version of the ssandpile_do_tile function
  - Compute a local offset from the current cell pointer each iteration to
  prevent pointer arithmetic operation
  - use unroll pragmas to optimize the loop execution
  - store each memory access in different variables to allow the compiler to
  best optimize the code
*/
int ssandPile_do_tile_opt (int x, int y, int width, int height)
{
  unsigned int offset = DIM - width;
  int diff            = 0;

  TYPE *restrict in_cell  = table_cell (TABLE, in, y, x);
  TYPE *restrict out_cell = table_cell (TABLE, out, y, x);

  for (int i = 0; i < height; i++) {
    SANDPILE_UNROLL_LOOP (4)
    for (int j = 0; j < width; j++) {
      unsigned int b   = (*(in_cell - 1) >> 2);
      unsigned int old = (*in_cell) & 3;
      unsigned int a   = (*(in_cell + 1) >> 2);
      unsigned int c   = (*(in_cell - DIM) >> 2);
      unsigned int d   = (*(in_cell + DIM) >> 2);

      unsigned int new = a + b + c + d + old;

      *out_cell = new;
      diff |= (new != old);

      in_cell++;
      out_cell++;
    }

    in_cell += offset;
    out_cell += offset;
  }

  return diff;
}

int ssandPile_do_tile_opt_border (int x, int y, int width, int height)
{
  unsigned int offset = DIM - width;
  int diff            = 0;

  TYPE *restrict in_cell  = table_cell (TABLE, in, y, x);
  TYPE *restrict out_cell = table_cell (TABLE, out, y, x);

  for (int i = 0; i < height; i++) {
    int is_border_up   = (i == 0);
    int is_border_down = (i == (height - 1));

    for (int j = 0; j < width; j++) {
      int is_border_left  = (j == 0);
      int is_border_right = (j == (width - 1));

      unsigned int b   = (*(in_cell - 1) >> 2);
      unsigned int old = (*in_cell) & 3;
      unsigned int a   = (*(in_cell + 1) >> 2);
      unsigned int c   = (*(in_cell - DIM) >> 2);
      unsigned int d   = (*(in_cell + DIM) >> 2);

      unsigned int new  = a + b + c + d + old;
      unsigned loc_diff = SANDPILE_BORDER_SET_SELF (new != old);

      int up_flag    = SANDPILE_BORDER_SET_UP (loc_diff, is_border_up);
      int down_flag  = SANDPILE_BORDER_SET_DOWN (loc_diff, is_border_down);
      int left_flag  = SANDPILE_BORDER_SET_LEFT (loc_diff, is_border_left);
      int right_flag = SANDPILE_BORDER_SET_RIGHT (loc_diff, is_border_right);

      diff |= loc_diff | up_flag | down_flag | left_flag | right_flag;

      *out_cell = new;
      in_cell++;
      out_cell++;
    }

    in_cell += offset;
    out_cell += offset;
  }

  return diff;
}

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned ssandPile_compute_seq (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {
    int change = do_tile (1, 1, DIM - 2, DIM - 2);
    swap_tables ();
    if (change == 0)
      return it;
  }
  return 0;
}

unsigned ssandPile_compute_tiled (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {
    int change = 0;

    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        change |= do_tile (x + (x == 0), y + (y == 0),
                           TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                           TILE_H - ((y + TILE_H == DIM) + (y == 0)));
    swap_tables ();
    if (change == 0)
      return it;
  }

  return 0;
}

unsigned ssandPile_compute_omp_taskloop (unsigned nb_iter)
{

  unsigned it;
#pragma omp parallel
#pragma omp single
  for (it = 1; it <= nb_iter; it++) {
    int change = 0;

#pragma omp taskloop reduction(| : change) collapse(2) num_tasks(num_threads)  \
    shared(TABLE)
    for (int y = 0; y < DIM; y += TILE_H) {
      for (int x = 0; x < DIM; x += TILE_W) {
        int y_0   = (y == 0);
        int y_end = (y + TILE_H == DIM);
        int x_0   = (x == 0);
        int x_end = (x + TILE_W == DIM);

        change |= do_tile (x + x_0, y + y_0, TILE_W - x_end - x_0,
                           TILE_H - y_end - y_0);
      }
    }

    swap_tables ();
    if (change == 0)
      break;
  }

  if (it <= nb_iter)
    return it;
  return 0;
}

// OpenMP parallelized version using tiled decomposition
// Usage: OMP_SCHEDULE=dynamic,4 ./run -k ssandPile -v omp_tiled -s <SIZE>
unsigned ssandPile_compute_omp_tiled (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {
    int change = 0;

// Parallelize the nested tile loops
// collapse(2) allows OpenMP to create a single iteration space from both loops,
// improving load balancing especially when the number of tiles is small
#pragma omp parallel for collapse(2) schedule(runtime) reduction(| : change)
    for (int y = 0; y < DIM; y += TILE_H) {
      for (int x = 0; x < DIM; x += TILE_W) {
        int y_0   = (y == 0);
        int y_end = (y + TILE_H == DIM);
        int x_0   = (x == 0);
        int x_end = (x + TILE_W == DIM);

        change |= do_tile (x + x_0, y + y_0, TILE_W - x_end - x_0,
                           TILE_H - y_end - y_0);
      }
    }

    swap_tables ();
    if (change == 0)
      return it;
  }

  return 0;
}

/*
 * =============================================
 * LAZY EVALUATION
 * =============================================
 */

void ssandPile_init_lazy (void)
{
  ssandPile_init ();

  if (TILESET != NULL)
    return;
  TILESET = tileset_init (NB_TILES_X, NB_TILES_Y);

  if (PER_SET != NULL)
    return;
  PER_SET = ezp_alloc (sizeof (char) * TOTAL_NB_SETS);
  memset (PER_SET, 0, TOTAL_NB_SETS);

  tileset_mark_full (TILESET);
}

void ssandPile_init_lazy_border (void)
{
  ssandPile_init_lazy ();
}

void ssandPile_init_omp_lazy (void)
{
  ssandPile_init_lazy ();

  LAZY_NB_TILESET = num_threads;
  tilesets        = malloc (sizeof (tileset_t *) * LAZY_NB_TILESET);
  for (unsigned i = 0; i < LAZY_NB_TILESET; i++)
    tilesets [i] = tileset_init (NB_TILES_X, NB_TILES_Y);
}

void ssandPile_init_omp_lazy_border (void)
{
  ssandPile_init_omp_lazy ();
}

void ssandPile_finalize_lazy (void)
{
  ssandPile_finalize ();
  ezp_free (PER_SET, sizeof (char) * TOTAL_NB_SETS);
  tileset_finalize (TILESET);
}

void ssandPile_finalize_lazy_border (void)
{
  ssandPile_finalize_lazy ();
}

void ssandPile_finalize_omp_lazy (void)
{
  ssandPile_finalize_lazy ();

  for (unsigned i = 0; i < LAZY_NB_TILESET; i++)
    tileset_finalize (tilesets [i]);

  free ((void *)tilesets);
}

void ssandPile_finalize_omp_lazy_border (void)
{
  ssandPile_finalize_omp_lazy ();
}

unsigned ssandPile_compute_lazy (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    bitset sets [TOTAL_NB_SETS];
    for (unsigned i = 0; i < TOTAL_NB_SETS; i++) {
      sets [i]                = tileset_at (TILESET, i);
      tileset_at (TILESET, i) = 0;
    }

    for (unsigned i = 0; i < TOTAL_NB_SETS; i++) {
      unsigned char in_set = PER_SET [i];

      unsigned tile_y = i / SETS_PER_ROW;
      unsigned set_x  = i - tile_y * SETS_PER_ROW;

      for (unsigned char k = 0; k < in_set; k++) {
        unsigned pos = bitset_clz (sets [i]);
        sets [i] -= bitset_at (pos);

        tile t;
        t.tx = BITSET_SIZE_MUL (set_x) + pos;
        t.ty = tile_y;

        unsigned long x, y;
        x = t.tx * TILE_W;
        y = t.ty * TILE_H;

        int y_0   = (y == 0);
        int y_end = (y + TILE_H == DIM);
        int x_0   = (x == 0);
        int x_end = (x + TILE_W == DIM);

        int loc_change = do_tile (x + x_0, y + y_0, TILE_W - x_end - x_0,
                                  TILE_H - y_end - y_0);
        loc_change &= 1;

        tile self  = t;
        tile up    = {.tx = t.tx, .ty = t.ty - 1 + y_0};
        tile down  = {.tx = t.tx, .ty = t.ty + 1 - y_end};
        tile left  = {.tx = t.tx - 1 + x_0, .ty = t.ty};
        tile right = {.tx = t.tx + 1 - x_end, .ty = t.ty};

        tileset_mark_at (TILESET, loc_change, self);
        tileset_mark_at (TILESET, loc_change, up);
        tileset_mark_at (TILESET, loc_change, down);
        tileset_mark_at (TILESET, loc_change, left);
        tileset_mark_at (TILESET, loc_change, right);
      }
    }

    swap_tables ();

    int change = tileset_count (TILESET);
    if (change == 0)
      return it;
  }

  return 0;
}

unsigned ssandPile_compute_lazy_border (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {
    bitset sets [TOTAL_NB_SETS];
    for (unsigned i = 0; i < TOTAL_NB_SETS; i++) {
      sets [i]                = tileset_at (TILESET, i);
      tileset_at (TILESET, i) = 0;
    }

    for (unsigned i = 0; i < TOTAL_NB_SETS; i++) {
      unsigned char in_set = PER_SET [i];
      unsigned tile_y      = i / SETS_PER_ROW;
      unsigned set_x       = i - tile_y * SETS_PER_ROW;

      for (unsigned char k = 0; k < in_set; k++) {
        unsigned pos = bitset_clz (sets [i]);
        sets [i] -= bitset_at (pos);

        tile t;
        t.tx = BITSET_SIZE_MUL (set_x) + pos;
        t.ty = tile_y;

        unsigned long x, y;
        x = t.tx * TILE_W;
        y = t.ty * TILE_H;

        int y_0   = (y == 0);
        int y_end = (y + TILE_H == DIM);
        int x_0   = (x == 0);
        int x_end = (x + TILE_W == DIM);

        int loc_change = do_tile (x + x_0, y + y_0, TILE_W - x_end - x_0,
                                  TILE_H - y_end - y_0);

        tile self  = t;
        tile up    = {.tx = t.tx, .ty = t.ty - 1 + y_0};
        tile down  = {.tx = t.tx, .ty = t.ty + 1 - y_end};
        tile left  = {.tx = t.tx - 1 + x_0, .ty = t.ty};
        tile right = {.tx = t.tx + 1 - x_end, .ty = t.ty};

        int mark_self  = SANDPILE_BORDER_GET_SELF (loc_change);
        int mark_up    = SANDPILE_BORDER_GET_UP (loc_change);
        int mark_down  = SANDPILE_BORDER_GET_DOWN (loc_change);
        int mark_left  = SANDPILE_BORDER_GET_LEFT (loc_change);
        int mark_right = SANDPILE_BORDER_GET_RIGHT (loc_change);

        tileset_mark_at (TILESET, mark_self, self);
        tileset_mark_at (TILESET, mark_up, up);
        tileset_mark_at (TILESET, mark_down, down);
        tileset_mark_at (TILESET, mark_left, left);
        tileset_mark_at (TILESET, mark_right, right);
      }
    }

    swap_tables ();

    int change = tileset_count (TILESET);
    if (change == 0)
      return it;
  }

  return 0;
}

unsigned ssandPile_compute_omp_lazy (unsigned nb_iter)
{
  unsigned it;
  for (it = 1; it <= nb_iter; it++) {
    int change = 0;

    bitset *restrict sets = &tileset_at (TILESET, 0);

#pragma omp parallel shared(TABLE)
    {
      tileset_t curr = tilesets [omp_get_thread_num ()];
      tileset_mark_empty (curr);
#pragma omp for schedule(static)
      for (unsigned i = 0; i < TOTAL_NB_SETS; i++) {
        unsigned char in_set = PER_SET [i];
        unsigned tile_y      = i / SETS_PER_ROW;
        unsigned set_x       = i - tile_y * SETS_PER_ROW;

        for (unsigned char k = 0; k < in_set; k++) {
          unsigned pos = bitset_clz (sets [i]);
          sets [i] -= bitset_at (pos);

          tile t;
          t.tx = BITSET_SIZE_MUL (set_x) + pos;
          t.ty = tile_y;

          unsigned long x, y;
          x = t.tx * TILE_W;
          y = t.ty * TILE_H;

          int y_0   = (y == 0);
          int y_end = (y + TILE_H == DIM);
          int x_0   = (x == 0);
          int x_end = (x + TILE_W == DIM);

          int loc_change = do_tile (x + x_0, y + y_0, TILE_W - x_end - x_0,
                                    TILE_H - y_end - y_0);

          tile self  = t;
          tile up    = {.tx = t.tx, .ty = t.ty - 1 + y_0};
          tile down  = {.tx = t.tx, .ty = t.ty + 1 - y_end};
          tile left  = {.tx = t.tx - 1 + x_0, .ty = t.ty};
          tile right = {.tx = t.tx + 1 - x_end, .ty = t.ty};

          tileset_mark_at (curr, loc_change, self);
          tileset_mark_at (curr, loc_change, up);
          tileset_mark_at (curr, loc_change, down);
          tileset_mark_at (curr, loc_change, left);
          tileset_mark_at (curr, loc_change, right);
        }
      }
    }

    swap_tables ();

    change = tileset_merge (TILESET, tilesets, LAZY_NB_TILESET);
    if (change == 0)
      return it;
  }

  return 0;
}

unsigned ssandPile_compute_omp_lazy_border (unsigned nb_iter)
{
  unsigned it;
  for (it = 1; it <= nb_iter; it++) {
    int change = 0;

    bitset *restrict sets = &tileset_at (TILESET, 0);

#pragma omp parallel
    {
      tileset_t curr = tilesets [omp_get_thread_num ()];
      tileset_mark_empty (curr);
#pragma omp for schedule(static)
      for (unsigned i = 0; i < TOTAL_NB_SETS; i++) {
        unsigned tile_y = i / SETS_PER_ROW;
        unsigned set_x  = i - tile_y * SETS_PER_ROW;
        bitset set      = sets [i];

        while (set != 0) {
          unsigned pos = bitset_clz (set);
          set -= bitset_at (pos);

          tile t;
          t.tx = BITSET_SIZE_MUL (set_x) + pos;
          t.ty = tile_y;

          unsigned long x, y;
          x = t.tx * TILE_W;
          y = t.ty * TILE_H;

          int loc_change;

          int y_0    = (y == 0);
          int x_0    = (x == 0);
          int y_end  = (y + TILE_H == DIM);
          int x_end  = (x + TILE_W == DIM);
          loc_change = do_tile (x + x_0, y + y_0, TILE_W - x_end - x_0,
                                TILE_H - y_end - y_0);

          tile self  = t;
          tile up    = {.tx = t.tx, .ty = t.ty - 1 + y_0};
          tile down  = {.tx = t.tx, .ty = t.ty + 1 - y_end};
          tile left  = {.tx = t.tx - 1 + x_0, .ty = t.ty};
          tile right = {.tx = t.tx + 1 - x_end, .ty = t.ty};

          int mark_self  = SANDPILE_BORDER_GET_SELF (loc_change);
          int mark_up    = SANDPILE_BORDER_GET_UP (loc_change);
          int mark_down  = SANDPILE_BORDER_GET_DOWN (loc_change);
          int mark_left  = SANDPILE_BORDER_GET_LEFT (loc_change);
          int mark_right = SANDPILE_BORDER_GET_RIGHT (loc_change);

          tileset_mark_at (curr, mark_self, self);
          tileset_mark_at (curr, mark_up, up);
          tileset_mark_at (curr, mark_down, down);
          tileset_mark_at (curr, mark_left, left);
          tileset_mark_at (curr, mark_right, right);
        }
      }
    }

    swap_tables ();

    change = TILESET_MERGE (TILESET, tilesets, num_threads);
    if (change == 0)
      break;
  }

  if (it <= nb_iter)
    return it;
  return 0;
}

#if __AVX2__ == 1

#include <immintrin.h>

void ssandpile_tile_check_opt_avx (void)
{
  easypap_vec_check (AVX_VEC_SIZE_INT, DIR_HORIZONTAL);
}

int ssandPile_do_tile_opt_avx (int x, int y, int width, int height)
{
  const __m256i m256_3 = _mm256_set1_epi32 (3);
  const int offset     = -(height * DIM) + AVX_VEC_SIZE_INT;
  int diff             = 0;

  int left_tile  = x == 1;
  int right_tile = x + width == DIM - 1;
  width += left_tile + right_tile;
  x -= left_tile;

  TYPE *in_cell  = table_cell (TABLE, in, y, x);
  TYPE *out_cell = table_cell (TABLE, out, y, x);

  for (int j = 0; j < width; j += AVX_VEC_SIZE_INT) {
    int is_border_left  = (j == 0);
    int is_border_right = (j == (width - AVX_VEC_SIZE_INT));
    int do_left         = -1 * !(is_border_left & left_tile);
    int do_right        = -1 * !(is_border_right & right_tile);
    __m256i mask = _mm256_set_epi32 (do_right, -1, -1, -1, -1, -1, -1, do_left);

    register __m256i up_vec = _mm256_load_si256 ((__m256i *)(in_cell - DIM));
    register __m256i center_vec = _mm256_load_si256 ((__m256i *)(in_cell));
    register __m256i down_vec;

    SANDPILE_UNROLL_LOOP (4)
    for (int i = 0; i < height; i++) {
      int is_border_up   = (i == 0);
      int is_border_down = (i == (height - 1));
      down_vec           = _mm256_load_si256 ((__m256i *)(in_cell + DIM));

      __m256i res = _mm256_and_si256 (center_vec, m256_3);

      __m256i left_vec  = _mm256_loadu_si256 ((__m256i *)(in_cell - 1));
      __m256i right_vec = _mm256_loadu_si256 ((__m256i *)(in_cell + 1));

      __m256i tops = _mm256_add_epi32 (_mm256_srli_epi32 (up_vec, 2),
                                       _mm256_srli_epi32 (down_vec, 2));

      __m256i sides = _mm256_add_epi32 (_mm256_srli_epi32 (left_vec, 2),
                                        _mm256_srli_epi32 (right_vec, 2));

      sides = _mm256_add_epi32 (tops, sides);
      res   = _mm256_add_epi32 (res, sides);

      res = _mm256_and_si256 (res, mask);
      _mm256_store_si256 ((__m256i *)(out_cell), res);

      __m256 cast              = (__m256)_mm256_cmpeq_epi32 (res, center_vec);
      unsigned char change_vec = ~(unsigned char)_mm256_movemask_ps (cast);

      char change_right = (change_vec >> (AVX_VEC_SIZE_INT - 1)) & 1;
      char change_left  = change_vec & 1;

      int self_flag = SANDPILE_BORDER_SET_SELF (change_vec != 0);
      int up_flag   = SANDPILE_BORDER_SET_UP (self_flag, is_border_up);
      int down_flag = SANDPILE_BORDER_SET_DOWN (self_flag, is_border_down);
      int left_flag = SANDPILE_BORDER_SET_LEFT (change_left, is_border_left);
      int right_flag =
          SANDPILE_BORDER_SET_RIGHT (change_right, is_border_right);

      diff |= self_flag | up_flag | down_flag | left_flag | right_flag;

      up_vec     = center_vec;
      center_vec = down_vec;

      in_cell += DIM;
      out_cell += DIM;
    }
    in_cell += offset;
    out_cell += offset;
  }

  return diff;
}

int ssandPile_do_tile_opt_avx_default (int x, int y, int width, int height)
{
  const __m256i m256_3 = _mm256_set1_epi32 (3);
  const int offset     = -(height * DIM) + AVX_VEC_SIZE_INT;
  int diff             = 0;

  if (width != TILE_W)
    return ssandPile_do_tile_opt_border (x, y, width, height);

  TYPE *in_cell  = table_cell (TABLE, in, y, x);
  TYPE *out_cell = table_cell (TABLE, out, y, x);

  for (int j = 0; j < width; j += AVX_VEC_SIZE_INT) {
    int is_border_left  = (j == 0);
    int is_border_right = (j == (width - AVX_VEC_SIZE_INT));

    __m256i up_vec     = _mm256_load_si256 ((__m256i *)(in_cell - DIM));
    __m256i center_vec = _mm256_load_si256 ((__m256i *)(in_cell));
    __m256i down_vec;

    SANDPILE_UNROLL_LOOP (4)
    for (int i = 0; i < height; i++) {
      __builtin_prefetch (in_cell + 2 * DIM - 1, 0, 3);
      __builtin_prefetch (in_cell + 2 * DIM, 0, 3);
      __builtin_prefetch (in_cell + 2 * DIM + 1, 0, 3);

      int is_border_up   = (i == 0);
      int is_border_down = (i == (height - 1));

      down_vec = _mm256_load_si256 ((__m256i *)(in_cell + DIM));

      __m256i res = _mm256_and_si256 (center_vec, m256_3);

      __m256i tops = _mm256_add_epi32 (_mm256_srli_epi32 (up_vec, 2),
                                       _mm256_srli_epi32 (down_vec, 2));

      __m256i left_vec  = _mm256_loadu_si256 ((__m256i *)(in_cell - 1));
      __m256i right_vec = _mm256_loadu_si256 ((__m256i *)(in_cell + 1));

      __m256i sides = _mm256_add_epi32 (_mm256_srli_epi32 (left_vec, 2),
                                        _mm256_srli_epi32 (right_vec, 2));

      sides = _mm256_add_epi32 (tops, sides);
      res   = _mm256_add_epi32 (res, sides);

      _mm256_store_si256 ((__m256i *)out_cell, res);

      __m256 cast              = (__m256)_mm256_cmpeq_epi32 (res, center_vec);
      unsigned char change_vec = ~(unsigned char)_mm256_movemask_ps (cast);

      char change_left  = change_vec & 1;
      char change_right = (change_vec >> (AVX_VEC_SIZE_INT - 1)) & 1;

      int self_flag = SANDPILE_BORDER_SET_SELF (change_vec != 0);
      int up_flag   = SANDPILE_BORDER_SET_UP (self_flag, is_border_up);
      int down_flag = SANDPILE_BORDER_SET_DOWN (self_flag, is_border_down);
      int left_flag = SANDPILE_BORDER_SET_LEFT (change_left, is_border_left);
      int right_flag =
          SANDPILE_BORDER_SET_RIGHT (change_right, is_border_right);

      diff |= self_flag | up_flag | down_flag | left_flag | right_flag;

      in_cell += DIM;
      out_cell += DIM;

      up_vec     = center_vec;
      center_vec = down_vec;
    }

    in_cell += offset;
    out_cell += offset;
  }

  return diff;
}

int ssandPile_do_tile_avx (int x, int y, int width, int height)
{
  int left_tile  = x == 1;
  int right_tile = x + width == DIM - 1;
  width += left_tile + right_tile;
  x -= left_tile;

  int diff = 0;

  TYPE *restrict in_cell            = table_cell (TABLE, in, y, x);
  TYPE *restrict out_cell           = table_cell (TABLE, out, y, x);
  const register __m256i MASK_ZERO  = _mm256_set1_epi32 (0);
  const register __m256i MASK_MOD_4 = _mm256_set1_epi32 (3);

  for (int i = 0; i < height; i++) {
    int is_border_up   = (i == 0);
    int is_border_down = (i == (height - 1));

    TYPE *restrict up_row     = in_cell - DIM;
    TYPE *restrict center_row = in_cell;
    TYPE *restrict down_row   = in_cell + DIM;

    register __m256i center_vec;
    center_vec = _mm256_load_si256 ((__m256i *)center_row);

    for (int j = 0; j < width; j += AVX_VEC_SIZE_INT) {
      int is_border_left  = (j == 0);
      int is_border_right = (j == (width - AVX_VEC_SIZE_INT));
      int do_left         = !(is_border_left & left_tile);
      int do_right        = !(is_border_right & right_tile);
      __m256i mask = _mm256_set_epi32 (-1 * do_right, -1, -1, -1, -1, -1, -1,
                                       -1 * do_left);

      __m256i up_vec   = _mm256_load_si256 ((__m256i *)(up_row + j));
      __m256i down_vec = _mm256_load_si256 ((__m256i *)(down_row + j));

      __m256i left_added_vec =
          _mm256_loadu_si256 ((__m256i *)(center_row + j + 1));
      __m256i right_added_vec =
          _mm256_loadu_si256 ((__m256i *)(center_row + j - 1));

      __m256i res = _mm256_add_epi32 (_mm256_srli_epi32 (left_added_vec, 2),
                                      _mm256_srli_epi32 (right_added_vec, 2));
      res = _mm256_add_epi32 (res, _mm256_and_si256 (center_vec, MASK_MOD_4));

      __m256i tops = _mm256_add_epi32 (_mm256_srli_epi32 (down_vec, 2),
                                       _mm256_srli_epi32 (up_vec, 2));
      res          = _mm256_add_epi32 (res, tops);

      res = _mm256_blendv_epi8 (MASK_ZERO, res, mask);
      _mm256_store_si256 ((__m256i *)(out_cell + j), res);

      __m256 cast              = (__m256)_mm256_cmpeq_epi32 (res, center_vec);
      unsigned char change_vec = ~(unsigned char)_mm256_movemask_ps (cast);

      char change_left  = change_vec & 1;
      char change_right = (change_vec >> (AVX_VEC_SIZE_INT - 1)) & 1;

      int self_flag = SANDPILE_BORDER_SET_SELF (change_vec != 0);
      int up_flag   = SANDPILE_BORDER_SET_UP (self_flag, is_border_up);
      int down_flag = SANDPILE_BORDER_SET_DOWN (self_flag, is_border_down);
      int left_flag = SANDPILE_BORDER_SET_LEFT (change_left, is_border_left);
      int right_flag =
          SANDPILE_BORDER_SET_RIGHT (change_right, is_border_right);

      diff |= self_flag | up_flag | down_flag | left_flag | right_flag;
    }

    in_cell += DIM;
    out_cell += DIM;
  }

  return diff;
}

int ssandPile_do_tile_avx_default (int x, int y, int width, int height)
{
  if (width != TILE_W)
    return ssandPile_do_tile_opt (x, y, width, height);

  const int offset = DIM - width;
  int diff         = 0;

  TYPE *in_cell  = table_cell (TABLE, in, y, x);
  TYPE *out_cell = table_cell (TABLE, out, y, x);

  const __m256i mask_mod_4 = _mm256_set1_epi32 (3);

  for (int i = 0; i < height; i++) {
    __m256i left_vec, center_vec, right_vec;

    center_vec = _mm256_set_epi32 (*(in_cell - 1), 0, 0, 0, 0, 0, 0, 0);
    right_vec  = _mm256_load_si256 ((__m256i *)in_cell);

    for (int j = 0; j < width; j += AVX_VEC_SIZE_INT) {
      left_vec   = center_vec;
      center_vec = right_vec;
      right_vec  = _mm256_load_si256 ((__m256i *)(in_cell + AVX_VEC_SIZE_INT));

      __m256i res = _mm256_and_si256 (center_vec, mask_mod_4);

      __m256i right_div_vec = _mm256_blend_epi32 (
          rotate_left (center_vec), rotate_left (right_vec), 1 << 7);

      res = _mm256_add_epi32 (res, _mm256_srli_epi32 (right_div_vec, 2));

      __m256i left_div_vec = _mm256_blend_epi32 (rotate_right (center_vec),
                                                 rotate_right (left_vec), 1);

      res = _mm256_add_epi32 (res, _mm256_srli_epi32 (left_div_vec, 2));

      __m256i up_vec = _mm256_load_si256 ((__m256i *)(in_cell - DIM));
      res            = _mm256_add_epi32 (res, _mm256_srli_epi32 (up_vec, 2));

      __m256i down_vec = _mm256_load_si256 ((__m256i *)(in_cell + DIM));
      res = _mm256_add_epi32 (res, _mm256_srli_epi32 (down_vec, 2));

      _mm256_store_si256 ((__m256i *)out_cell, res);

      __m256 cast              = (__m256)_mm256_cmpeq_epi32 (res, center_vec);
      unsigned char change_vec = ~(unsigned char)_mm256_movemask_ps (cast);
      diff |= change_vec != 0;

      in_cell += AVX_VEC_SIZE_INT;
      out_cell += AVX_VEC_SIZE_INT;
    }

    in_cell += offset;
    out_cell += offset;
  }

  return diff;
}

#endif

#ifdef ENABLE_OPENCL
// OpenCL basic

static cl_mem ocl_changed_buffer = NULL;

void ssandPile_init_ocl (void)
{
  // Allocate TABLE here
  ssandPile_init ();

  cl_int err;
  ocl_changed_buffer =
      clCreateBuffer (context, CL_MEM_READ_WRITE, sizeof (int), NULL, &err);
  check (err, "Failed to create changed buffer");
}

unsigned ssandPile_compute_ocl (unsigned nb_iter)
{
  size_t global [2] = {GPU_SIZE_X, GPU_SIZE_Y};
  size_t local [2]  = {TILE_W, TILE_H};
  cl_int err;

  const unsigned BATCH_SIZE = 256;

  const int zero = 0;

  err = clSetKernelArg (ocl_compute_kernel (0), 2, sizeof (cl_mem),
                        &ocl_changed_buffer);
  check (err, "Failed to set kernel arg 2");

  unsigned total_it = 0;
  monitoring_start (easypap_gpu_lane (0));

  for (unsigned it = 1; it <= nb_iter; it += BATCH_SIZE) {

    err = clEnqueueFillBuffer (ocl_queue (0), ocl_changed_buffer, &zero,
                               sizeof (int), 0, sizeof (int), 0, NULL, NULL);
    check (err, "Failed to fill changed buffer");

    unsigned max_k =
        (it + BATCH_SIZE - 1 <= nb_iter) ? BATCH_SIZE : (nb_iter - it + 1);

    for (unsigned k = 0; k < max_k; k++) {
      total_it++;
      err = clSetKernelArg (ocl_compute_kernel (0), 0, sizeof (cl_mem),
                            &ocl_cur_buffer (0));
      err |= clSetKernelArg (ocl_compute_kernel (0), 1, sizeof (cl_mem),
                             &ocl_next_buffer (0));
      check (err, "Failed to set kernel args 0-1");

      err = clEnqueueNDRangeKernel (ocl_queue (0), ocl_compute_kernel (0), 2,
                                    NULL, global, local, 0, NULL, NULL);
      check (err, "Kernel launch failed");

      cl_mem tmp          = ocl_cur_buffer (0);
      ocl_cur_buffer (0)  = ocl_next_buffer (0);
      ocl_next_buffer (0) = tmp;
    }

    int changed;
    cl_event read_evt;
    err = clEnqueueReadBuffer (ocl_queue (0), ocl_changed_buffer, CL_FALSE, 0,
                               sizeof (int), &changed, 0, NULL, &read_evt);
    check (err, "Failed to enqueue changed read");

    clFlush (ocl_queue (0));
    clWaitForEvents (1, &read_evt);
    clReleaseEvent (read_evt);

    if (changed == 0) {
      clFinish (ocl_queue (0));
      monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (0));
      return total_it;
    }
  }

  clFinish (ocl_queue (0));
  monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (0));
  return 0;
}

void ssandPile_refresh_img_ocl (void)
{
  cl_int err;

  err =
      clEnqueueReadBuffer (ocl_queue (0), ocl_cur_buffer (0), CL_TRUE, 0,
                           sizeof (unsigned) * DIM * DIM, TABLE, 0, NULL, NULL);
  check (err, "Failed to read buffer from GPU");

  ssandPile_refresh_img ();
}

/*
 * Variant : multi_m4_loop
 * Strategy : tiling + 4 iterations in LDS using dynamic loops
 * Specificity: flexible workgroup size, but has higher control overhead
 * Usage : ./run -k ssandPile -g -v ocl_multi_m4_loop -s 1024 -i 1000
 */
void ssandPile_init_ocl_multi_m4_loop (void)
{
  ssandPile_init_ocl ();
}

unsigned ssandPile_compute_ocl_multi_m4_loop (unsigned nb_iter)
{
  size_t global [2] = {GPU_SIZE_X, GPU_SIZE_Y};
  size_t local [2]  = {TILE_W, TILE_H};
  cl_int err;

  const int zero = 0;

  err = clSetKernelArg (ocl_compute_kernel (0), 2, sizeof (cl_mem),
                        &ocl_changed_buffer);
  check (err, "Failed to set kernel arg 2");

  unsigned total_it = 0;
  monitoring_start (easypap_gpu_lane (0));

  for (unsigned it = 1; it <= nb_iter; it += gpu_batch_size * 4) {
    err = clEnqueueFillBuffer (ocl_queue (0), ocl_changed_buffer, &zero,
                               sizeof (int), 0, sizeof (int), 0, NULL, NULL);

    unsigned remaining = nb_iter - it + 1;
    unsigned max_k     = (remaining + 3) / 4;
    if (max_k > gpu_batch_size)
      max_k = gpu_batch_size;
    if (max_k == 0)
      break;

    // Set args once before the batch inner loop
    err = clSetKernelArg (ocl_compute_kernel (0), 0, sizeof (cl_mem),
                          &ocl_cur_buffer (0));
    err |= clSetKernelArg (ocl_compute_kernel (0), 1, sizeof (cl_mem),
                           &ocl_next_buffer (0));
    check (err, "Failed to set kernel args 0-1");

    for (unsigned k = 0; k < max_k; k++) {
      total_it += 4; // Kernel does 4 iterations internally

      err = clEnqueueNDRangeKernel (ocl_queue (0), ocl_compute_kernel (0), 2,
                                    NULL, global, local, 0, NULL, NULL);

      // Swap and re-bind for next iteration
      cl_mem tmp          = ocl_cur_buffer (0);
      ocl_cur_buffer (0)  = ocl_next_buffer (0);
      ocl_next_buffer (0) = tmp;

      err = clSetKernelArg (ocl_compute_kernel (0), 0, sizeof (cl_mem),
                            &ocl_cur_buffer (0));
      err |= clSetKernelArg (ocl_compute_kernel (0), 1, sizeof (cl_mem),
                             &ocl_next_buffer (0));
    }

    int changed;
    // Read back the changed flag to detect stability
    err = clEnqueueReadBuffer (ocl_queue (0), ocl_changed_buffer, CL_TRUE, 0,
                               sizeof (int), &changed, 0, NULL, NULL);
    check (err, "Failed to read changed flag");

    if (changed == 0) {
      clFinish (ocl_queue (0));
      monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (0));
      return total_it;
    }
  }

  clFinish (ocl_queue (0));
  monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (0));

  return 0;
}

void ssandPile_refresh_img_ocl_multi_m4_loop (void)
{
  ssandPile_refresh_img_ocl ();
}

/*
 * Variant : multi_m5_static (best performance)
 * Strategy : margin 5 with 5 iterations per launch and static LDS mapping
 * Usage : ./run -k ssandPile -g -v ocl_multi_m5_static -s 4096 -tw 32 -th 16
 */
void ssandPile_init_ocl_multi_m5_static (void)
{
  ssandPile_init_ocl ();
}

unsigned ssandPile_compute_ocl_multi_m5_static (unsigned nb_iter)
{
  size_t global [2] = {GPU_SIZE_X, GPU_SIZE_Y};
  size_t local [2]  = {TILE_W, TILE_H};
  cl_int err;

  const unsigned BATCH_SIZE = 510; // Multiple of 5 for safety
  const int zero            = 0;

  err = clSetKernelArg (ocl_compute_kernel (0), 2, sizeof (cl_mem),
                        &ocl_changed_buffer);
  check (err, "Failed to set kernel arg 2");

  unsigned total_it = 0;
  monitoring_start (easypap_gpu_lane (0));

  for (unsigned it = 1; it <= nb_iter; it += BATCH_SIZE) {
    err = clEnqueueFillBuffer (ocl_queue (0), ocl_changed_buffer, &zero,
                               sizeof (int), 0, sizeof (int), 0, NULL, NULL);

    unsigned remaining = nb_iter - it + 1;
    unsigned max_k     = (remaining + 4) / 5;
    if (max_k > (BATCH_SIZE / 5))
      max_k = BATCH_SIZE / 5;
    if (max_k == 0)
      break;

    err = clSetKernelArg (ocl_compute_kernel (0), 0, sizeof (cl_mem),
                          &ocl_cur_buffer (0));
    err |= clSetKernelArg (ocl_compute_kernel (0), 1, sizeof (cl_mem),
                           &ocl_next_buffer (0));
    check (err, "Failed to set kernel args 0-1");

    for (unsigned k = 0; k < max_k; k++) {
      total_it += 5;

      err = clEnqueueNDRangeKernel (ocl_queue (0), ocl_compute_kernel (0), 2,
                                    NULL, global, local, 0, NULL, NULL);

      cl_mem tmp          = ocl_cur_buffer (0);
      ocl_cur_buffer (0)  = ocl_next_buffer (0);
      ocl_next_buffer (0) = tmp;

      err = clSetKernelArg (ocl_compute_kernel (0), 0, sizeof (cl_mem),
                            &ocl_cur_buffer (0));
      err |= clSetKernelArg (ocl_compute_kernel (0), 1, sizeof (cl_mem),
                             &ocl_next_buffer (0));
    }

    int changed;
    err = clEnqueueReadBuffer (ocl_queue (0), ocl_changed_buffer, CL_TRUE, 0,
                               sizeof (int), &changed, 0, NULL, NULL);
    check (err, "Failed to read changed flag");

    if (changed == 0) {
      clFinish (ocl_queue (0));
      monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (0));
      return total_it;
    }
  }

  clFinish (ocl_queue (0));
  monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (0));

  return 0;
}

void ssandPile_refresh_img_ocl_multi_m5_static (void)
{
  ssandPile_refresh_img_ocl ();
}

/*
 * Variant : multi_m6_robust
 * Strategy : margin 6 with 6 iterations per launch
 * Specificity : features an explicit tail handler to support 1000 iterations
 * since 1000 / 6 is not an integer
 * Usage : ./run -k ssandPile -g -v ocl_multi_m6_robust -s 1024 -i 1000
 */
static cl_kernel k_multi_m6_robust = NULL;
static cl_kernel k_multi_m5_static = NULL;
static cl_kernel k_multi_m4_loop   = NULL;
static cl_kernel k_naive           = NULL;

void ssandPile_init_ocl_multi_m6_robust (void)
{
  ssandPile_init_ocl ();

  cl_int err;
  k_multi_m6_robust =
      clCreateKernel (program, "ssandPile_ocl_multi_m6_robust", &err);
  check (err, "Failed to create k_multi_m6_robust");

  k_multi_m5_static =
      clCreateKernel (program, "ssandPile_ocl_multi_m5_static", &err);
  check (err, "Failed to create k_multi_m5_static");

  k_multi_m4_loop =
      clCreateKernel (program, "ssandPile_ocl_multi_m4_loop", &err);
  check (err, "Failed to create k_multi_m4_loop");

  k_naive = clCreateKernel (program, "ssandPile_ocl", &err);
  check (err, "Failed to create k_naive");
}

unsigned ssandPile_compute_ocl_multi_m6_robust (unsigned nb_iter)
{
  size_t global [2] = {GPU_SIZE_X, GPU_SIZE_Y};
  size_t local [2]  = {TILE_W, TILE_H};
  cl_int err;

  const unsigned BATCH_SIZE = 510;
  const int zero            = 0;

  unsigned total_it = 0;
  monitoring_start (easypap_gpu_lane (0));

  // 1. MAIN PASS
  unsigned full_batches = nb_iter / 6;
  for (unsigned b = 0; b < full_batches;) {
    err = clEnqueueFillBuffer (ocl_queue (0), ocl_changed_buffer, &zero,
                               sizeof (int), 0, sizeof (int), 0, NULL, NULL);

    unsigned max_k =
        (full_batches - b > BATCH_SIZE / 6) ? BATCH_SIZE / 6 : full_batches - b;

    for (unsigned k = 0; k < max_k; k++) {
      err = clSetKernelArg (k_multi_m6_robust, 0, sizeof (cl_mem),
                            &ocl_cur_buffer (0));
      err |= clSetKernelArg (k_multi_m6_robust, 1, sizeof (cl_mem),
                             &ocl_next_buffer (0));
      err |= clSetKernelArg (k_multi_m6_robust, 2, sizeof (cl_mem),
                             &ocl_changed_buffer);
      check (err, "Failed to set multi_m6_robust args");

      err = clEnqueueNDRangeKernel (ocl_queue (0), k_multi_m6_robust, 2, NULL,
                                    global, local, 0, NULL, NULL);
      total_it += 6;
      cl_mem tmp          = ocl_cur_buffer (0);
      ocl_cur_buffer (0)  = ocl_next_buffer (0);
      ocl_next_buffer (0) = tmp;
    }
    b += max_k;

    int changed;
    clEnqueueReadBuffer (ocl_queue (0), ocl_changed_buffer, CL_TRUE, 0,
                         sizeof (int), &changed, 0, NULL, NULL);
    if (changed == 0) {
      clFinish (ocl_queue (0));
      monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (0));
      return total_it;
    }
  }

  // 2. TAIL PASS, handle the remainder
  unsigned remainder = nb_iter - total_it;
  if (remainder > 0) {
    cl_kernel tail_k = NULL;
    unsigned steps   = 0;

    if (remainder == 5) {
      tail_k = k_multi_m5_static;
      steps  = 5;
    } else if (remainder == 4) {
      tail_k = k_multi_m4_loop;
      steps  = 4;
    } else {
      // Remainder 1, 2, or 3
      tail_k = k_naive;
      steps  = 1;
    }

    unsigned tail_launches = (steps == 1) ? remainder : 1;

    for (unsigned l = 0; l < tail_launches; l++) {
      err = clSetKernelArg (tail_k, 0, sizeof (cl_mem), &ocl_cur_buffer (0));
      err |= clSetKernelArg (tail_k, 1, sizeof (cl_mem), &ocl_next_buffer (0));
      err |= clSetKernelArg (tail_k, 2, sizeof (cl_mem), &ocl_changed_buffer);
      check (err, "Failed to set tail kernel args");

      err = clEnqueueNDRangeKernel (ocl_queue (0), tail_k, 2, NULL, global,
                                    local, 0, NULL, NULL);
      total_it += steps;
      cl_mem tmp          = ocl_cur_buffer (0);
      ocl_cur_buffer (0)  = ocl_next_buffer (0);
      ocl_next_buffer (0) = tmp;
    }
  }

  clFinish (ocl_queue (0));
  monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (0));
  return total_it;
}

void ssandPile_refresh_img_ocl_multi_m6_robust (void)
{
  ssandPile_refresh_img_ocl ();
}

/*
 * Variant : multi_m8_static
 * Strategy : margin 8 with 8 iterations per launch and 4-pass static load
 * Specificity : highest iteration density but increased LDS pressure
 * Usage : ./run -k ssandPile -g -v ocl_multi_m8_static -s 4096 -tw 32 -th 16
 */
void ssandPile_init_ocl_multi_m8_static (void)
{
  ssandPile_init_ocl ();
}

unsigned ssandPile_compute_ocl_multi_m8_static (unsigned nb_iter)
{
  size_t global [2] = {GPU_SIZE_X, GPU_SIZE_Y};
  size_t local [2]  = {TILE_W, TILE_H};
  cl_int err;

  const unsigned STEP       = 8;
  const unsigned BATCH_SIZE = 504;
  const int zero            = 0;

  err = clSetKernelArg (ocl_compute_kernel (0), 2, sizeof (cl_mem),
                        &ocl_changed_buffer);
  check (err, "Failed to set kernel arg 2");

  unsigned total_it = 0;
  monitoring_start (easypap_gpu_lane (0));

  for (unsigned it = 1; it <= nb_iter; it += BATCH_SIZE * STEP) {
    err = clEnqueueFillBuffer (ocl_queue (0), ocl_changed_buffer, &zero,
                               sizeof (int), 0, sizeof (int), 0, NULL, NULL);

    unsigned remaining = nb_iter - it + 1;
    unsigned max_k     = (remaining + STEP - 1) / STEP;
    if (max_k > BATCH_SIZE)
      max_k = BATCH_SIZE;
    if (max_k == 0)
      break;

    err = clSetKernelArg (ocl_compute_kernel (0), 0, sizeof (cl_mem),
                          &ocl_cur_buffer (0));
    err |= clSetKernelArg (ocl_compute_kernel (0), 1, sizeof (cl_mem),
                           &ocl_next_buffer (0));
    check (err, "Failed to set kernel args 0-1");

    for (unsigned k = 0; k < max_k; k++) {
      total_it += STEP;

      err = clEnqueueNDRangeKernel (ocl_queue (0), ocl_compute_kernel (0), 2,
                                    NULL, global, local, 0, NULL, NULL);

      cl_mem tmp          = ocl_cur_buffer (0);
      ocl_cur_buffer (0)  = ocl_next_buffer (0);
      ocl_next_buffer (0) = tmp;

      err = clSetKernelArg (ocl_compute_kernel (0), 0, sizeof (cl_mem),
                            &ocl_cur_buffer (0));
      err |= clSetKernelArg (ocl_compute_kernel (0), 1, sizeof (cl_mem),
                             &ocl_next_buffer (0));
    }

    int changed;
    err = clEnqueueReadBuffer (ocl_queue (0), ocl_changed_buffer, CL_TRUE, 0,
                               sizeof (int), &changed, 0, NULL, NULL);
    check (err, "Failed to read changed flag");

    if (changed == 0) {
      clFinish (ocl_queue (0));
      monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (0));
      return total_it;
    }
  }

  clFinish (ocl_queue (0));
  monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (0));

  return 0;
}

void ssandPile_refresh_img_ocl_multi_m8_static (void)
{
  ssandPile_refresh_img_ocl ();
}

/*
 * Variant : multi_m5_static_quad
 * Strategy : exploits 4-way symmetry by computing only the top-left quadrant
 * Usage : ./run -k ssandPile -g -v ocl_multi_m5_static_quad -s 4096 -tw 32 -th
 * 16
 */
void ssandPile_init_ocl_multi_m5_static_quad (void)
{
  ssandPile_init_ocl ();
}

unsigned ssandPile_compute_ocl_multi_m5_static_quad (unsigned nb_iter)
{
  /* Only launch over the top-left quadrant so that we have 4x fewer workgroups
   */
  size_t global [2] = {GPU_SIZE_X / 2, GPU_SIZE_Y / 2};
  size_t local [2]  = {TILE_W, TILE_H};
  cl_int err;

  const unsigned BATCH_SIZE = 510;
  const int zero            = 0;

  err = clSetKernelArg (ocl_compute_kernel (0), 2, sizeof (cl_mem),
                        &ocl_changed_buffer);
  check (err, "Failed to set kernel arg 2");

  unsigned total_it = 0;
  monitoring_start (easypap_gpu_lane (0));

  for (unsigned it = 1; it <= nb_iter; it += BATCH_SIZE) {

    err = clEnqueueFillBuffer (ocl_queue (0), ocl_changed_buffer, &zero,
                               sizeof (int), 0, sizeof (int), 0, NULL, NULL);

    unsigned remaining = nb_iter - it + 1;
    unsigned max_k     = (remaining + 4) / 5;
    if (max_k > BATCH_SIZE / 5)
      max_k = BATCH_SIZE / 5;
    if (max_k == 0)
      break;

    err = clSetKernelArg (ocl_compute_kernel (0), 0, sizeof (cl_mem),
                          &ocl_cur_buffer (0));
    err |= clSetKernelArg (ocl_compute_kernel (0), 1, sizeof (cl_mem),
                           &ocl_next_buffer (0));
    check (err, "Failed to set kernel args 0-1");

    for (unsigned k = 0; k < max_k; k++) {
      total_it += 5;
      err = clEnqueueNDRangeKernel (ocl_queue (0), ocl_compute_kernel (0), 2,
                                    NULL, global, local, 0, NULL, NULL);

      cl_mem tmp          = ocl_cur_buffer (0);
      ocl_cur_buffer (0)  = ocl_next_buffer (0);
      ocl_next_buffer (0) = tmp;

      err = clSetKernelArg (ocl_compute_kernel (0), 0, sizeof (cl_mem),
                            &ocl_cur_buffer (0));
      err |= clSetKernelArg (ocl_compute_kernel (0), 1, sizeof (cl_mem),
                             &ocl_next_buffer (0));
    }

    int changed;
    err = clEnqueueReadBuffer (ocl_queue (0), ocl_changed_buffer, CL_TRUE, 0,
                               sizeof (int), &changed, 0, NULL, NULL);
    check (err, "Failed to read changed flag");

    if (changed == 0) {
      clFinish (ocl_queue (0));
      monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (0));
      return total_it;
    }
  }

  clFinish (ocl_queue (0));
  monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (0));
  return 0;
}

void ssandPile_refresh_img_ocl_multi_m5_static_quad (void)
{
  /* 1. Read the GPU buffer into the TABLE and convert to image as usual */
  ssandPile_refresh_img_ocl ();

  /* 2. Replicate the top-left quadrant to the other three */
  const int QDIM = DIM / 2;

  for (int y = 0; y < QDIM; y++) {
    for (int x = 0; x < QDIM; x++) {
      uint32_t color = cur_img (y, x);

      cur_img (y, DIM - 1 - x)           = color; /* top-right */
      cur_img (DIM - 1 - y, x)           = color; /* bottom-left */
      cur_img (DIM - 1 - y, DIM - 1 - x) = color; /* bottom-right */
    }
  }
}

/*
 * =============================================
 * HYBRID OpenMP-OpenCL (Step 4)
 * =============================================
 * Domain split horizontally :
 *   GPU : rows [0, gpu_y_end)
 *   CPU : rows [gpu_y_end, DIM)
 */

static int gpu_y_end        = 0;
static int border_thickness = 0;

/* ---- Config ---- */

void ssandPile_config_ocl_hybrid (char *param)
{
  sandPile_config (param);
  easypap_gl_buffer_sharing = 0;
}

void ssandPile_config_ocl_hybrid_thick (char *param)
{
  ssandPile_config_ocl_hybrid (param);
}

/* ---- Init ---- */

static int border_thickness_mult = 1;

static void hybrid_common_init (void)
{
  ssandPile_init_ocl ();

  int gpu_pct = 75;
  if (config_param != NULL) {
    char *colon = strchr (config_param, ':');
    if (colon != NULL) {
      border_thickness_mult = atoi (colon + 1);
    }
    gpu_pct = atoi (config_param);
  }

  gpu_y_end = (DIM * gpu_pct) / 100;
  gpu_y_end = (gpu_y_end / TILE_H) * TILE_H;
  if (gpu_y_end <= 0)
    gpu_y_end = TILE_H;
  if (gpu_y_end >= (int)DIM)
    gpu_y_end = DIM - TILE_H;

  PRINT_DEBUG ('u', "Hybrid: GPU rows [0,%d), CPU rows [%d,%d), split=%d%%\n",
               gpu_y_end, gpu_y_end, DIM, gpu_pct);
  PRINT_DEBUG ('u', "tile size: %d %d\n", TILE_W, TILE_H);
}

void ssandPile_init_ocl_hybrid (void)
{
  hybrid_common_init ();
  border_thickness = 1;
}

void ssandPile_init_ocl_hybrid_thick (void)
{
  hybrid_common_init ();
  border_thickness = border_thickness_mult * TILE_H;
  if (gpu_y_end + border_thickness > (int)DIM)
    border_thickness = DIM - gpu_y_end;
  if (gpu_y_end - border_thickness < 0)
    border_thickness = gpu_y_end;
  PRINT_DEBUG ('u', "Hybrid thick: border_thickness = %d\n", border_thickness);
}

/* ---- Send data (TABLE -> GPU) ---- */

void ssandPile_send_data_ocl_hybrid (void)
{
  cl_int err;
  const unsigned size = DIM * DIM * sizeof (unsigned);

  err =
      clEnqueueWriteBuffer (ocl_queue (0), ocl_cur_buffer (0), CL_TRUE, 0, size,
                            table_cell (TABLE, in, 0, 0), 0, NULL, NULL);
  check (err, "hybrid: failed to write cur_buffer");

  err =
      clEnqueueWriteBuffer (ocl_queue (0), ocl_next_buffer (0), CL_TRUE, 0,
                            size, table_cell (TABLE, out, 0, 0), 0, NULL, NULL);
  check (err, "hybrid: failed to write next_buffer");
}

void ssandPile_send_data_ocl_hybrid_thick (void)
{
  ssandPile_send_data_ocl_hybrid ();
}

/* ---- Refresh image (GPU -> TABLE -> pixels) ---- */

void ssandPile_refresh_img_ocl_hybrid (void)
{
  cl_int err;
  err = clEnqueueReadBuffer (ocl_queue (0), ocl_cur_buffer (0), CL_TRUE, 0,
                             gpu_y_end * DIM * sizeof (unsigned),
                             table_cell (TABLE, in, 0, 0), 0, NULL, NULL);
  check (err, "hybrid: failed to read GPU rows for refresh");
  ssandPile_refresh_img ();
}

void ssandPile_refresh_img_ocl_hybrid_thick (void)
{
  ssandPile_refresh_img_ocl_hybrid ();
}

/* ---- V1 : thin border (exchange 1 row per iteration) ---- */

unsigned ssandPile_compute_ocl_hybrid (unsigned nb_iter)
{
  size_t global [2] = {GPU_SIZE_X, (size_t)gpu_y_end};
  size_t local [2]  = {TILE_W, TILE_H};
  cl_int err;
  const int zero = 0;

  err = clSetKernelArg (ocl_compute_kernel (0), 2, sizeof (cl_mem),
                        &ocl_changed_buffer);
  err |= clSetKernelArg (ocl_compute_kernel (0), 3, sizeof (int), &gpu_y_end);
  check (err, "hybrid: failed to set args 2-3");

  monitoring_start (easypap_gpu_lane (0));

  for (unsigned it = 1; it <= nb_iter; it++) {
    int cpu_changed = 0;

    err = clEnqueueFillBuffer (ocl_queue (0), ocl_changed_buffer, &zero,
                               sizeof (int), 0, sizeof (int), 0, NULL, NULL);

    /* GPU launch (async) */
    err = clSetKernelArg (ocl_compute_kernel (0), 0, sizeof (cl_mem),
                          &ocl_cur_buffer (0));
    err |= clSetKernelArg (ocl_compute_kernel (0), 1, sizeof (cl_mem),
                           &ocl_next_buffer (0));
    check (err, "hybrid: failed to set args 0-1");

    err = clEnqueueNDRangeKernel (ocl_queue (0), ocl_compute_kernel (0), 2,
                                  NULL, global, local, 0, NULL, NULL);
    check (err, "hybrid: kernel launch failed");

    /* CPU computation (concurrent with GPU) */
#pragma omp parallel for collapse(2) schedule(static) reduction(| : cpu_changed)
    for (int y = gpu_y_end; y < (int)DIM; y += TILE_H) {
      for (int x = 0; x < (int)DIM; x += TILE_W) {
        int y_0 = (y == 0);
        int yf  = (y + TILE_H == (int)DIM);
        int x_0 = (x == 0);
        int xf  = (x + TILE_W == (int)DIM);
        cpu_changed |=
            do_tile (x + x_0, y + y_0, TILE_W - xf - x_0, TILE_H - yf - y_0);
      }
    }

    /* Wait for GPU */
    clFinish (ocl_queue (0));

    /* GPU -> CPU : row (gpu_y_end-1) from GPU out buffer -> TABLE[out] */
    err = clEnqueueReadBuffer (
        ocl_queue (0), ocl_next_buffer (0), CL_TRUE,
        (size_t)(gpu_y_end - 1) * DIM * sizeof (unsigned),
        DIM * sizeof (unsigned), table_cell (TABLE, out, gpu_y_end - 1, 0), 0,
        NULL, NULL);
    check (err, "hybrid: GPU -> CPU border read failed");

    /* CPU -> GPU : row gpu_y_end from TABLE[out] -> GPU out buffer */
    err = clEnqueueWriteBuffer (
        ocl_queue (0), ocl_next_buffer (0), CL_TRUE,
        (size_t)gpu_y_end * DIM * sizeof (unsigned), DIM * sizeof (unsigned),
        table_cell (TABLE, out, gpu_y_end, 0), 0, NULL, NULL);
    check (err, "hybrid: CPU -> GPU border write failed");

    /* Swap both sides */
    cl_mem tmp          = ocl_cur_buffer (0);
    ocl_cur_buffer (0)  = ocl_next_buffer (0);
    ocl_next_buffer (0) = tmp;
    swap_tables ();

    /* Termination check */
    int gpu_changed;
    err = clEnqueueReadBuffer (ocl_queue (0), ocl_changed_buffer, CL_TRUE, 0,
                               sizeof (int), &gpu_changed, 0, NULL, NULL);

    if (gpu_changed == 0 && cpu_changed == 0) {
      monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (0));
      return it;
    }
  }

  monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (0));
  return 0;
}

/* ---- V2 : thick border (exchange every border_thickness iterations) ---- */

unsigned ssandPile_compute_ocl_hybrid_thick (unsigned nb_iter)
{
  size_t global [2] = {GPU_SIZE_X, (size_t)(gpu_y_end + border_thickness)};
  size_t local [2]  = {TILE_W, TILE_H};
  cl_int err;
  const int zero = 0;
  const int bt   = border_thickness;

  err = clSetKernelArg (ocl_compute_kernel (0), 2, sizeof (cl_mem),
                        &ocl_changed_buffer);
  err |= clSetKernelArg (ocl_compute_kernel (0), 3, sizeof (int), &gpu_y_end);
  check (err, "hybrid_thick: failed to set args 2-3");

  monitoring_start (easypap_gpu_lane (0));

  unsigned total_it = 0;
  int cpu_start     = gpu_y_end - bt;
  if (cpu_start < 0)
    cpu_start = 0;

  for (unsigned it = 1; it <= nb_iter; it += bt) {
    int cpu_changed = 0;

    err = clEnqueueFillBuffer (ocl_queue (0), ocl_changed_buffer, &zero,
                               sizeof (int), 0, sizeof (int), 0, NULL, NULL);

    /* Run bt sub-iterations */
    for (int sub = 0; sub < bt && total_it < nb_iter; sub++) {

      err = clSetKernelArg (ocl_compute_kernel (0), 0, sizeof (cl_mem),
                            &ocl_cur_buffer (0));
      err |= clSetKernelArg (ocl_compute_kernel (0), 1, sizeof (cl_mem),
                             &ocl_next_buffer (0));
      check (err, "hybrid_thick: failed to set args");

      err = clEnqueueNDRangeKernel (ocl_queue (0), ocl_compute_kernel (0), 2,
                                    NULL, global, local, 0, NULL, NULL);
      check (err, "hybrid_thick: kernel launch failed");

      /* CPU : rows [gpu_y_end - bt, DIM) */
      int sub_cpu_changed = 0;
#pragma omp parallel for collapse(2) schedule(static)                          \
    reduction(| : sub_cpu_changed)
      for (int y = cpu_start; y < (int)DIM; y += TILE_H) {
        for (int x = 0; x < (int)DIM; x += TILE_W) {
          int y_0 = (y == 0);
          int yf  = (y + TILE_H == (int)DIM);
          int x_0 = (x == 0);
          int xf  = (x + TILE_W == (int)DIM);
          int changed =
              do_tile (x + x_0, y + y_0, TILE_W - xf - x_0, TILE_H - yf - y_0);
          if (y >= gpu_y_end) {
            sub_cpu_changed |= changed;
          }
        }
      }
      cpu_changed |= sub_cpu_changed;

      clFinish (ocl_queue (0));

      cl_mem tmp          = ocl_cur_buffer (0);
      ocl_cur_buffer (0)  = ocl_next_buffer (0);
      ocl_next_buffer (0) = tmp;
      swap_tables ();

      total_it++;
    }

    /* Thick border exchange (on current/input buffers after swaps) */
    /* GPU -> CPU : rows [gpu_y_end-bt, gpu_y_end) */
    err = clEnqueueReadBuffer (
        ocl_queue (0), ocl_cur_buffer (0), CL_TRUE,
        (size_t)(gpu_y_end - bt) * DIM * sizeof (unsigned),
        (size_t)bt * DIM * sizeof (unsigned),
        table_cell (TABLE, in, gpu_y_end - bt, 0), 0, NULL, NULL);
    check (err, "hybrid_thick: GPU -> CPU border read failed");

    /* CPU -> GPU : rows [gpu_y_end, gpu_y_end+bt) */
    err = clEnqueueWriteBuffer (ocl_queue (0), ocl_cur_buffer (0), CL_TRUE,
                                (size_t)gpu_y_end * DIM * sizeof (unsigned),
                                (size_t)bt * DIM * sizeof (unsigned),
                                table_cell (TABLE, in, gpu_y_end, 0), 0, NULL,
                                NULL);
    check (err, "hybrid_thick: CPU -> GPU border write failed");

    /* Termination */
    int gpu_changed;
    err = clEnqueueReadBuffer (ocl_queue (0), ocl_changed_buffer, CL_TRUE, 0,
                               sizeof (int), &gpu_changed, 0, NULL, NULL);

    if (gpu_changed == 0 && cpu_changed == 0) {
      monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (0));
      return total_it;
    }
  }

  monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (0));
  return 0;
}

/* ---- V3 : dynamic thick border (adjusts GPU/CPU ratio via EWMA) ---- */

void ssandPile_config_ocl_hybrid_dynamic (char *param)
{
  ssandPile_config_ocl_hybrid (param);
}

void ssandPile_init_ocl_hybrid_dynamic (void)
{
  hybrid_common_init ();
  border_thickness = TILE_H;
  if (gpu_y_end + border_thickness > (int)DIM)
    border_thickness = DIM - gpu_y_end;
  if (gpu_y_end - border_thickness < 0)
    border_thickness = gpu_y_end;
  PRINT_DEBUG ('u', "Hybrid dynamic: initial border_thickness = %d\n",
               border_thickness);
}

void ssandPile_send_data_ocl_hybrid_dynamic (void)
{
  ssandPile_send_data_ocl_hybrid ();
}

void ssandPile_refresh_img_ocl_hybrid_dynamic (void)
{
  ssandPile_refresh_img_ocl_hybrid ();
}

unsigned ssandPile_compute_ocl_hybrid_dynamic (unsigned nb_iter)
{
  size_t local [2] = {TILE_W, TILE_H};
  cl_int err;
  const int zero = 0;
  const int bt   = border_thickness;

  err = clSetKernelArg (ocl_compute_kernel (0), 2, sizeof (cl_mem),
                        &ocl_changed_buffer);
  err |= clSetKernelArg (ocl_compute_kernel (0), 3, sizeof (int), &gpu_y_end);
  check (err, "hybrid_dyn: failed to set args 2-3");

  monitoring_start (easypap_gpu_lane (0));

  unsigned total_it   = 0;
  float ewma_gpu_wait = 0.0f;

  for (unsigned it = 1; it <= nb_iter; it += bt) {
    int cpu_changed = 0;

    int cpu_start = gpu_y_end - bt;
    if (cpu_start < 0)
      cpu_start = 0;

    size_t global [2] = {GPU_SIZE_X, (size_t)(gpu_y_end + bt)};

    err = clEnqueueFillBuffer (ocl_queue (0), ocl_changed_buffer, &zero,
                               sizeof (int), 0, sizeof (int), 0, NULL, NULL);

    /* Run bt sub-iterations */
    double t_start  = omp_get_wtime ();
    double cpu_time = 0.0;

    for (int sub = 0; sub < bt && total_it < nb_iter; sub++) {

      err = clSetKernelArg (ocl_compute_kernel (0), 0, sizeof (cl_mem),
                            &ocl_cur_buffer (0));
      err |= clSetKernelArg (ocl_compute_kernel (0), 1, sizeof (cl_mem),
                             &ocl_next_buffer (0));
      err |=
          clSetKernelArg (ocl_compute_kernel (0), 3, sizeof (int), &gpu_y_end);
      check (err, "hybrid_dyn: failed to set args");

      err = clEnqueueNDRangeKernel (ocl_queue (0), ocl_compute_kernel (0), 2,
                                    NULL, global, local, 0, NULL, NULL);
      check (err, "hybrid_dyn: kernel launch failed");

      /* CPU */
      int sub_cpu_changed = 0;
      double t_cpu_start  = omp_get_wtime ();
#pragma omp parallel for collapse(2) schedule(static)                          \
    reduction(| : sub_cpu_changed)
      for (int y = cpu_start; y < (int)DIM; y += TILE_H) {
        for (int x = 0; x < (int)DIM; x += TILE_W) {
          int y_0 = (y == 0);
          int yf  = (y + TILE_H == (int)DIM);
          int x_0 = (x == 0);
          int xf  = (x + TILE_W == (int)DIM);
          int changed =
              do_tile (x + x_0, y + y_0, TILE_W - xf - x_0, TILE_H - yf - y_0);
          if (y >= gpu_y_end) {
            sub_cpu_changed |= changed;
          }
        }
      }
      cpu_time += (omp_get_wtime () - t_cpu_start);
      cpu_changed |= sub_cpu_changed;

      clFinish (ocl_queue (0));

      cl_mem tmp          = ocl_cur_buffer (0);
      ocl_cur_buffer (0)  = ocl_next_buffer (0);
      ocl_next_buffer (0) = tmp;
      swap_tables ();

      total_it++;
    }

    double total_time    = omp_get_wtime () - t_start;
    double gpu_wait_time = total_time - cpu_time;

    // EWMA update (alpha = 0.2)
    ewma_gpu_wait = 0.2f * (float)gpu_wait_time + 0.8f * ewma_gpu_wait;

    /* Thick border exchange */
    err = clEnqueueReadBuffer (
        ocl_queue (0), ocl_cur_buffer (0), CL_TRUE,
        (size_t)(gpu_y_end - bt) * DIM * sizeof (unsigned),
        (size_t)bt * DIM * sizeof (unsigned),
        table_cell (TABLE, in, gpu_y_end - bt, 0), 0, NULL, NULL);
    check (err, "hybrid_dyn: GPU->CPU border read failed");

    err = clEnqueueWriteBuffer (ocl_queue (0), ocl_cur_buffer (0), CL_TRUE,
                                (size_t)gpu_y_end * DIM * sizeof (unsigned),
                                (size_t)bt * DIM * sizeof (unsigned),
                                table_cell (TABLE, in, gpu_y_end, 0), 0, NULL,
                                NULL);
    check (err, "hybrid_dyn: CPU->GPU border write failed");

    /* Termination */
    int gpu_changed;
    err = clEnqueueReadBuffer (ocl_queue (0), ocl_changed_buffer, CL_TRUE, 0,
                               sizeof (int), &gpu_changed, 0, NULL, NULL);

    if (gpu_changed == 0 && cpu_changed == 0) {
      monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (0));
      return total_it;
    }

    /* Dynamic load balancing */
    int old_gpu_y_end = gpu_y_end;
    // 0.0005 seconds = 500 us, 0.0001 seconds = 100 us
    if (ewma_gpu_wait > 0.0005f) {
      if (gpu_y_end >= 3 * TILE_H)
        gpu_y_end -= TILE_H;
    } else if (ewma_gpu_wait < 0.0001f) {
      if (gpu_y_end <= (int)DIM - 3 * TILE_H)
        gpu_y_end += TILE_H;
    }

    if (gpu_y_end != old_gpu_y_end) {
      if (gpu_y_end > old_gpu_y_end) {
        // GPU grew : transfer rows [old, new) from CPU in to GPU in
        err = clEnqueueWriteBuffer (
            ocl_queue (0), ocl_cur_buffer (0), CL_TRUE,
            (size_t)old_gpu_y_end * DIM * sizeof (unsigned),
            (size_t)(gpu_y_end - old_gpu_y_end) * DIM * sizeof (unsigned),
            table_cell (TABLE, in, old_gpu_y_end, 0), 0, NULL, NULL);
        check (err, "hybrid_dyn: CPU->GPU dynamic shift failed");
      } else {
        // CPU grew : transfer rows [new, old) from GPU in to CPU in
        err = clEnqueueReadBuffer (
            ocl_queue (0), ocl_cur_buffer (0), CL_TRUE,
            (size_t)gpu_y_end * DIM * sizeof (unsigned),
            (size_t)(old_gpu_y_end - gpu_y_end) * DIM * sizeof (unsigned),
            table_cell (TABLE, in, gpu_y_end, 0), 0, NULL, NULL);
        check (err, "hybrid_dyn: GPU->CPU dynamic shift failed");
      }
    }
  }

  monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (0));
  return 0;
}

#endif
