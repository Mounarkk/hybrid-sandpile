#include "sandPile_common.h"

#include <omp.h>
#include <stdbool.h>
#include <sys/mman.h>
#include <unistd.h>

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

int num_threads;
tileset_t *restrict tilesets = NULL;
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

  printf ("TOTAL_NB_SETS %d, SETS_PER_ROW %d, \n", TOTAL_NB_SETS, SETS_PER_ROW);

  tileset_mark_full (TILESET);
}

void ssandPile_init_lazy_border (void)
{
  ssandPile_init_lazy ();
}

void ssandPile_init_omp_lazy (void)
{
  ssandPile_init_lazy ();

  tilesets = malloc (sizeof (tileset_t *) * num_threads);
  for (unsigned i = 0; i < num_threads; i++)
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

  for (unsigned i = 0; i < num_threads; i++)
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

    change = tileset_merge (TILESET, tilesets, num_threads);
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

#pragma omp parallel shared(TABLE)
    {
      tileset_t curr = tilesets [omp_get_thread_num ()];
      tileset_mark_empty (curr);
#pragma omp for schedule(static)
      for (unsigned i = 0; i < TOTAL_NB_SETS; i++) {
        unsigned char in_set = PER_SET [i];
        unsigned tile_y      = i / SETS_PER_ROW;
        unsigned set_x       = i - tile_y * SETS_PER_ROW;
        bitset set           = sets [i];

        for (unsigned char k = 0; k < in_set; k++) {
          unsigned pos = bitset_clz (set);
          set -= bitset_at (pos);

          tile t;
          t.tx = BITSET_SIZE_MUL (set_x) + pos;
          t.ty = tile_y;

          unsigned long x, y;
          x = t.tx * TILE_W;
          y = t.ty * TILE_H;

          int y_0   = (y == 0);
          int x_0   = (x == 0);
          int y_end = (y + TILE_H == DIM);
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

          tileset_mark_at (curr, mark_self, self);
          tileset_mark_at (curr, mark_up, up);
          tileset_mark_at (curr, mark_down, down);
          tileset_mark_at (curr, mark_left, left);
          tileset_mark_at (curr, mark_right, right);
        }

        sets [i] = 0;
      }
    }

    swap_tables ();

    change = tileset_merge (TILESET, tilesets, num_threads);
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

  // if (width != TILE_W) {
  //   return ssandPile_do_tile_opt_border (x, y, width, height);
  int on_side = width != TILE_W;
  width += on_side;
  x &= ~1;

  TYPE *in_cell  = table_cell (TABLE, in, y, x);
  TYPE *out_cell = table_cell (TABLE, out, y, x);

  for (int j = 0; j < width; j += AVX_VEC_SIZE_INT) {
    int do_left  = -1 * !((x + j == 0) & on_side);
    int do_right = -1 * !((x + j == DIM - AVX_VEC_SIZE_INT) & on_side);
    __m256i mask = _mm256_set_epi32 (do_right, -1, -1, -1, -1, -1, -1, do_left);

    __m256i up_vec     = _mm256_load_si256 ((__m256i *)(in_cell - DIM));
    __m256i center_vec = _mm256_load_si256 ((__m256i *)(in_cell));
    __m256i down_vec;

    int is_border_left  = (j == 0);
    int is_border_right = (j == (width - AVX_VEC_SIZE_INT));

    for (int i = 0; i < height; i++) {
      int is_border_up   = (i == 0);
      int is_border_down = (i == (height - 1));

      __m256i res = _mm256_and_si256 (center_vec, m256_3);

      down_vec          = _mm256_load_si256 ((__m256i *)(in_cell + DIM));
      __m256i left_vec  = _mm256_maskload_epi32 ((int *)(in_cell - 1), mask);
      __m256i right_vec = _mm256_maskload_epi32 ((int *)(in_cell + 1), mask);

      /* cell & 3 */

      res = _mm256_add_epi32 (res, _mm256_srli_epi32 (up_vec, 2));
      res = _mm256_add_epi32 (res, _mm256_srli_epi32 (down_vec, 2));
      res = _mm256_add_epi32 (res, _mm256_srli_epi32 (left_vec, 2));
      res = _mm256_add_epi32 (res, _mm256_srli_epi32 (right_vec, 2));

      _mm256_maskstore_epi32 ((int *)out_cell, mask, res);

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
    __m256i up_vec     = _mm256_load_si256 ((__m256i *)(in_cell - DIM));
    __m256i center_vec = _mm256_load_si256 ((__m256i *)(in_cell));
    __m256i down_vec;

    for (int i = 0; i < height; i++) {

      __m256i res = _mm256_and_si256 (center_vec, m256_3);
      res         = _mm256_add_epi32 (res, _mm256_srli_epi32 (up_vec, 2));

      down_vec = _mm256_load_si256 ((__m256i *)(in_cell + DIM));
      res      = _mm256_add_epi32 (res, _mm256_srli_epi32 (down_vec, 2));

      __m256i left_vec = _mm256_loadu_si256 ((__m256i *)(in_cell - 1));
      res = _mm256_add_epi32 (res, _mm256_srli_epi32 (left_vec, 2));

      __m256i right_vec = _mm256_loadu_si256 ((__m256i *)(in_cell + 1));
      res = _mm256_add_epi32 (res, _mm256_srli_epi32 (right_vec, 2));

      _mm256_store_si256 ((__m256i *)out_cell, res);

      __m256 cast              = (__m256)_mm256_cmpeq_epi32 (res, center_vec);
      unsigned char change_vec = ~(unsigned char)_mm256_movemask_ps (cast);

      int self_flag = SANDPILE_BORDER_SET_SELF (change_vec != 0);
      // int up_flag   = SANDPILE_BORDER_SET_UP (self_flag, is_border_up);
      // int down_flag = SANDPILE_BORDER_SET_DOWN (self_flag, is_border_down);
      // int left_flag = SANDPILE_BORDER_SET_LEFT (change_left, is_border_left);
      // int right_flag =
      //     SANDPILE_BORDER_SET_RIGHT (change_right, is_border_right);

      diff |= self_flag; //| up_flag | down_flag | left_flag | right_flag;

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

  const int offset = DIM - width;
  int diff         = 0;

  TYPE *in_cell  = table_cell (TABLE, in, y, x);
  TYPE *out_cell = table_cell (TABLE, out, y, x);

  const __m256i mask_zero  = _mm256_set1_epi32 (0);
  const __m256i mask_mod_4 = _mm256_set1_epi32 (3);

  for (int i = 0; i < height; i++) {
    int is_border_up   = (i == 0);
    int is_border_down = (i == (height - 1));
    __m256i left_vec, center_vec, right_vec;

    left_vec =
        _mm256_set_epi32 (*(in_cell - 1 * !left_tile), 0, 0, 0, 0, 0, 0, 0);
    center_vec = _mm256_load_si256 ((__m256i *)in_cell);

    for (int j = 0; j < width; j += AVX_VEC_SIZE_INT) {
      int is_border_left  = (j == 0);
      int is_border_right = (j == (width - AVX_VEC_SIZE_INT));
      int do_left         = -1 * (!is_border_left | !left_tile);
      int do_right        = -1 * (!is_border_right | !right_tile);

      right_vec = _mm256_load_si256 ((__m256i *)(in_cell + AVX_VEC_SIZE_INT));

      __m256i cross_vec = _mm256_blend_epi32 (center_vec, left_vec, 0b10000000);
      cross_vec         = _mm256_blend_epi32 (cross_vec, right_vec, 0b00000001);

      __m256i vec_crossed = _mm256_permutevar8x32_epi32 (
          cross_vec, _mm256_setr_epi32 (7, 0, 0, 4, 3, 0, 0, 0));

      __m256i right_added_vec = _mm256_blend_epi32 (
          _mm256_bsrli_epi128 (center_vec, sizeof (unsigned int)), vec_crossed,
          0b10001000);

      __m256i left_added_vec = _mm256_blend_epi32 (
          _mm256_bslli_epi128 (center_vec, sizeof (unsigned int)), vec_crossed,
          0b00010001);

      __m256i res = _mm256_add_epi32 (_mm256_srli_epi32 (left_added_vec, 2),
                                      _mm256_srli_epi32 (right_added_vec, 2));
      res = _mm256_add_epi32 (res, _mm256_and_si256 (center_vec, mask_mod_4));

      __m256i up_vec   = _mm256_load_si256 ((__m256i *)(in_cell - DIM));
      __m256i down_vec = _mm256_load_si256 ((__m256i *)(in_cell + DIM));
      __m256i tops     = _mm256_add_epi32 (_mm256_srli_epi32 (down_vec, 2),
                                           _mm256_srli_epi32 (up_vec, 2));
      res              = _mm256_add_epi32 (res, tops);

      __m256i mask =
          _mm256_set_epi32 (do_right, -1, -1, -1, -1, -1, -1, do_left);
      res = _mm256_blendv_epi8 (mask_zero, res, mask);

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

      in_cell += AVX_VEC_SIZE_INT;
      out_cell += AVX_VEC_SIZE_INT;

      left_vec   = center_vec;
      center_vec = right_vec;
    }

    in_cell += offset;
    out_cell += offset;
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

      // __m256i right_div_vec = _mm256_loadu_si256 ((__m256i *)(in_cell + 1));
      res = _mm256_add_epi32 (res, _mm256_srli_epi32 (right_div_vec, 2));

      __m256i left_div_vec = _mm256_blend_epi32 (rotate_right (center_vec),
                                                 rotate_right (left_vec), 1);

      // __m256i left_div_vec = _mm256_loadu_si256 ((__m256i *)(in_cell - 1));
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

void ssandPile_refresh_img_ocl (void)
{
  cl_int err;

  err =
      clEnqueueReadBuffer (ocl_queue (0), ocl_cur_buffer (0), CL_TRUE, 0,
                           sizeof (unsigned) * DIM * DIM, TABLE, 0, NULL, NULL);
  check (err, "Failed to read buffer from GPU");

  ssandPile_refresh_img ();
}

#endif
