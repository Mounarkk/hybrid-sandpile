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

void ssandPile_init_lazy (void)
{
  ssandPile_init ();

  if (TILESET != NULL)
    return;

  TILESET = tileset_init (NB_TILES_X, NB_TILES_Y);
  tileset_mark_full (TILESET);
}

void ssandPile_init_omp_lazy (void)
{
  ssandPile_init_lazy ();
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
#ifdef __GNUC__
#pragma GCC                                                                    \
    unroll 4 /* Here, an unroll of 4 seems to give out the best results*/
#elif __clang__
#pragma unroll 4
#endif
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
//
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

unsigned ssandPile_compute_lazy (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {
    int change = 0;

    const unsigned long nb_tiles = tileset_nb_tiles (TILESET);
    const tile *restrict tiles   = tileset_flush_tiles (TILESET);

    for (unsigned i = 0; i < nb_tiles; i++) {
      unsigned long x, y;
      tile t = tiles [i];
      x      = t.tx * TILE_W;
      y      = t.ty * TILE_H;

      int y_0   = (y == 0);
      int x_0   = (x == 0);
      int y_end = (y + TILE_H == DIM);
      int x_end = (x + TILE_W == DIM);

      int loc_change = do_tile (x + x_0, y + y_0, TILE_W - x_end - x_0,
                                TILE_H - y_end - y_0);

      tileset_mark_at (TILESET, loc_change, tiles [i]);
      tileset_mark_at (TILESET, loc_change,
                       (tile){.tx = t.tx - 1 + x_0, .ty = t.ty});
      tileset_mark_at (TILESET, loc_change,
                       (tile){.tx = t.tx + 1 - x_end, .ty = t.ty});
      tileset_mark_at (TILESET, loc_change,
                       (tile){.tx = t.tx, .ty = t.ty + 1 - y_end});
      tileset_mark_at (TILESET, loc_change,
                       (tile){.tx = t.tx, .ty = t.ty - 1 + y_0});

      change |= loc_change;
    }

    swap_tables ();
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

    const unsigned long nb_tiles = tileset_nb_tiles (TILESET);
    const tile *restrict tiles   = tileset_flush_tiles (TILESET);

    int ratio = 100 * ((float)nb_tiles) / tileset_get_total_tiles (TILESET);
    int nb_threads = num_threads; // num_threads * ratio + 1;
    if (ratio < 50)
      nb_threads = num_threads >> 1;
    if (ratio < 25)
      nb_threads = num_threads >> 2;

    // printf ("Number of tiles dropped to %lu/%lu, ratio is %d"
    //         "which makes %d "
    //         "threads\n",
    //         nb_tiles, tileset_get_total_tiles (TILESET), ratio, nb_threads);

#pragma omp parallel num_threads(nb_threads)
#pragma omp for schedule(runtime)
    for (unsigned i = 0; i < nb_tiles; i++) {
      unsigned long x, y;
      tile t = tiles [i];
      x      = t.tx * TILE_W;
      y      = t.ty * TILE_H;

      int y_0   = (y == 0);
      int x_0   = (x == 0);
      int y_end = (y + TILE_H == DIM);
      int x_end = (x + TILE_W == DIM);

      int loc_change = do_tile (x + x_0, y + y_0, TILE_W - x_end - x_0,
                                TILE_H - y_end - y_0);
#pragma omp critical
      {
        tileset_mark_at (TILESET, loc_change, tiles [i]);
        tileset_mark_at (TILESET, loc_change,
                         (tile){.tx = t.tx - 1 + x_0, .ty = t.ty});
        tileset_mark_at (TILESET, loc_change,
                         (tile){.tx = t.tx + 1 - x_end, .ty = t.ty});
        tileset_mark_at (TILESET, loc_change,
                         (tile){.tx = t.tx, .ty = t.ty + 1 - y_end});
        tileset_mark_at (TILESET, loc_change,
                         (tile){.tx = t.tx, .ty = t.ty - 1 + y_0});

        change |= loc_change;
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
