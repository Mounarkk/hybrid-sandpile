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

  tilesets = malloc (sizeof (tileset_t *) * num_threads);
  for (unsigned i = 0; i < num_threads; i++)
    tilesets [i] = tileset_init (NB_TILES_X, NB_TILES_Y);
}

void ssandPile_init_omp_lazy_border (void)
{
  ssandPile_init_omp_lazy ();
}

void ssandPile_finalize (void)
{
  const unsigned size = 2 * DIM * DIM * sizeof (TYPE);

  ezp_free (TABLE, size);
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
/*
 * =============================================
 * LAZY EVALUATION
 * =============================================
 */

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
        unsigned pos = __builtin_clzll (sets [i]);
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
        unsigned pos = __builtin_clzll (sets [i]);
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
          unsigned pos = __builtin_clzll (sets [i]);
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
          unsigned pos = __builtin_clzll (set);
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

void ssandPile_init_ocl_opt3 (void)
{
  ssandPile_init_ocl ();
}

unsigned ssandPile_compute_ocl_opt3 (unsigned nb_iter)
{
  size_t global [2] = {GPU_SIZE_X, GPU_SIZE_Y};
  size_t local [2]  = {TILE_W, TILE_H};
  cl_int err;

  const unsigned BATCH_SIZE = 512;
  const int zero            = 0;

  err = clSetKernelArg (ocl_compute_kernel (0), 2, sizeof (cl_mem),
                        &ocl_changed_buffer);
  check (err, "Failed to set kernel arg 2");

  unsigned total_it = 0;
  monitoring_start (easypap_gpu_lane (0));

  for (unsigned it = 1; it <= nb_iter; it += BATCH_SIZE * 4) {
    err = clEnqueueFillBuffer (ocl_queue (0), ocl_changed_buffer, &zero,
                               sizeof (int), 0, sizeof (int), 0, NULL, NULL);

    unsigned remaining = nb_iter - it + 1;
    unsigned max_k     = (remaining + 3) / 4;
    if (max_k > BATCH_SIZE)
      max_k = BATCH_SIZE;
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

void ssandPile_refresh_img_ocl_opt3 (void)
{
  ssandPile_refresh_img_ocl ();
}

void ssandPile_init_ocl_opt4 (void)
{
  ssandPile_init_ocl ();
}

unsigned ssandPile_compute_ocl_opt4 (unsigned nb_iter)
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

void ssandPile_refresh_img_ocl_opt4 (void)
{
  ssandPile_refresh_img_ocl ();
}

void ssandPile_init_ocl_opt5 (void)
{
  ssandPile_init_ocl ();
}

unsigned ssandPile_compute_ocl_opt5 (unsigned nb_iter)
{
  size_t global [2] = {GPU_SIZE_X, GPU_SIZE_Y};
  size_t local [2]  = {TILE_W, TILE_H};
  cl_int err;

  const unsigned BATCH_SIZE = 510;
  const int zero            = 0;

  err = clSetKernelArg (ocl_compute_kernel (0), 2, sizeof (cl_mem),
                        &ocl_changed_buffer);
  check (err, "Failed to set kernel arg 2 (opt5)");

  unsigned total_it = 0;
  monitoring_start (easypap_gpu_lane (0));

  for (unsigned it = 1; it <= nb_iter; it += BATCH_SIZE) {
    err = clEnqueueFillBuffer (ocl_queue (0), ocl_changed_buffer, &zero,
                               sizeof (int), 0, sizeof (int), 0, NULL, NULL);

    unsigned remaining = nb_iter - it + 1;
    unsigned max_k     = (remaining + 5) / 6;
    if (max_k > (BATCH_SIZE / 6))
      max_k = BATCH_SIZE / 6;
    if (max_k == 0)
      break;

    err = clSetKernelArg (ocl_compute_kernel (0), 0, sizeof (cl_mem),
                          &ocl_cur_buffer (0));
    err |= clSetKernelArg (ocl_compute_kernel (0), 1, sizeof (cl_mem),
                           &ocl_next_buffer (0));
    check (err, "Failed to set kernel args 0-1 (opt5)");

    for (unsigned k = 0; k < max_k; k++) {
      total_it += 6;

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
    check (err, "Failed to read changed flag (opt5)");

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

void ssandPile_refresh_img_ocl_opt5 (void)
{
  ssandPile_refresh_img_ocl ();
}

#endif