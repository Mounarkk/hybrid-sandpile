#include "sandPile_common.h"
#include <sys/mman.h>
#include <unistd.h>

// Global variables
TYPE * /*restrict*/ TABLE = NULL;
int in  = 0;
int out = 1;

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
