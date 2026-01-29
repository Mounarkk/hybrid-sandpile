#ifndef SANDPILE_COMMON_H
#define SANDPILE_COMMON_H

#include "easypap.h"
#include <stdbool.h>

// Type definition
typedef unsigned int TYPE;

// Global variables
extern TYPE * /*restrict */ TABLE;
extern int in;
extern int out;

// Table access macros and functions
static inline TYPE *atable_cell (TYPE */* restrict */ i, int y, int x)
{
  return i + y * DIM + x;
}

#define atable(y, x) (*atable_cell (TABLE, (y), (x)))

static inline TYPE *table_cell (TYPE * /* restrict */ i, int step, int y, int x)
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

#endif // SANDPILE_COMMON_H
