#include "kernel/ocl/common.cl"

__kernel void ssandPile_ocl (__global unsigned *in, __global unsigned *out, __global int *changed)
{
  // Shared memory for the tile + 1 pixel border to the right and left
  __local unsigned tile [(TILE_W + 2) * (TILE_H + 2)];
  __local int local_changed;

  int lx = get_local_id (0);
  int ly = get_local_id (1);
  int gx = get_global_id (0);
  int gy = get_global_id (1);

  // Initialize local flag
  if (lx == 0 && ly == 0)
    local_changed = 0;

  // Cooperative loading of the tile into local memory
  // Each thread loads its own cell + helps with the border
  int row_size = TILE_W + 2;
  
  // (lx+1, ly+1) are the right local coordinates of the current cell
  tile[(ly + 1) * row_size + (lx + 1)] = in[gy * DIM + gx];

  // Borders (only threads on the edges of the workgroup load these)
  if (lx == 0 && gx > 0) 
      tile[(ly + 1) * row_size + 0] = in[gy * DIM + (gx - 1)];
  if (lx == TILE_W - 1 && gx < DIM - 1) 
      tile[(ly + 1) * row_size + (TILE_W + 1)] = in[gy * DIM + (gx + 1)];
  if (ly == 0 && gy > 0) 
      tile[0 * row_size + (lx + 1)] = in[(gy - 1) * DIM + gx];
  if (ly == TILE_H - 1 && gy < DIM - 1) 
      tile[(TILE_H + 1) * row_size + (lx + 1)] = in[(gy + 1) * DIM + gx];

  barrier (CLK_LOCAL_MEM_FENCE);

  // Compute stencil using local memory
  if (gx > 0 && gx < DIM - 1 && gy > 0 && gy < DIM - 1) {
    unsigned center = tile[(ly + 1) * row_size + (lx + 1)];
    
    unsigned res = (center % 4) +
                   (tile[(ly + 0) * row_size + (lx + 1)] >> 2) + // up
                   (tile[(ly + 2) * row_size + (lx + 1)] >> 2) + // down
                   (tile[(ly + 1) * row_size + (lx + 0)] >> 2) + // left
                   (tile[(ly + 1) * row_size + (lx + 2)] >> 2);  // right

    out[gy * DIM + gx] = res;

    // Local change detection
    if (res != center)
      atomic_or(&local_changed, 1);
  }

  // Single global atomic update per workgroup
  barrier (CLK_LOCAL_MEM_FENCE);
  if (lx == 0 && ly == 0 && local_changed != 0)
    atomic_or (changed, 1);
}

#ifdef GL_BUFFER_SHARING

// DO NOT MODIFY: this kernel updates the OpenGL texture buffer
// This is a ssandPile-specific version (generic version is defined in
// common.cl)
__kernel void ssandPile_update_texture (__global unsigned *cur,
                                        __write_only image2d_t tex)
{
  int y      = get_global_id (1);
  int x      = get_global_id (0);
  int2 pos   = (int2)(x, y);
  unsigned c = cur [y * DIM + x];
  unsigned r = 0, v = 0, b = 0;

  if (c == 1)
    v = 255;
  else if (c == 2)
    b = 255;
  else if (c == 3)
    r = 255;
  else if (c == 4)
    r = v = b = 255;
  else if (c > 4)
    r = v = b = (2 * c);

  c = rgb (r, v, b);
  write_imagef (tex, pos, color_to_float4 (c));
}

#endif
