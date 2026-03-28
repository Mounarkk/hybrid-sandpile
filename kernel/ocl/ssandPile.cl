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

__kernel void ssandPile_ocl_opt(__global unsigned *in,
                                __global unsigned *out,
                                __global int *changed)
{
  __local unsigned tile[(TILE_W + 2) * (TILE_H + 2)];
  __local int local_changed;

  int lx = get_local_id(0);
  int ly = get_local_id(1);
  int gx = get_global_id(0);
  int gy = get_global_id(1);

  int lsize_x = get_local_size(0);
  int lsize_y = get_local_size(1);

  int tid     = ly * lsize_x + lx;
  int wg_size = lsize_x * lsize_y;

  int row_size  = TILE_W + 2;
  int tile_size = (TILE_W + 2) * (TILE_H + 2);

  // Use workgroup origin as shared base for all threads
  int base_gx = get_group_id(0) * TILE_W;
  int base_gy = get_group_id(1) * TILE_H;

  if (tid == 0)
    local_changed = 0;

  // Cooperative tile loading — all threads share the same base now
  for (int i = tid; i < tile_size; i += wg_size) {
    int tx = i % row_size;
    int ty = i / row_size;

    // Clamp to [0, DIM-1] for boundary cells 
    int gx_load = clamp(base_gx + tx - 1, 0, DIM - 1);
    int gy_load = clamp(base_gy + ty - 1, 0, DIM - 1);

    tile[i] = in[gy_load * DIM + gx_load];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  int changed_flag = 0;

  if (gx > 0 && gx < DIM - 1 && gy > 0 && gy < DIM - 1) {
    unsigned center = tile[(ly + 1) * row_size + (lx + 1)];

    unsigned up    = tile[(ly + 0) * row_size + (lx + 1)] >> 2;
    unsigned down  = tile[(ly + 2) * row_size + (lx + 1)] >> 2;
    unsigned left  = tile[(ly + 1) * row_size + (lx + 0)] >> 2;
    unsigned right = tile[(ly + 1) * row_size + (lx + 2)] >> 2;

    unsigned res = (center & 3) + up + down + left + right;

    out[gy * DIM + gx] = res;

    changed_flag = (res != center);
  }

  // Parallel tree reduction instead of serial loop by thread 0
  // Reuse tile memory (as int) to avoid an extra __local array
  __local int lflags[TILE_W * TILE_H];
  lflags[tid] = changed_flag;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int stride = wg_size >> 1; stride > 0; stride >>= 1) {
    if (tid < stride)
      lflags[tid] |= lflags[tid + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tid == 0 && lflags[0])
    atomic_or(changed, 1);
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
