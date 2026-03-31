#include "kernel/ocl/common.cl"

__attribute__((reqd_work_group_size(TILE_W, TILE_H, 1)))
__kernel void ssandPile_ocl (__global unsigned *in, __global unsigned *out, __global int *changed)
{
  // We try to eliminate bank conflicts
  #define ROW_STRIDE (TILE_W + 1)
  
  __local unsigned tile[(TILE_H + 2) * ROW_STRIDE];

  int lx = get_local_id(0),  ly = get_local_id(1);
  int gx = get_global_id(0), gy = get_global_id(1);

  tile[(ly + 1) * ROW_STRIDE + (lx + 1)] = in[gy * DIM + gx];

  // Load halo
  if (lx == 0) {
      tile[(ly + 1) * ROW_STRIDE + 0] = (gx > 0) ? in[gy * DIM + (gx - 1)] : 0;
  } else if (lx == TILE_W - 1) {
      tile[(ly + 1) * ROW_STRIDE + TILE_W + 1] = (gx < DIM - 1) ? in[gy * DIM + (gx + 1)] : 0;
  }

  if (ly == 0) {
      tile[0 * ROW_STRIDE + (lx + 1)] = (gy > 0) ? in[(gy - 1) * DIM + gx] : 0;
  } else if (ly == TILE_H - 1) {
      tile[(TILE_H + 1) * ROW_STRIDE + (lx + 1)] = (gy < DIM - 1) ? in[(gy + 1) * DIM + gx] : 0;
  }

  barrier(CLK_LOCAL_MEM_FENCE);   

  int my_changed = 0;

  if (gx > 0 && gx < DIM - 1 && gy > 0 && gy < DIM - 1) {
    unsigned center = tile[(ly + 1) * ROW_STRIDE + (lx + 1)];
    unsigned res = (center & 3)
      + (tile[(ly + 0) * ROW_STRIDE + (lx + 1)] >> 2)   /* up    */
      + (tile[(ly + 2) * ROW_STRIDE + (lx + 1)] >> 2)   /* down  */
      + (tile[(ly + 1) * ROW_STRIDE + (lx + 0)] >> 2)   /* left  */
      + (tile[(ly + 1) * ROW_STRIDE + (lx + 2)] >> 2);  /* right */

    out[gy * DIM + gx] = res;
    my_changed = (res != center);
  }

  // Parallel reduction
  __local int lflags[TILE_W * TILE_H];
  int tid = ly * TILE_W + lx;
  lflags[tid] = my_changed;

  barrier(CLK_LOCAL_MEM_FENCE); 

  for (int s = (TILE_W * TILE_H) >> 1; s > 0; s >>= 1) {
      if (tid < s) {
          lflags[tid] |= lflags[tid + s];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tid == 0 && lflags[0])
      atomic_or(changed, 1);       
}

__kernel void ssandPile_ocl_opt3(__global unsigned *in,
                                 __global unsigned *out,
                                 __global int      *changed)
{
    #define MARGIN 4
    #define TRUE_W (TILE_W + 2 * MARGIN)
    #define IN_W TRUE_W
    #define IN_H (TILE_H + 2 * MARGIN)
    
    __local unsigned tile0[IN_H * IN_W];
    __local unsigned tile1[IN_H * IN_W];
    __local int local_changed;
    
    int lx = get_local_id(0), ly = get_local_id(1);
    int tid = ly * TILE_W + lx;
    int wg_size = TILE_W * TILE_H;
    int gx = get_global_id(0), gy = get_global_id(1);
    
    int base_gx = get_group_id(0) * TILE_W;
    int base_gy = get_group_id(1) * TILE_H;
    
    int total_cells = IN_H * IN_W;
    
    if (tid == 0) local_changed = 0;

    // Cooperative load of the (TILE_W+8)x(TILE_H+8) block
    for (int i = tid; i < total_cells; i += wg_size) {
        int tx = i % IN_W;
        int ty = i / IN_W;
        
        int g_x = base_gx - MARGIN + tx;
        int g_y = base_gy - MARGIN + ty;
        
        g_x = clamp(g_x, 0, DIM - 1);
        g_y = clamp(g_y, 0, DIM - 1);
        
        tile0[i] = in[g_y * DIM + g_x];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    // Iteration 1 (valid : 1..TRUE_W-2, 1..IN_H-2)
    for (int i = tid; i < total_cells; i += wg_size) {
        int tx = i % IN_W;
        int ty = i / IN_W;
        if (tx >= 1 && tx < IN_W - 1 && ty >= 1 && ty < IN_H - 1) {
            unsigned c = tile0[i];
            tile1[i] = (c & 3)
                + (tile0[i - IN_W] >> 2)
                + (tile0[i + IN_W] >> 2)
                + (tile0[i - 1] >> 2)
                + (tile0[i + 1] >> 2);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Iteration 2 (valid : 2..TRUE_W-3, 2..IN_H-3)
    for (int i = tid; i < total_cells; i += wg_size) {
        int tx = i % IN_W;
        int ty = i / IN_W;
        if (tx >= 2 && tx < IN_W - 2 && ty >= 2 && ty < IN_H - 2) {
            unsigned c = tile1[i];
            tile0[i] = (c & 3)
                + (tile1[i - IN_W] >> 2)
                + (tile1[i + IN_W] >> 2)
                + (tile1[i - 1] >> 2)
                + (tile1[i + 1] >> 2);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Iteration 3 (valid : 3..TRUE_W-4, 3..IN_H-4)
    for (int i = tid; i < total_cells; i += wg_size) {
        int tx = i % IN_W;
        int ty = i / IN_W;
        if (tx >= 3 && tx < IN_W - 3 && ty >= 3 && ty < IN_H - 3) {
            unsigned c = tile0[i];
            tile1[i] = (c & 3)
                + (tile0[i - IN_W] >> 2)
                + (tile0[i + IN_W] >> 2)
                + (tile0[i - 1] >> 2)
                + (tile0[i + 1] >> 2);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Iteration 4: write directly to global memory
    int out_i = (ly + MARGIN) * IN_W + (lx + MARGIN);
    
    unsigned center = tile1[out_i];
    unsigned res = (center & 3)
        + (tile1[out_i - IN_W] >> 2)
        + (tile1[out_i + IN_W] >> 2)
        + (tile1[out_i - 1] >> 2)
        + (tile1[out_i + 1] >> 2);
    
    if (gx > 0 && gx < DIM - 1 && gy > 0 && gy < DIM - 1) {
        out[gy * DIM + gx] = res;
        if (res != center)
            atomic_or(&local_changed, 1);
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (tid == 0 && local_changed)
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
