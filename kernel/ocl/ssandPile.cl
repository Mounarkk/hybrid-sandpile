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

__kernel void ssandPile_ocl_opt4(__global unsigned *in,
                                 __global unsigned *out,
                                 __global int      *changed)
{
    #define MARGIN4 5
    #define TRUE_W4 (TILE_W + 2 * MARGIN4)
    #define TRUE_H4 (TILE_H + 2 * MARGIN4)
    #define IN_W4 (TRUE_W4 | 1) 
    #define IN_H4 TRUE_H4
    
    __local unsigned tile0[IN_H4 * IN_W4];
    __local unsigned tile1[IN_H4 * IN_W4];
    __local int local_changed;
    
    int lx = get_local_id(0), ly = get_local_id(1);
    int tid = ly * TILE_W + lx;
    int wg_size = TILE_W * TILE_H;
    int gx = get_global_id(0), gy = get_global_id(1);
    int base_gx = get_group_id(0) * TILE_W;
    int base_gy = get_group_id(1) * TILE_H;

    if (tid == 0) local_changed = 0;

    int total_cells = IN_H4 * IN_W4;

    // 3-pass mapping
    int i1 = tid, i2 = tid + wg_size, i3 = tid + 2 * wg_size;
    int tx1 = i1 % IN_W4, ty1 = i1 / IN_W4;
    int tx2 = i2 % IN_W4, ty2 = i2 / IN_W4;
    int tx3 = i3 % IN_W4, ty3 = i3 / IN_W4;

    // 1. STATIC LOAD (3 passes)
    {
        tile0[i1] = in[clamp(base_gy - MARGIN4 + ty1, 0, DIM - 1) * DIM + clamp(base_gx - MARGIN4 + tx1, 0, DIM - 1)];
        if (i2 < total_cells) 
            tile0[i2] = in[clamp(base_gy - MARGIN4 + ty2, 0, DIM - 1) * DIM + clamp(base_gx - MARGIN4 + tx2, 0, DIM - 1)];
        if (i3 < total_cells) 
            tile0[i3] = in[clamp(base_gy - MARGIN4 + ty3, 0, DIM - 1) * DIM + clamp(base_gx - MARGIN4 + tx3, 0, DIM - 1)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Iterations 1 to 4
    #define PASS_OPT4(step, t_in, t_out) \
    { \
        if (tx1 >= step && tx1 < IN_W4 - step && ty1 >= step && ty1 < IN_H4 - step) { \
            unsigned c = t_in[i1]; \
            t_out[i1] = (c & 3) + (t_in[i1 - IN_W4] >> 2) + (t_in[i1 + IN_W4] >> 2) + (t_in[i1 - 1] >> 2) + (t_in[i1 + 1] >> 2); \
        } \
        if (i2 < total_cells && tx2 >= step && tx2 < IN_W4 - step && ty2 >= step && ty2 < IN_H4 - step) { \
            unsigned c = t_in[i2]; \
            t_out[i2] = (c & 3) + (t_in[i2 - IN_W4] >> 2) + (t_in[i2 + IN_W4] >> 2) + (t_in[i2 - 1] >> 2) + (t_in[i2 + 1] >> 2); \
        } \
        if (i3 < total_cells && tx3 >= step && tx3 < IN_W4 - step && ty3 >= step && ty3 < IN_H4 - step) { \
            unsigned c = t_in[i3]; \
            t_out[i3] = (c & 3) + (t_in[i3 - IN_W4] >> 2) + (t_in[i3 + IN_W4] >> 2) + (t_in[i3 - 1] >> 2) + (t_in[i3 + 1] >> 2); \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    }

    PASS_OPT4(1, tile0, tile1);
    PASS_OPT4(2, tile1, tile0);
    PASS_OPT4(3, tile0, tile1);
    PASS_OPT4(4, tile1, tile0);

    // Final iteration (exact match for output tile)
    int i_final = (ly + MARGIN4) * IN_W4 + (lx + MARGIN4);
    unsigned center = tile0[i_final];
    unsigned res = (center & 3) + (tile0[i_final - IN_W4] >> 2) + (tile0[i_final + IN_W4] >> 2) + (tile0[i_final - 1] >> 2) + (tile0[i_final + 1] >> 2);
    
    if (gx > 0 && gx < DIM - 1 && gy > 0 && gy < DIM - 1) {
        out[gy * DIM + gx] = res;
        if (res != center) atomic_or(&local_changed, 1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid == 0 && local_changed) atomic_or(changed, 1);
}

__kernel void ssandPile_ocl_opt5(__global unsigned *in,
                                 __global unsigned *out,
                                 __global int      *changed)
{
    #define MARGIN5 6
    #define TRUE_W5 (TILE_W + 2 * MARGIN5)
    #define TRUE_H5 (TILE_H + 2 * MARGIN5)
    #define IN_W5 (TRUE_W5 | 1)  // 45: odd stride, bank-conflict free
    #define IN_H5 TRUE_H5

    __local unsigned tile0[IN_H5 * IN_W5];
    __local unsigned tile1[IN_H5 * IN_W5];
    __local int local_changed;

    int lx = get_local_id(0), ly = get_local_id(1);
    int tid = ly * TILE_W + lx;
    int wg_size = TILE_W * TILE_H;
    int gx = get_global_id(0), gy = get_global_id(1);
    int base_gx = get_group_id(0) * TILE_W;
    int base_gy = get_group_id(1) * TILE_H;

    if (tid == 0) local_changed = 0;

    int total_cells = IN_H5 * IN_W5;

    // 3-pass static mapping (45*28=1260 cells, 3*512=1536 >= 1260)
    int i1 = tid, i2 = tid + wg_size, i3 = tid + 2 * wg_size;
    int tx1 = i1 % IN_W5, ty1 = i1 / IN_W5;
    int tx2 = i2 % IN_W5, ty2 = i2 / IN_W5;
    int tx3 = i3 % IN_W5, ty3 = i3 / IN_W5;

    // STATIC LOAD
    tile0[i1] = in[clamp(base_gy - MARGIN5 + ty1, 0, DIM - 1) * DIM + clamp(base_gx - MARGIN5 + tx1, 0, DIM - 1)];
    if (i2 < total_cells)
        tile0[i2] = in[clamp(base_gy - MARGIN5 + ty2, 0, DIM - 1) * DIM + clamp(base_gx - MARGIN5 + tx2, 0, DIM - 1)];
    if (i3 < total_cells)
        tile0[i3] = in[clamp(base_gy - MARGIN5 + ty3, 0, DIM - 1) * DIM + clamp(base_gx - MARGIN5 + tx3, 0, DIM - 1)];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Iterations 1 to 5
    #define PASS_OPT5(step, t_in, t_out) \
    { \
        if (tx1 >= step && tx1 < IN_W5 - step && ty1 >= step && ty1 < IN_H5 - step) { \
            unsigned c = t_in[i1]; \
            t_out[i1] = (c & 3) + (t_in[i1 - IN_W5] >> 2) + (t_in[i1 + IN_W5] >> 2) + (t_in[i1 - 1] >> 2) + (t_in[i1 + 1] >> 2); \
        } \
        if (i2 < total_cells && tx2 >= step && tx2 < IN_W5 - step && ty2 >= step && ty2 < IN_H5 - step) { \
            unsigned c = t_in[i2]; \
            t_out[i2] = (c & 3) + (t_in[i2 - IN_W5] >> 2) + (t_in[i2 + IN_W5] >> 2) + (t_in[i2 - 1] >> 2) + (t_in[i2 + 1] >> 2); \
        } \
        if (i3 < total_cells && tx3 >= step && tx3 < IN_W5 - step && ty3 >= step && ty3 < IN_H5 - step) { \
            unsigned c = t_in[i3]; \
            t_out[i3] = (c & 3) + (t_in[i3 - IN_W5] >> 2) + (t_in[i3 + IN_W5] >> 2) + (t_in[i3 - 1] >> 2) + (t_in[i3 + 1] >> 2); \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    }

    PASS_OPT5(1, tile0, tile1);
    PASS_OPT5(2, tile1, tile0);
    PASS_OPT5(3, tile0, tile1);
    PASS_OPT5(4, tile1, tile0);
    PASS_OPT5(5, tile0, tile1);

    // Final iteration (exact match for output tile)
    int i_final = (ly + MARGIN5) * IN_W5 + (lx + MARGIN5);
    unsigned center = tile1[i_final];
    unsigned res = (center & 3) + (tile1[i_final - IN_W5] >> 2) + (tile1[i_final + IN_W5] >> 2) + (tile1[i_final - 1] >> 2) + (tile1[i_final + 1] >> 2);

    if (gx > 0 && gx < DIM - 1 && gy > 0 && gy < DIM - 1) {
        out[gy * DIM + gx] = res;
        if (res != center) atomic_or(&local_changed, 1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid == 0 && local_changed) atomic_or(changed, 1);
}

__kernel void ssandPile_ocl_opt6(__global unsigned *in,
                                 __global unsigned *out,
                                 __global int      *changed)
{
    #define MARGIN6 8
    #define TRUE_W6 (TILE_W + 2 * MARGIN6)
    #define TRUE_H6 (TILE_H + 2 * MARGIN6)
    #define IN_W6 (TRUE_W6 | 1)  // odd stride, bank-conflict free
    #define IN_H6 TRUE_H6

    __local unsigned tile0[IN_H6 * IN_W6];
    __local unsigned tile1[IN_H6 * IN_W6];
    __local int local_changed;

    int lx = get_local_id(0), ly = get_local_id(1);
    int tid = ly * TILE_W + lx;
    int wg_size = TILE_W * TILE_H;
    int gx = get_global_id(0), gy = get_global_id(1);
    int base_gx = get_group_id(0) * TILE_W;
    int base_gy = get_group_id(1) * TILE_H;

    if (tid == 0) local_changed = 0;

    int total_cells = IN_H6 * IN_W6;

    // 4-pass static mapping (49*32=1568 cells, 4*512=2048 >= 1568)
    int i1 = tid, i2 = tid + wg_size, i3 = tid + 2 * wg_size, i4 = tid + 3 * wg_size;
    int tx1 = i1 % IN_W6, ty1 = i1 / IN_W6;
    int tx2 = i2 % IN_W6, ty2 = i2 / IN_W6;
    int tx3 = i3 % IN_W6, ty3 = i3 / IN_W6;
    int tx4 = i4 % IN_W6, ty4 = i4 / IN_W6;

    // STATIC LOAD (4 passes)
    tile0[i1] = in[clamp(base_gy - MARGIN6 + ty1, 0, DIM - 1) * DIM + clamp(base_gx - MARGIN6 + tx1, 0, DIM - 1)];
    if (i2 < total_cells)
        tile0[i2] = in[clamp(base_gy - MARGIN6 + ty2, 0, DIM - 1) * DIM + clamp(base_gx - MARGIN6 + tx2, 0, DIM - 1)];
    if (i3 < total_cells)
        tile0[i3] = in[clamp(base_gy - MARGIN6 + ty3, 0, DIM - 1) * DIM + clamp(base_gx - MARGIN6 + tx3, 0, DIM - 1)];
    if (i4 < total_cells)
        tile0[i4] = in[clamp(base_gy - MARGIN6 + ty4, 0, DIM - 1) * DIM + clamp(base_gx - MARGIN6 + tx4, 0, DIM - 1)];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Iterations 1 to 7
    #define PASS_OPT6(step, t_in, t_out) \
    { \
        if (tx1 >= step && tx1 < IN_W6 - step && ty1 >= step && ty1 < IN_H6 - step) { \
            unsigned c = t_in[i1]; \
            t_out[i1] = (c & 3) + (t_in[i1 - IN_W6] >> 2) + (t_in[i1 + IN_W6] >> 2) + (t_in[i1 - 1] >> 2) + (t_in[i1 + 1] >> 2); \
        } \
        if (i2 < total_cells && tx2 >= step && tx2 < IN_W6 - step && ty2 >= step && ty2 < IN_H6 - step) { \
            unsigned c = t_in[i2]; \
            t_out[i2] = (c & 3) + (t_in[i2 - IN_W6] >> 2) + (t_in[i2 + IN_W6] >> 2) + (t_in[i2 - 1] >> 2) + (t_in[i2 + 1] >> 2); \
        } \
        if (i3 < total_cells && tx3 >= step && tx3 < IN_W6 - step && ty3 >= step && ty3 < IN_H6 - step) { \
            unsigned c = t_in[i3]; \
            t_out[i3] = (c & 3) + (t_in[i3 - IN_W6] >> 2) + (t_in[i3 + IN_W6] >> 2) + (t_in[i3 - 1] >> 2) + (t_in[i3 + 1] >> 2); \
        } \
        if (i4 < total_cells && tx4 >= step && tx4 < IN_W6 - step && ty4 >= step && ty4 < IN_H6 - step) { \
            unsigned c = t_in[i4]; \
            t_out[i4] = (c & 3) + (t_in[i4 - IN_W6] >> 2) + (t_in[i4 + IN_W6] >> 2) + (t_in[i4 - 1] >> 2) + (t_in[i4 + 1] >> 2); \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    }

    PASS_OPT6(1, tile0, tile1);
    PASS_OPT6(2, tile1, tile0);
    PASS_OPT6(3, tile0, tile1);
    PASS_OPT6(4, tile1, tile0);
    PASS_OPT6(5, tile0, tile1);
    PASS_OPT6(6, tile1, tile0);
    PASS_OPT6(7, tile0, tile1);

    // Final iteration 8: write directly to global memory
    int i_final = (ly + MARGIN6) * IN_W6 + (lx + MARGIN6);
    unsigned center = tile1[i_final];
    unsigned res = (center & 3) + (tile1[i_final - IN_W6] >> 2) + (tile1[i_final + IN_W6] >> 2) + (tile1[i_final - 1] >> 2) + (tile1[i_final + 1] >> 2);

    if (gx > 0 && gx < DIM - 1 && gy > 0 && gy < DIM - 1) {
        out[gy * DIM + gx] = res;
        if (res != center) atomic_or(&local_changed, 1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid == 0 && local_changed) atomic_or(changed, 1);
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
