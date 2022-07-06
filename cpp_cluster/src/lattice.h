#ifndef LATTICE_H
#define LATTICE_H

#include <cassert>

/// Lattice geometry
#define ND 3
typedef struct {
  // primary shape
  int dims[ND];
  // derived quantities useful for index manipulation
  int strides[ND];
  int blocks[ND];
  int vol;
} latt_shape;

inline latt_shape make_latt_shape(int* dims) {
  latt_shape shape = {.vol = 1};
  for (int i = ND-1; i >= 0; --i) {
    shape.dims[i] = dims[i];
    shape.strides[i] = shape.vol;
    shape.vol *= shape.dims[i];
    shape.blocks[i] = shape.vol;
  }
  return shape;
}

inline int compute_comp(int idx, int ax, const latt_shape* shape) {
  return (idx % shape->blocks[ax]) / shape->strides[ax];
}

inline int shift_site_idx(int idx, int diff, int ax, const latt_shape* shape) {
  if (diff < 0) {
    diff += shape->dims[ax];
  }
  int full_block_idx = idx - (idx % shape->blocks[ax]);
  int new_idx = ((idx + diff*shape->strides[ax]) % shape->blocks[ax]) + full_block_idx;
  for (int i = 0; i < ND; ++i) {
    if (i == ax) {
      assert(compute_comp(new_idx, i, shape) ==
             (compute_comp(idx, i, shape) + diff) % shape->dims[i]);
    }
    else {
      assert(compute_comp(new_idx, i, shape) ==
             compute_comp(idx, i, shape));
    }
  }
  return new_idx;
}

inline int get_bond_idx(int site_idx, int mu, const latt_shape* shape) {
  return site_idx*ND + mu;
}

inline int get_site_idx(int x, int y, int z, const latt_shape* shape) {
  return x*shape->strides[0] + y*shape->strides[1] + z;
}

#endif
