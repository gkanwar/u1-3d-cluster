#ifndef LATTICE_H
#define LATTICE_H

#include <cassert>
#include <complex>

/// Lattice geometry
#define ND 3
typedef struct {
  // primary shape
  int dims[ND];
  // derived quantities useful for index manipulation
  int strides[ND];
  int blocks[ND];
  int vol;
  // c-periodicity in spatial dirs
  bool cper;
  // half-integer staggered vs unstaggered
  bool staggered;
} latt_shape;

inline latt_shape make_latt_shape(int* dims, bool cper, bool staggered) {
  latt_shape shape = {
    .dims = {0}, .strides = {0}, .blocks = {0},
    .vol = 1, .cper = cper, .staggered =  staggered
  };
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

inline int boundary_sign(int idx, int diff, int ax, const latt_shape* shape) {
  if (shape->cper && ax != (ND-1) &&
      (diff + compute_comp(idx, ax, shape) >= shape->dims[ax] ||
       diff + compute_comp(idx, ax, shape) < 0 )) {
    return -1;
  }
  else {
    return 1;
  }
}

inline std::pair<int,int> shift_site_idx(int idx, int diff, int ax, const latt_shape* shape) {
  int sign = boundary_sign(idx, diff, ax, shape);
  if (diff < 0) {
    diff += shape->dims[ax];
  }
  int full_block_idx = idx - (idx % shape->blocks[ax]);
  int new_idx = ((idx + diff*shape->strides[ax]) % shape->blocks[ax]) + full_block_idx;
  return std::make_pair(new_idx, sign);
}

inline int get_bond_idx(int site_idx, int mu, [[maybe_unused]] const latt_shape* shape) {
  return site_idx*ND + mu;
}

inline int get_site_idx(int x, int y, int z, const latt_shape* shape) {
  return x*shape->strides[0] + y*shape->strides[1] + z;
}

/// Misc util
using cdouble = std::complex<double>;
using namespace std::complex_literals;

#endif
