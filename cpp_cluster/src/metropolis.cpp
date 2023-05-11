#include "metropolis.h"

#include <random>
#include "lattice.h"
#include "util.h"

static std::bernoulli_distribution binary_dist;
static std::uniform_real_distribution<double> unif_dist =
    std::uniform_real_distribution<double>(0.0, 1.0);

inline static int offset_dist(my_rand &rng) {
  return 2*binary_dist(rng)-1;
}

float_t metropolis_update(
    cfg_t* cfg, float_t e2, my_rand &rng, const latt_shape* shape) {
  float_t acc = 0.0;
  
  for (int x = 0; x < shape->vol; ++x) {
    // const int cfg_x = cfg[x];
    // const int cfg_x_p = cfg_x + 2*offset_dist(rng);
    // const double hx = cfg_x / 2.0;
    // const double hx_p = cfg_x_p / 2.0;
    const float_t hx = cfg[x];
    const float_t hx_p = hx + offset_dist(rng);
    // double dS = 0.0;
    float_t neighbors = 0.0;
    for (int i = 0; i < ND; ++i) {
      const auto [x_fwd, sx_fwd] = shift_fwd(x, i, shape);
      const auto [x_bwd, sx_bwd] = shift_bwd(x, i, shape);
      const double hx_fwd = sx_fwd * cfg[x_fwd];
      const double hx_bwd = sx_bwd * cfg[x_bwd];
      neighbors += hx_fwd + hx_bwd;
      // dS += (e2/2) * (
      //     sq(hx_p - hx_fwd) + sq(hx_p - hx_bwd)
      //     - sq(hx - hx_fwd) - sq(hx - hx_bwd) );
    }
    const float_t dS = e2 * (ND*(sq(hx_p) - sq(hx)) + (hx - hx_p)*neighbors);
    if (unif_dist(rng) < exp(-dS)) {
      cfg[x] = hx_p;
      acc += 1;
    }
  }
  return acc / shape->vol;
}

inline int shift_fwd_2(int idx, int ax, const latt_shape* shape) {
  int full_block_idx = idx - (idx % shape->blocks[ax]);
  int new_idx = ((idx + shape->strides[ax]) % shape->blocks[ax]) + full_block_idx;
  return new_idx;
}
inline int shift_bwd_2(int idx, int ax, const latt_shape* shape) {
  int full_block_idx = idx - (idx % shape->blocks[ax]);
  int new_idx = ((idx + (shape->dims[ax]-1)*shape->strides[ax]) % shape->blocks[ax]) + full_block_idx;
  return new_idx;
}
float_t metropolis_update_with_wloop(
    cfg_t* cfg, float_t e2, my_rand &rng, const latt_shape* shape,
    const WloopShape& wloop) {
  float_t acc = 0;
  for (int x = 0; x < shape->vol; ++x) {
    const float_t hx = cfg[x];
    const float_t hx_p = hx + offset_dist(rng);
    float_t neighbors = 0.0;
    for (int i = 0; i < ND; ++i) {
      /*
      const auto [x_fwd, sx_fwd] = shift_site_idx(x, 1, i, shape);
      const auto [x_bwd, sx_bwd] = shift_site_idx(x, -1, i, shape);
      float_t hx_fwd = sx_fwd * cfg[x_fwd];
      float_t hx_bwd = sx_bwd * cfg[x_bwd];
      */
      /// FORNOW
      float_t hx_fwd = cfg[shift_fwd_2(x, i, shape)];
      float_t hx_bwd = cfg[shift_bwd_2(x, i, shape)];
      // dislocations from Wilson loop
      const int x0 = compute_comp(x, 0, shape);
      const int x1 = compute_comp(x, 1, shape);
      const int x2 = compute_comp(x, 2, shape);
      if (i == 1 && (x0 < wloop.x || (x0 == wloop.x && x2 < wloop.t))) {
        if (x1 == 0) {
          hx_fwd += 1;
        }
        else if (x1 == 1) {
          hx_bwd -= 1;
        }
      }
      neighbors += hx_fwd + hx_bwd;
    }
    const float_t dS = e2 * (ND*(sq(hx_p) - sq(hx)) + (hx - hx_p)*neighbors);
    if (unif_dist(rng) < exp(-dS)) {
      cfg[x] = hx_p;
      acc += 1;
    }
  }
  return acc / shape->vol;
}
