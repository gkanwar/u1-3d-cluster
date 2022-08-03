#include "metropolis.h"

#include <random>
#include "lattice.h"
#include "util.h"

static std::uniform_int_distribution<int> offset_dist =
    std::uniform_int_distribution<int>(-1, 1);
static std::uniform_real_distribution<double> unif_dist =
    std::uniform_real_distribution<double>(0.0, 1.0);

double metropolis_update(
    int* cfg, double e2, my_rand &rng, const latt_shape* shape) {
  double acc = 0.0;
  for (int x = 0; x < shape->vol; ++x) {
    const int cfg_x = cfg[x];
    const int cfg_x_p = cfg_x + 2*offset_dist(rng);
    const double hx = cfg_x / 2.0;
    const double hx_p = cfg_x_p / 2.0;
    double dS = 0.0;
    for (int i = 0; i < ND; ++i) {
      const auto [x_fwd, sx_fwd] = shift_site_idx(x, 1, i, shape);
      const auto [x_bwd, sx_bwd] = shift_site_idx(x, -1, i, shape);
      const double hx_fwd = sx_fwd * cfg[x_fwd] / 2.0;
      const double hx_bwd = sx_bwd * cfg[x_bwd] / 2.0;
      dS += (e2/2) * (
          sq(hx_p - hx_fwd) + sq(hx_p - hx_bwd)
          - sq(hx - hx_fwd) - sq(hx - hx_bwd) );
    }
    if (unif_dist(rng) < exp(-dS)) {
      cfg[x] = cfg_x_p;
      acc += 1.0 / shape->vol;
    }
  }
  return acc;
}
