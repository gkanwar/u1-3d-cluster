#ifndef LATT_OPS_H
#define LATT_OPS_H

#include <vector>

#include "lattice.h"
#include "util.h"

std::vector<int> make_init_cfg(const latt_shape* shape) {
  if (!shape->staggered) {
    return std::vector<int>(shape->vol, 0);
  }
  std::vector<int> cfg(shape->vol);
  for (int idx = 0; idx < shape->vol; ++idx) {
    int tot_coord = 0;
    for (int i = 0; i < ND; ++i) {
      tot_coord += compute_comp(idx, i, shape);
    }
    cfg[idx] = tot_coord % 2;
  }
  return cfg;
}
std::vector<cfg_t> make_init_hx(const latt_shape* shape) {
  auto cfg = make_init_cfg(shape);
  std::vector<cfg_t> hx(cfg.size());
  for (unsigned i = 0; i < cfg.size(); ++i) {
    hx[i] = cfg[i] / 2.0;
  }
  return hx;
}

inline void rezero_cfg(std::vector<cfg_t>& cfg) {
  cfg_t x = cfg[0];
  for (unsigned i = 0; i < cfg.size(); ++i) {
    cfg[i] -= x;
  }
}


#endif
