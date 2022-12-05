#ifndef LATT_OPS_H
#define LATT_OPS_H

#include <vector>

#include "lattice.h"

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

inline void rezero_cfg(std::vector<int>& cfg) {
  int x = cfg[0];
  for (unsigned i = 0; i < cfg.size(); ++i) {
    cfg[i] -= x;
  }
}


#endif
