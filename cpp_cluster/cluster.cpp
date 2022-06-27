#include "cluster.h"

#include <cassert>
#include <stack>
#include <random>

#define LSTR(x) #x
#define STR(x) LSTR(x)

static std::uniform_int_distribution<int> bit_dist =
    std::uniform_int_distribution<int>(0, 1);
static std::uniform_real_distribution<double> unif_dist =
    std::uniform_real_distribution<double>(0.0, 1.0);

void flip_clusters(
    const int* bonds, int* cfg, int* labels, int cfg_star,
    std::minstd_rand &rng, const latt_shape* shape) {
  int cur_label = 0;
  std::stack<int> queue;
  for (int x = 0; x < shape->vol; ++x) {
    if (labels[x] != 0) continue;
    int cur_flip = bit_dist(rng);
    labels[x] = ++cur_label;
    if (cur_flip) {
      cfg[x] = 2*cfg_star - cfg[x];
    }
    queue.push(x);
    while (queue.size() > 0) {
      int y = queue.top();
      queue.pop();
      for (int i = 0; i < ND; ++i) {
        int y_fwd = shift_site_idx(y, 1, i, shape);
        int y_bwd = shift_site_idx(y, -1, i, shape);
        if (bonds[get_bond_idx(y, i, shape)] && labels[y_fwd] != cur_label) {
          assert(labels[y_fwd] == 0);
          labels[y_fwd] = cur_label;
          if (cur_flip) {
            cfg[y_fwd] = 2*cfg_star - cfg[y_fwd];
          }
          queue.push(y_fwd);
        }
        if (bonds[get_bond_idx(y_bwd, i, shape)] && labels[y_bwd] != cur_label) {
          assert(labels[y_bwd] == 0);
          labels[y_bwd] = cur_label;
          if (cur_flip) {
            cfg[y_bwd] = 2*cfg_star - cfg[y_bwd];
          }
          queue.push(y_bwd);
        }
      }
    }
  }

  // CHECK:
  for (int x = 0; x < shape->vol; ++x) {
    for (int i = 0; i < ND; ++i) {
      if (bonds[get_bond_idx(x, i, shape)]) {
        int x_fwd = shift_site_idx(x, 1, i, shape);
        assert(labels[x] == labels[x_fwd]);
      }
    }
  }
}

void sample_bonds(
    const int* cfg, int* bonds, double e2, double h_star,
    std::minstd_rand& rng, const latt_shape* shape) {
  for (int a = 0; a < shape->dims[0]; ++a) {
    for (int b = 0; b < shape->dims[1]; ++b) {
      for (int c = 0; c < shape->dims[2]; ++c) {
        const int x = get_site_idx(a, b, c, shape);
        const double hx = cfg[x] / 2.0;
        for (int i = 0; i < ND; ++i) {
          const int x_fwd = shift_site_idx(x, 1, i, shape);
          const double hy = cfg[x_fwd] / 2.0;
          const double R = exp(-2.0*e2*(h_star - hx)*(h_star - hy));
          const int bond = R < 1.0 && unif_dist(rng) < (1.0 - R);
          bonds[get_bond_idx(x, i, shape)] = bond;
        }
      }
    }
  }
}
