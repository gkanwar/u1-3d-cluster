#include "cluster.h"

#include <cassert>
#include <stack>
#include <vector>
#include <random>

#define LSTR(x) #x
#define STR(x) LSTR(x)

static std::uniform_int_distribution<int> bit_dist =
    std::uniform_int_distribution<int>(0, 1);
static std::uniform_real_distribution<double> unif_dist =
    std::uniform_real_distribution<double>(0.0, 1.0);
// inline int bit_dist(my_rand &rng) {
//   return rng() % 2;
// }
// inline double unif_dist(my_rand &rng) {
//   return (rng() - rng.min()) / ((double) (rng.max() - rng.min() + 1));
// }

static bool sample_bond(
    int cfg_x, int cfg_y, int cfg_star, double e2, my_rand& rng) {
  const double h_star = cfg_star / 2.0;
  const double hx = cfg_x / 2.0;
  const double hy = cfg_y / 2.0;
  const double R = exp(-2.0*e2*(h_star - hx)*(h_star - hy));
  return unif_dist(rng) < (1.0 - R);
}

void flip_clusters(
    int* cfg, int cfg_star, double e2,
    my_rand &rng, const latt_shape* shape) {
  int cur_label = 0;
  std::vector<int> labels(shape->vol, 0);
  std::vector<int> flips(shape->vol, 0);
  std::stack<int> queue;
  for (int x = 0; x < shape->vol; ++x) {
    if (labels[x] != 0) continue;
    int cur_flip = 2*bit_dist(rng) - 1;
    labels[x] = ++cur_label;
    flips[x] = cur_flip;
    queue.push(x);
    while (queue.size() > 0) {
      int y = queue.top();
      queue.pop();
      for (int i = 0; i < ND; ++i) {
        int y_fwd = shift_site_idx(y, 1, i, shape);
        int y_bwd = shift_site_idx(y, -1, i, shape);
        // lazy bonds:
        if (labels[y_fwd] == 0) {
          bool bond_fwd = sample_bond(cfg[y], cfg[y_fwd], cfg_star, e2, rng);
          // bool bond_fwd = bonds[get_bond_idx(y, i, shape)];
          if (bond_fwd) {
            flips[y_fwd] = cur_flip;
            labels[y_fwd] = cur_label;
            queue.push(y_fwd);
          }
        }
        if (labels[y_bwd] == 0) {
          bool bond_bwd = sample_bond(cfg[y], cfg[y_bwd], cfg_star, e2, rng);
          // bool bond_bwd = bonds[get_bond_idx(y_bwd, i, shape)];
          if (bond_bwd) {
            flips[y_bwd] = cur_flip;
            labels[y_bwd] = cur_label;
            queue.push(y_bwd);
          }
        }
      }
    }
  }

  // do the flips
  for (int x = 0; x < shape->vol; ++x) {
    cfg[x] = cfg_star + flips[x]*(cfg_star - cfg[x]);
  }

  // CHECK:
  // for (int x = 0; x < shape->vol; ++x) {
  //   for (int i = 0; i < ND; ++i) {
  //     if (bonds[get_bond_idx(x, i, shape)]) {
  //       int x_fwd = shift_site_idx(x, 1, i, shape);
  //       assert(labels[x] == labels[x_fwd]);
  //     }
  //   }
  // }
}

void sample_bonds(
    const int* cfg, int* bonds, double e2, double h_star,
    my_rand& rng, const latt_shape* shape) {
  for (int x = 0; x < shape->vol; ++x) {
    const double hx = cfg[x] / 2.0;
    for (int i = 0; i < ND; ++i) {
      const int x_fwd = shift_site_idx(x, 1, i, shape);
      const double hy = cfg[x_fwd] / 2.0;
      const double R = exp(-2.0*e2*(h_star - hx)*(h_star - hy));
      bonds[get_bond_idx(x, i, shape)] = (int)(unif_dist(rng) < (1.0 - R));
    }
  }
}
