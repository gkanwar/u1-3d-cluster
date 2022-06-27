/// We would like to run the cluster algorithm much faster, so here is a C++
/// version. Most of the bottleneck in Python likely has to do with lack of
/// fused operations and data locality.

#include <algorithm>
#include <array>
#include <iostream>
#include <random>
#include <vector>
#include "cluster.h"

constexpr int SHIFT_FREQ = 100;

using namespace std;

vector<int> make_init_cfg(const latt_shape* shape) {
  vector<int> cfg(shape->vol);
  for (int idx = 0; idx < shape->vol; ++idx) {
    int tot_coord = 0;
    for (int i = 0; i < ND; ++i) {
      tot_coord += compute_comp(idx, i, shape);
    }
    cfg[idx] = tot_coord % 2;
  }
  return cfg;
}

void rezero_cfg(vector<int>& cfg) {
  int x = cfg[0];
  for (int i = 0; i < cfg.size(); ++i) {
    cfg[i] -= x;
  }
}

void run_cluster(double e2, int n_iter, minstd_rand& rng, const latt_shape* shape) {
  vector<int> cfg = make_init_cfg(shape);

  // internal cluster storage
  vector<int> bonds(shape->vol*ND);
  vector<int> labels(shape->vol);

  uniform_int_distribution<int> site_dist =
      uniform_int_distribution<int>(0, shape->vol);

  for (int i = 0; i < n_iter; ++i) {
    const int site = site_dist(rng);
    const int cfg_star = cfg[site];
    const double h_star = cfg_star / 2.0;
    sample_bonds(cfg.data(), bonds.data(), e2, h_star, rng, shape);
    std::fill(labels.begin(), labels.end(), 0);
    flip_clusters(bonds.data(), cfg.data(), labels.data(), cfg_star, rng, shape);

    if ((i+1) % SHIFT_FREQ == 0) {
      cout << (i+1) << " / " << n_iter << "\n";
      rezero_cfg(cfg);
    }

    /// TODO: measurements

    /// DEBUG
    // for (int i = 0; i < shape->dims[1]; ++i) {
    //   for (int j = 0; j < shape->dims[2]; ++j) {
    //     cout << labels[i*shape->strides[1] + j] << " ";
    //   }
    //   cout << "\n";
    // }
    // cout << "\n";
    // for (int i = 0; i < shape->dims[1]; ++i) {
    //   for (int j = 0; j < shape->dims[2]; ++j) {
    //     cout << cfg[i*shape->strides[1] + j] << " ";
    //   }
    //   cout << "\n";
    // }
    // cout << "\n";
    
  }

}

int main(int argc, char** argv) {
  const double e2 = 0.5;
  const int seed = 1234;
  const int L = 32;
  array<int,3> dims = { L, L, L };
  latt_shape shape = make_latt_shape(&dims.front());

  minstd_rand rng(1235);

  run_cluster(e2, 1000, rng, &shape);

  return 0;
}
