#ifndef MEASUREMENTS_H
#define MEASUREMENTS_H

#include <vector>
#include "lattice.h"

inline double sq(double x) {
  return x*x;
}

inline double measure_E(const int* cfg, const latt_shape* shape) {
  double E = 0;
  for (int x = 0; x < shape->vol; ++x) {
    const double hx = cfg[x] / 2.0;
    for (int i = 0; i < ND; ++i) {
      const int x_fwd = shift_site_idx(x, 1, i, shape);
      const double hy = cfg[x_fwd] / 2.0;
      E += sq(hx - hy);
    }
  }
  return E;
}

inline double measure_M(const int* cfg, const latt_shape* shape) {
  double M = 0;
  for (int x = 0; x < shape->vol; ++x) {
    int tot_coord = 0;
    for (int i = 0; i < ND; ++i) {
      tot_coord += compute_comp(x, i, shape);
    }
    const int parity = 1 - 2*(tot_coord % 2);
    const double hx = cfg[x] / 2.0;
    M += parity * hx;
  }
  return M;
}

inline double measure_MT(const int* cfg, const latt_shape* shape) {
  double h_bar = 0;
  for (int x = 0; x < shape->vol; ++x) {
    const double hx = cfg[x] / 2.0;
    h_bar += hx / shape->vol;
  }
  double MT = 0;
  for (int x = 0; x < shape->vol; ++x) {
    int tot_coord = 0;
    for (int i = 0; i < ND; ++i) {
      tot_coord += compute_comp(x, i, shape);
    }
    const int parity = 1 - 2*(tot_coord % 2);
    const double hx = cfg[x] / 2.0;
    MT += parity * sq(hx - h_bar);
  }
  return MT;
}

inline std::vector<double> measure_Cl(const int* cfg, const latt_shape* shape) {
  std::vector<double> Cl(shape->dims[ND-1]);
  const int mu = ND-1;
  for (int x = 0; x < shape->vol; ++x) {
    const int x_mu = shift_site_idx(x, 1, mu, shape);
    const double lx_mu = (cfg[x_mu] - cfg[x]) / 2.0;
    for (int dt = 0; dt < Cl.size(); ++dt) {
      const int y = shift_site_idx(x, dt, ND-1, shape);
      const int y_mu = shift_site_idx(y, 1, mu, shape);
      const double ly_mu = (cfg[y_mu] - cfg[y]) / 2.0;
      Cl[dt] += (lx_mu * ly_mu) / shape->vol;
    }
  }
  return Cl;
}

inline std::vector<double> measure_Cl_mom(const int* cfg, const latt_shape* shape) {
  std::vector<double> Cl_mom(shape->dims[ND-1]);
  const int mu = 0;
  for (int x = 0; x < shape->vol; ++x) {
    const int x_mu = shift_site_idx(x, 1, mu, shape);
    const double lx_mu = (cfg[x_mu] - cfg[x]) / 2.0;
    const int t = compute_comp(x, ND-1, shape);
    Cl_mom[t] += lx_mu;
  }
  return Cl_mom;
}

inline double compute_Ox(const int* cfg, int x, const latt_shape* shape) {
  const double h = cfg[x] / 2.0;
  int x1 = shift_site_idx(x, 1, 0, shape);
  int x2 = shift_site_idx(x, 1, 1, shape);
  int x3 = shift_site_idx(x, 1, 2, shape);
  const double h1 = cfg[x1] / 2.0;
  const double h2 = cfg[x2] / 2.0;
  const double h3 = cfg[x3] / 2.0;
  int x12 = shift_site_idx(x1, 1, 1, shape);
  int x23 = shift_site_idx(x2, 1, 2, shape);
  int x13 = shift_site_idx(x1, 1, 2, shape);
  const double h12 = cfg[x12] / 2.0;
  const double h23 = cfg[x23] / 2.0;
  const double h13 = cfg[x13] / 2.0;
  int x123 = shift_site_idx(x12, 1, 2, shape);
  const double h123 = cfg[x123] / 2.0;
  const double h_bar = (
      h + h1 + h2 + h3 + h12 + h23 + h13 + h123) / 8.0;
  return (
      sq(h - h_bar) + sq(h12 - h_bar)
      + sq(h23 - h_bar) + sq(h13 - h_bar)
      - sq(h1 - h_bar) - sq(h2 - h_bar)
      - sq(h3 - h_bar) - sq(h123 - h_bar) );
}

inline double measure_MC(const int* cfg, const latt_shape* shape) {
  double MC = 0;
  for (int x = 0; x < shape->vol; ++x) {
    int tot_coord = 0;
    for (int i = 0; i < ND; ++i) {
      tot_coord += compute_comp(x, i, shape);
    }
    const int parity = 1 - 2*(tot_coord % 2);
    const double Ox = compute_Ox(cfg, x, shape);
    MC += parity * Ox;
  }
  return MC;
}


#endif
