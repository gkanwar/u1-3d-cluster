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
      const auto [x_fwd, sx_fwd] = shift_site_idx(x, 1, i, shape);
      const double hy = sx_fwd*cfg[x_fwd] / 2.0;
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

inline double measure_hsq(const int* cfg, const latt_shape* shape) {
  double hsq = 0.0;
  for (int x = 0; x < shape->vol; ++x) {
    const double hx = cfg[x] / 2.0;
    hsq += sq(hx);
  }
  return hsq;
}


inline std::vector<double> measure_Cl_mom(
    const int* cfg, const latt_shape* shape, const double e2) {
  std::vector<double> Cl(2*shape->dims[ND-1]);
  const int mu = ND-1;
  for (int x = 0; x < shape->vol; ++x) {
    const auto [x_mu, sx_mu] = shift_site_idx(x, 1, mu, shape);
    const double lx_mu = (sx_mu*cfg[x_mu] - cfg[x]) / 2.0;
    const int t = compute_comp(x, ND-1, shape);
    Cl[t] += std::exp(-e2*lx_mu); // pos-parity
    Cl[t + shape->dims[ND-1]] += std::exp(e2*lx_mu); // neg-parity
  }
  
  return Cl;
}

inline std::vector<cdouble> measure_Ch_mom(
    const int* cfg, const std::vector<double>& p,
    const latt_shape* shape) {
  assert(p.size() == ND-1);
  std::vector<cdouble> Ch_mom(shape->dims[ND-1]);
  for (int x = 0; x < shape->vol; ++x) {
    const int t = compute_comp(x, ND-1, shape);
    double px = 0.0;
    for (int i = 0; i < ND-1; ++i) {
      px += compute_comp(x, i, shape) * p[i];
    }
    Ch_mom[t] += std::exp(1i * px) * (cfg[x] / 2.0);
  }
  return Ch_mom;
}

inline double compute_Ox(const int* cfg, int x, const latt_shape* shape) {
  const double h = cfg[x] / 2.0;
  const auto [x1, sx1] = shift_site_idx(x, 1, 0, shape);
  const auto [x2, sx2] = shift_site_idx(x, 1, 1, shape);
  const auto [x3, sx3] = shift_site_idx(x, 1, 2, shape);
  const double h1 = sx1*cfg[x1] / 2.0;
  const double h2 = sx2*cfg[x2] / 2.0;
  const double h3 = sx3*cfg[x3] / 2.0;
  const auto [x12, sx12] = shift_site_idx(x1, 1, 1, shape);
  const auto [x23, sx23] = shift_site_idx(x2, 1, 2, shape);
  const auto [x13, sx13] = shift_site_idx(x1, 1, 2, shape);
  const double h12 = sx12*sx1*cfg[x12] / 2.0;
  const double h23 = sx23*sx2*cfg[x23] / 2.0;
  const double h13 = sx13*sx1*cfg[x13] / 2.0;
  const auto [x123, sx123] = shift_site_idx(x12, 1, 2, shape);
  const double h123 = sx123*sx12*sx1*cfg[x123] / 2.0;
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
