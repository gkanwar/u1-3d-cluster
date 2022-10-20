#ifndef UTIL_H
#define UTIL_H

inline double sum_array(const double* arr, unsigned n) {
  double tot = 0.0;
  for (int j = 0; j < n; ++j) {
    tot += arr[j];
  }
  return tot;
}

inline double sum_field(const double* arr, unsigned L) {
  return sum_array(arr, L*L*L);
}

inline double sum_field_staggered(const double* arr, unsigned L) {
  double tot = 0.0;
  for (int x = 0; x < L; ++x) {
    for (int y = 0; y < L; ++y) {
      for (int z = 0; z < L; ++z) {
        int parity = 2*((x+y+z) % 2) - 1;
        int j = x*L*L + y*L + z;
        tot += parity * arr[j];
      }
    }
  }
  return tot;
}

#endif
