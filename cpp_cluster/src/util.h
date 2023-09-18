#ifndef UTIL_H
#define UTIL_H

#include <complex>
#include <iostream>
#include <vector>

using float_t = float;
using cfg_t = float;

inline double sq(double x) {
  return x*x;
}

inline double sum_array(const double* arr, unsigned n) {
  double tot = 0.0;
  for (unsigned j = 0; j < n; ++j) {
    tot += arr[j];
  }
  return tot;
}

inline std::string int_to_fname_str(int n) {
  if (n >= 0) {
    return std::to_string(n);
  }
  else {
    return "m" + std::to_string(-n);
  }
}

template <typename T>
void write_array_to_file(const std::vector<T>& arr, std::ostream &os) {
  for (T val : arr) {
    os.write(reinterpret_cast<char*>(&val), sizeof(val));
  }
}

using cdouble = std::complex<double>;
using namespace std::complex_literals;

constexpr double PI = 3.141592653589793238462643383279502884197169399;

struct WloopShape {
  int x;
  int t;
};


#endif
