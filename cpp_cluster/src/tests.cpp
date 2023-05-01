/// Unit tests.

#include <algorithm>
#include <array>
#include <chrono>
#include <complex>
#include <iostream>
#include <fstream>
#include <random>
#include <vector>

#include "gtest/gtest.h"

#include "cluster.h"
#include "lattice.h"
#include "measurements.h"

using namespace std;

TEST(LatticeTest, IdxToCoord){
  array<int,ND> dims{4, 6, 8};
  const bool cper = false;
  const bool stag = false;
  const latt_shape shape = make_latt_shape(&dims.front(), cper, stag);
  for (int x = 0; x < shape.vol; ++x) {
    for (int i = 0; i < ND; ++i) {
      for (int diff = -1; diff <= 1; diff += 2) {
        const auto [y, sy] = shift_site_idx(x, diff, i, &shape);
        EXPECT_EQ(sy, 1);
        for (int j = 0; j < ND; ++j) {
          if (i == j) {
            EXPECT_EQ(
                compute_comp(y, j, &shape),
                (compute_comp(x, j, &shape) + diff + shape.dims[i]) % shape.dims[i] );
          }
          else {
            EXPECT_EQ(
                compute_comp(y, j, &shape),
                compute_comp(x, j, &shape) );
          }
        }
      }
    }
  }
}

TEST(LatticeTest, IdxToCoordCper){
  array<int,ND> dims{4, 6, 8};
  const bool cper = true;
  const bool stag = false;
  const latt_shape shape = make_latt_shape(&dims.front(), cper, stag);
  for (int x = 0; x < shape.vol; ++x) {
    for (int i = 0; i < ND; ++i) {
      for (int diff = -1; diff <= 1; diff += 2) {
        const auto [y, sy] = shift_site_idx(x, diff, i, &shape);
        if (i != ND-1 &&
            ((diff < 0 && compute_comp(x, i, &shape) == 0) ||
             (diff > 0 && compute_comp(x, i, &shape) == shape.dims[i]-1))) {
          EXPECT_EQ(sy, -1);
        }
        else {
          EXPECT_EQ(sy, 1);
        }
        for (int j = 0; j < ND; ++j) {
          if (i == j) {
            EXPECT_EQ(
                compute_comp(y, j, &shape),
                (compute_comp(x, j, &shape) + diff + shape.dims[i]) % shape.dims[i] );
          }
          else {
            EXPECT_EQ(
                compute_comp(y, j, &shape),
                compute_comp(x, j, &shape) );
          }
        }
      }
    }
  }
}


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
