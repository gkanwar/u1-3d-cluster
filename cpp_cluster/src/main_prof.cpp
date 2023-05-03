/// C++-based Metropolis simulation with an inserted Polyakov loop structure.
/// We begin with a setup that allows precise string tension measurements using
/// the snake algorithm. The Polyakov loop structures considered always consist
/// of a pair of Polyakov lines based at sites (0,1,:) and (x,1,:), with an
/// offset to (x+1,1,:) for all t coords in the range [0,t). In particular this
/// means the dual-lattice links with mu=1 and in the union of the ranges
/// [:x,0,:] and [x,0,:t] have a dislocation by 1 which must be included in
/// the simulation.

#include <algorithm>
#include <array>
#include <chrono>
#include <complex>
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <random>

#include "cluster.h"
#include "lattice.h"
#include "latt_ops.h"
#include "measurements.h"
// #include "metropolis.h"
#include "my_rand.h"
#include "util.h"
#include "args.hxx"

using namespace std;
using namespace std::chrono;
constexpr double BIL = 1000000000;
using my_float = float;
using t_cfg = float;

/// Metropolis code for profiling
static std::bernoulli_distribution binary_dist;
static std::uniform_real_distribution<double> unif_dist =
    std::uniform_real_distribution<double>(0.0, 1.0);

inline int shift_fwd_2(int idx, int ax, const latt_shape* shape) {
  int full_block_idx = idx - (idx % shape->blocks[ax]);
  int new_idx = ((idx + shape->strides[ax]) % shape->blocks[ax]) + full_block_idx;
  return new_idx;
}
inline int shift_bwd_2(int idx, int ax, const latt_shape* shape) {
  int full_block_idx = idx - (idx % shape->blocks[ax]);
  int new_idx = ((idx + (shape->dims[ax]-1)*shape->strides[ax]) % shape->blocks[ax]) + full_block_idx;
  return new_idx;
}

inline static int offset_dist(my_rand &rng) {
  return 2*binary_dist(rng)-1;
}
my_float metropolis_update(
    t_cfg* cfg, my_float e2, my_rand &rng) {
  my_float acc = 0.0;
  const int L = 96;
  const latt_shape _shape{
    .dims = {L, L, L},
    .strides = {L*L, L, 1},
    .blocks = {L*L*L, L*L, L},
    .vol = L*L*L,
    .cper = false,
    .staggered = true
  };
  const latt_shape* shape = &_shape;
  for (int x = 0; x < shape->vol; ++x) {
    // const int cfg_x = cfg[x];
    // const int cfg_x_p = cfg_x + 2*offset_dist(rng);
    // const my_float hx = cfg_x / 2.0;
    // const my_float hx_p = cfg_x_p / 2.0;
    const my_float hx = cfg[x];
    const my_float hx_p = hx + offset_dist(rng);
    my_float neighbors = 0.0;
    for (int i = 0; i < ND; ++i) {
      // V1
      // const auto [x_fwd, sx_fwd] = shift_site_idx(x, 1, i, shape);
      // const auto [x_bwd, sx_bwd] = shift_site_idx(x, -1, i, shape);
      // V2
      // const int x_fwd_block = x - (x % shape->blocks[i]);
      // const int x_fwd = ((x + shape->strides[i]) % shape->blocks[i]) + x_fwd_block;
      // const int x_bwd_block = x - (x % shape->blocks[i]);
      // const int x_bwd = ((x + (shape->dims[i]-1)*shape->strides[i]) % shape->blocks[i]) + x_bwd_block;
      // V3
      // const int x_fwd = xi_fwd[i];
      // const int x_bwd = xi_bwd[i];
      // V4
      // const int x_fwd = shift_fwd(x, i, shape).first;
      // const int x_bwd = shift_bwd(x, i, shape).first;
      const int x_fwd = shift_fwd_2(x, i, shape);
      const int x_bwd = shift_bwd_2(x, i, shape);
      
      // const my_float hx_fwd = cfg[x_fwd] / 2.0;
      // const my_float hx_bwd = cfg[x_bwd] / 2.0;
      const my_float hx_fwd = cfg[x_fwd];
      const my_float hx_bwd = cfg[x_bwd];
      neighbors += hx_fwd + hx_bwd;
    }
    // dS *= (e2/2);
    // assert(e2*(ND*(sq(hx_p) - sq(hx)) + (hx - hx_p)*neighbors) == dS);
    const my_float dS = e2*(ND*(sq(hx_p) - sq(hx)) + (hx - hx_p)*neighbors);

    if (unif_dist(rng) < exp(-dS)) {
      cfg[x] = hx_p;
      acc += 1.0 / shape->vol;
    }
  }

  return acc;
}


void run_metropolis_sim(
    my_float e2, int n_iter, my_rand& rng, const latt_shape* shape) {

  vector<int> _cfg = make_init_cfg(shape);
  vector<my_float> cfg(_cfg.size());
  for (int i = 0; i < cfg.size(); ++i) {
    cfg[i] = _cfg[i] / 2.0;
  }

  auto _tot_start = steady_clock::now();

  for (int i = 0; i < n_iter; ++i) {
    // metropolis update
    auto _start = steady_clock::now();
    double acc = metropolis_update(cfg.data(), e2, rng); // , shape);
    if ((i+1) % 1000 == 0) {
      cout << (i+1) << " / " << n_iter << "\n";
      cout << "Metropolis...\n";
      cout << "\tacc = " << acc << "\n";
      cout << "TIME metropolis update "
           << (steady_clock::now() - _start).count() / BIL << "\n";
      double _tot_time = (steady_clock::now() - _tot_start).count() / BIL;
      cout << "EST TIME "
           << _tot_time << " of " << (_tot_time * n_iter)/(i+1) << "\n";
      cout << "AVG ITER TIME " << _tot_time/(i+1) << "\n";
    }
  }

  cout << "TIME total "
       << (steady_clock::now() - _tot_start).count() / BIL << "\n";
}


int main(int argc, char** argv) {
  args::ArgumentParser parser("U(1) Metropolis [profiling]");
  args::ValueFlag<int> flag_n_iter(
      parser, "n_iter", "Number of MC iters", {"n_iter"},
      args::Options::Required);
  args::ValueFlag<int> flag_seed(
      parser, "seed", "RNG seed", {"seed"}, 1234, args::Options::None);
  args::ValueFlag<double> flag_e2(
      parser, "e2", "Coupling squared", {"e2"}, 1.0, args::Options::None);
  args::ValueFlag<int> flag_L(
      parser, "L", "Lattice side length", {"L"}, args::Options::Required);
  args::Flag flag_cper(parser, "cper", "C-periodic BCs", {"cper"});
  args::Flag flag_stag(parser, "stag", "Staggered model", {"stag"});

  try {
    parser.ParseCLI(argc, argv);
  }
  catch (args::Help&) {
    cout << parser;
    return 0;
  }
  catch (args::ParseError& e) {
    cerr << e.what() << "\n";
    cerr << parser;
    return 1;
  }
  catch (args::ValidationError& e) {
    cerr << e.what() << "\n";
    cerr << parser;
    return 1;
  }

  const int L = args::get(flag_L);
  const my_float e2 = args::get(flag_e2);
  const int n_iter = args::get(flag_n_iter);
  const unsigned long seed = args::get(flag_seed);
  const bool cper = args::get(flag_cper);
  const bool stag = args::get(flag_stag);
  if (cper) {
    cout << "Running with C-periodic boundaries.\n";
  }
  else {
    cout << "Running with Periodic boundaries.\n";
  }
  if (stag) {
    cout << "Running staggered model.\n";
  }
  else {
    cout << "Running unstaggered model.\n";
  }

  array<int,3> dims = { L, L, L };
  latt_shape shape = make_latt_shape(&dims.front(), cper, stag);

  my_rand rng(seed);
  run_metropolis_sim(e2, n_iter, rng, &shape);

  return 0;
}
