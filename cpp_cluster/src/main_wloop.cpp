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

#include "cluster.h"
#include "lattice.h"
#include "latt_ops.h"
#include "measurements.h"
#include "metropolis.h"
#include "my_rand.h"
#include "args.hxx"

constexpr int SHIFT_FREQ = 100;

using namespace std;
using namespace std::chrono;
constexpr double BIL = 1000000000;

// Measure plaq from Wilson loop snake sim. Forward plaq
// is on the dual link between (x,0,t) and (x,1,t) while backward plaq is on the
// dual link between (x,0,t-1) and (x,1,t-1).
pair<double, double> measure_wloop_plaqs(
    const int* cfg, double e2, const latt_shape* shape, const WloopShape& wloop) {
  assert(wloop.t < shape->dims[ND-1] && wloop.t > 0);
  const int x_fwd = get_site_idx(wloop.x, 0, wloop.t, shape);
  const int y_fwd = get_site_idx(wloop.x, 1, wloop.t, shape);
  const double hx_fwd = cfg[x_fwd] / 2.0;
  const double hy_fwd = cfg[y_fwd] / 2.0;
  double fwd_plaq = exp(-e2*(hy_fwd - hx_fwd + 0.5));
  const int x_bwd = get_site_idx(wloop.x, 0, wloop.t-1, shape);
  const int y_bwd = get_site_idx(wloop.x, 1, wloop.t-1, shape);
  const double hx_bwd = cfg[x_bwd] / 2.0;
  const double hy_bwd = cfg[y_bwd] / 2.0;
  double bwd_plaq = exp(e2*(hy_bwd - hx_bwd + 0.5));
  return make_pair(fwd_plaq, bwd_plaq);
}

void run_wloop_sim(
    double e2, int n_iter, int n_therm, int n_bin_meas,
    my_rand& rng, const latt_shape* shape, WloopShape wloop,
    vector<double> &E_hist, vector<double> &M_hist,
    vector<double> &MT_hist, vector<double> &MC_hist,
    vector<double> &hsq_hist, vector<double> &Pf_hist,
    vector<double> &Pb_hist) {

  vector<int> cfg = make_init_cfg(shape);
  double Pf_bin = 0.0;
  double Pb_bin = 0.0;

  auto _tot_start = steady_clock::now();

  for (int i = -n_therm; i < n_iter; ++i) {
    // metropolis update
    auto _start = steady_clock::now();
    cout << "Metropolis...\n";
    double acc = metropolis_update_wloop(cfg.data(), e2, rng, shape, wloop);
    cout << "\tacc = " << acc << "\n";
    cout << "TIME metropolis update "
         << (steady_clock::now() - _start).count() / BIL << "\n";

    // re-zero cfg periodically, if periodic BCs
    if (!shape->cper && (i+1) % SHIFT_FREQ == 0) {
      rezero_cfg(cfg);
    }

    // measurements
    if (i >= 0) {
      const auto [fwd_plaq, bwd_plaq] =
          measure_wloop_plaqs(cfg.data(), e2, shape, wloop);
      Pf_bin += fwd_plaq / n_bin_meas;
      Pb_bin += bwd_plaq / n_bin_meas;
      
      if ((i+1) % n_bin_meas == 0) {
        cout << (i+1) << " / " << n_iter << "\n";
        const double E = measure_E(cfg.data(), shape);
        const double M = measure_M(cfg.data(), shape);
        const double MT = measure_MT(cfg.data(), shape);
        const double MC = measure_MC(cfg.data(), shape);
        const double hsq = measure_hsq(cfg.data(), shape);
        int meas_ind = ((i+1) / n_bin_meas) - 1;
        E_hist[meas_ind] = E;
        M_hist[meas_ind] = M;
        MT_hist[meas_ind] = MT;
        MC_hist[meas_ind] = MC;
        hsq_hist[meas_ind] = hsq;
        Pf_hist[meas_ind] = Pf_bin;
        Pb_hist[meas_ind] = Pb_bin;
        Pf_bin = 0.0;
        Pb_bin = 0.0;
      }
    }

    cout << "TIME total "
         << (steady_clock::now() - _tot_start).count() / BIL << "\n";
    
  }

}


int main(int argc, char** argv) {
  args::ArgumentParser parser("U(1) cluster in C++");
  args::ValueFlag<int> flag_n_iter(
      parser, "n_iter", "Number of MC iters", {"n_iter"},
      args::Options::Required);
  args::ValueFlag<int> flag_n_therm(
      parser, "n_therm", "Number of thermalization iters", {"n_therm"},
      args::Options::Required);
  args::ValueFlag<int> flag_n_bin_meas(
      parser, "n_bin_meas", "Number of iters to bin measurements",
      {"n_bin_meas"}, args::Options::Required);
  args::ValueFlag<int> flag_seed(
      parser, "seed", "RNG seed", {"seed"}, args::Options::Required);
  args::ValueFlag<double> flag_e2(
      parser, "e2", "Coupling squared", {"e2"}, args::Options::Required);
  args::ValueFlag<int> flag_L(
      parser, "L", "Lattice side length", {"L"}, args::Options::Required);
  args::ValueFlag<int> flag_wloop_x(
      parser, "wloop_x", "Wloop spatial separation", {"x"}, args::Options::Required);
  args::ValueFlag<int> flag_wloop_t(
      parser, "wloop_t", "Wloop kink time extent", {"t"}, args::Options::Required);
  args::ValueFlag<string> flag_out_prefix(
      parser, "out_prefix", "Output file prefix", {"out_prefix"},
      args::Options::Required);
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
  const double e2 = args::get(flag_e2);
  const int n_iter = args::get(flag_n_iter);
  const int n_therm = args::get(flag_n_therm);
  const int n_bin_meas = args::get(flag_n_bin_meas);
  const unsigned long seed = args::get(flag_seed);
  const string out_prefix = args::get(flag_out_prefix);
  const bool cper = args::get(flag_cper);
  const bool stag = args::get(flag_stag);
  const int wloop_x = args::get(flag_wloop_x);
  const int wloop_t = args::get(flag_wloop_t);
  const WloopShape wloop { .x = wloop_x, .t = wloop_t };
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

  const int n_meas = n_iter / n_bin_meas;
  vector<double> E_hist(n_meas);
  vector<double> M_hist(n_meas);
  vector<double> MT_hist(n_meas);
  vector<double> MC_hist(n_meas);
  vector<double> hsq_hist(n_meas);
  vector<double> Pf_hist(n_meas);
  vector<double> Pb_hist(n_meas);
  run_wloop_sim(
      e2, n_iter, n_therm, n_bin_meas, rng, &shape, wloop,
      E_hist, M_hist, MT_hist, MC_hist, hsq_hist, Pf_hist, Pb_hist);

  double E = sum_array(E_hist.data(), E_hist.size()) / E_hist.size();
  cout << "Mean E/V = " << (E/shape.vol) << "\n";
  double M = sum_array(M_hist.data(), M_hist.size()) / M_hist.size();
  cout << "Mean M = " << M << "\n";
  double MT = sum_array(MT_hist.data(), MT_hist.size()) / MT_hist.size();
  cout << "Mean MT = " << MT << "\n";
  double MC = sum_array(MC_hist.data(), MC_hist.size()) / MC_hist.size();
  cout << "Mean MC = " << MC << "\n";

  double Pf = sum_array(Pf_hist.data(), Pf_hist.size()) / Pf_hist.size();
  double Pb = sum_array(Pb_hist.data(), Pb_hist.size()) / Pb_hist.size();
  cout << "sigma(Pf) = " << -log(Pf) << ", sigma(Pb) = " << log(Pb) << "\n";

  {
    ofstream f(out_prefix + "_E.dat", ios::binary);
    write_array_to_file(E_hist, f);
  }
  {
    ofstream f(out_prefix + "_M.dat", ios::binary);
    write_array_to_file(M_hist, f);
  }
  {
    ofstream f(out_prefix + "_MT.dat", ios::binary);
    write_array_to_file(MT_hist, f);
  }
  {
    ofstream f(out_prefix + "_MC.dat", ios::binary);
    write_array_to_file(MC_hist, f);
  }
  {
    ofstream f(out_prefix + "_hsq.dat", ios::binary);
    write_array_to_file(hsq_hist, f);
  }
  {
    ofstream f(out_prefix + "_Pf.dat", ios::binary);
    write_array_to_file(Pf_hist, f);
  }
  {
    ofstream f(out_prefix + "_Pb.dat", ios::binary);
    write_array_to_file(Pb_hist, f);
  }

  return 0;
}
