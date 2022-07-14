/// We would like to run the cluster algorithm much faster, so here is a C++
/// version. Most of the bottleneck in Python likely has to do with lack of
/// fused operations and data locality.

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
#include "measurements.h"
#include "args.hxx"

constexpr int SHIFT_FREQ = 100;
constexpr double PI = 3.141592653589793238462643383279502884197169399;

using namespace std;
using namespace std::chrono;
constexpr double BIL = 1000000000;

double sum_array(const double* arr, unsigned n) {
  double tot = 0.0;
  for (int j = 0; j < n; ++j) {
    tot += arr[j];
  }
  return tot;
}

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

void run_cluster(
    double e2, int n_iter, int n_therm, int n_skip_meas,
    my_rand& rng, const latt_shape* shape,
    vector<double> &E_hist, vector<double> &M_hist,
    vector<double> &MT_hist, vector<double> &MC_hist,
    vector<double> &hsq_hist, vector<double> &Cl_hist,
    vector<cdouble> &Ch_mom_hist, vector<cdouble> &Ch_mom1_hist) {

  vector<int> cfg = make_init_cfg(shape);

  uniform_int_distribution<int> site_dist =
      uniform_int_distribution<int>(0, shape->vol);
  assert(ND == 3);
  const vector<double> p0{0, 0};
  const vector<double> p1{2*PI / shape->dims[0], 2*PI / shape->dims[1]};

  for (int i = -n_therm; i < n_iter; ++i) {
    // cluster update
    const int site = site_dist(rng); // rng() % shape->vol
    const int cfg_star = cfg[site];
    auto _start = steady_clock::now();
    flip_clusters(cfg.data(), cfg_star, e2, rng, shape);
    cout << "TIME flip_clusters "
         << (steady_clock::now() - _start).count() / BIL << "\n";

    // re-zero cfg periodically
    if ((i+1) % SHIFT_FREQ == 0) {
      rezero_cfg(cfg);
    }

    // measurements
    if (i >= 0 && (i+1) % n_skip_meas == 0) {
      cout << (i+1) << " / " << n_iter << "\n";
      double E = measure_E(cfg.data(), shape);
      double M = measure_M(cfg.data(), shape);
      double MT = measure_MT(cfg.data(), shape);
      double MC = measure_MC(cfg.data(), shape);
      double hsq = measure_hsq(cfg.data(), shape);
      // vector<double> Cl = measure_Cl(cfg.data(), shape);
      vector<cdouble> Ch_mom_0 = measure_Ch_mom(cfg.data(), p0, shape);
      vector<cdouble> Ch_mom_1 = measure_Ch_mom(cfg.data(), p1, shape);
      int meas_ind = ((i+1) / n_skip_meas) - 1;
      E_hist[meas_ind] = E;
      M_hist[meas_ind] = M;
      MT_hist[meas_ind] = MT;
      MC_hist[meas_ind] = MC;
      hsq_hist[meas_ind] = hsq;
      // assert(Cl_hist.size() >= meas_ind*Cl.size());
      // std::copy(
      //     Cl.begin(), Cl.end(),
      //     Cl_hist.begin() + meas_ind*Cl.size());
      assert(Ch_mom_hist.size() >= meas_ind*Ch_mom_0.size());
      std::copy(
          Ch_mom_0.begin(), Ch_mom_0.end(),
          Ch_mom_hist.begin() + meas_ind*Ch_mom_0.size());
      assert(Ch_mom1_hist.size() >= meas_ind*Ch_mom_1.size());
      std::copy(
          Ch_mom_1.begin(), Ch_mom_1.end(),
          Ch_mom1_hist.begin() + meas_ind*Ch_mom_1.size());
    }
    
  }

}

void write_array_to_file(const vector<double>& arr, ostream &os) {
  for (double val : arr) {
    os.write(reinterpret_cast<char*>(&val), sizeof(val));
  }
}
void write_array_to_file(const vector<cdouble>& arr, ostream &os) {
  for (cdouble val : arr) {
    os.write(reinterpret_cast<char*>(&val), sizeof(val));
  }
}


int main(int argc, char** argv) {
  args::ArgumentParser parser("U(1) cluster in C++");
  args::ValueFlag<int> flag_n_iter(
      parser, "n_iter", "Number of MC iters", {"n_iter"}, args::Options::Required);
  args::ValueFlag<int> flag_n_therm(
      parser, "n_therm", "Number of thermalization iters", {"n_therm"}, args::Options::Required);
  args::ValueFlag<int> flag_n_skip_meas(
      parser, "n_skip_meas", "Number of iters between measurements", {"n_skip_meas"}, args::Options::Required);
  args::ValueFlag<int> flag_seed(parser, "seed", "RNG seed", {"seed"}, args::Options::Required);
  args::ValueFlag<double> flag_e2(parser, "e2", "Coupling squared", {"e2"}, args::Options::Required);
  args::ValueFlag<int> flag_L(parser, "L", "Lattice side length", {"L"}, args::Options::Required);
  args::ValueFlag<string> flag_out_prefix(parser, "out_prefix", "Output file prefix", {"out_prefix"}, args::Options::Required);
  args::Flag flag_cper(parser, "cper", "C-periodic BCs", {"cper"});

  try {
    parser.ParseCLI(argc, argv);
  }
  catch (args::Help) {
    cout << parser;
    return 0;
  }
  catch (args::ParseError e) {
    cerr << e.what() << "\n";
    cerr << parser;
    return 1;
  }
  catch (args::ValidationError e) {
    cerr << e.what() << "\n";
    cerr << parser;
    return 1;
  }

  const int L = args::get(flag_L);
  const double e2 = args::get(flag_e2);
  const int n_iter = args::get(flag_n_iter);
  const int n_therm = args::get(flag_n_therm);
  const int n_skip_meas = args::get(flag_n_skip_meas);
  const unsigned long seed = args::get(flag_seed);
  const string out_prefix = args::get(flag_out_prefix);
  const bool cper = args::get(flag_cper);
  if (cper) {
    cout << "Running with C-periodic boundaries.\n";
  }
  else {
    cout << "Running with Periodic boundaries.\n";
  }

  array<int,3> dims = { L, L, L };
  latt_shape shape = make_latt_shape(&dims.front(), cper);

  my_rand rng(seed);

  vector<double> E_hist(n_iter / n_skip_meas);
  vector<double> M_hist(n_iter / n_skip_meas);
  vector<double> MT_hist(n_iter / n_skip_meas);
  vector<double> MC_hist(n_iter / n_skip_meas);
  vector<double> hsq_hist(n_iter / n_skip_meas);
  vector<double> Cl_hist(L * n_iter / n_skip_meas);
  vector<cdouble> Ch_mom_hist(L * n_iter / n_skip_meas);
  vector<cdouble> Ch_mom1_hist(L * n_iter / n_skip_meas);
  run_cluster(
      e2, n_iter, n_therm, n_skip_meas, rng, &shape,
      E_hist, M_hist, MT_hist, MC_hist, hsq_hist, Cl_hist,
      Ch_mom_hist, Ch_mom1_hist);

  double E = sum_array(E_hist.data(), E_hist.size()) / E_hist.size();
  cout << "Mean E/V = " << (E/shape.vol) << "\n";
  double M = sum_array(M_hist.data(), M_hist.size()) / M_hist.size();
  cout << "Mean M = " << M << "\n";
  double MT = sum_array(MT_hist.data(), MT_hist.size()) / MT_hist.size();
  cout << "Mean MT = " << MT << "\n";
  double MC = sum_array(MC_hist.data(), MC_hist.size()) / MC_hist.size();
  cout << "Mean MC = " << MC << "\n";

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
  // {
  //   ofstream f(out_prefix + "_Cl.dat", ios::binary);
  //   write_array_to_file(Cl_hist, f);
  // }
  {
    ofstream f(out_prefix + "_Ch_mom.dat", ios::binary);
    write_array_to_file(Ch_mom_hist, f);
  }
  {
    ofstream f(out_prefix + "_Ch_mom1.dat", ios::binary);
    write_array_to_file(Ch_mom1_hist, f);
  }

  return 0;
}
