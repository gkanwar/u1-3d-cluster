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
#include "metropolis.h"
#include "my_rand.h"
#include "args.hxx"

constexpr int SHIFT_FREQ = 100;
constexpr double PI = 3.141592653589793238462643383279502884197169399;

using namespace std;
using namespace std::chrono;
constexpr double BIL = 1000000000;

double sum_array(const double* arr, unsigned n) {
  double tot = 0.0;
  for (unsigned j = 0; j < n; ++j) {
    tot += arr[j];
  }
  return tot;
}

vector<int> make_init_cfg(const latt_shape* shape) {
  if (!shape->staggered) {
    return vector<int>(shape->vol, 0);
  }
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
  for (unsigned i = 0; i < cfg.size(); ++i) {
    cfg[i] -= x;
  }
}

void run_cluster(
    double e2, int n_iter, int n_therm, int n_skip_meas, int n_metr,
    my_rand& rng, const latt_shape* shape,
    vector<double> &E_hist, vector<double> &M_hist,
    vector<double> &MT_hist, vector<double> &MC_hist,
    vector<double> &hsq_hist, vector<double> &Cl_mom_hist,
    vector<cdouble> &Ch_mom_hist, vector<cdouble> &Ch_mom1_hist,
    vector<int> &cluster_histogram) {

  vector<int> cfg = make_init_cfg(shape);

  uniform_int_distribution<int> site_dist =
      uniform_int_distribution<int>(0, shape->vol-1);
  uniform_int_distribution<int> offset_dist =
      uniform_int_distribution<int>(-1, 1);
  assert(ND == 3);
  const vector<double> p0{0, 0};
  const vector<double> p1{2*PI / shape->dims[0], 2*PI / shape->dims[1]};

  assert(cluster_histogram.size() == (unsigned) shape->vol);

  for (int i = -n_therm; i < n_iter; ++i) {
    // cluster update
    int cfg_star = 0;
    if (!shape->cper) {
      const int site = site_dist(rng); // rng() % shape->vol
      cfg_star = cfg[site];
      if (!shape->staggered) {
        cfg_star += offset_dist(rng);
      }
    }
    
    auto _start = steady_clock::now();
    const vector<int> cluster_sizes =
        flip_clusters(cfg.data(), cfg_star, e2, rng, shape);
    cout << "TIME flip_clusters "
         << (steady_clock::now() - _start).count() / BIL << "\n";

    // metropolis update
    _start = steady_clock::now();
    for (int i = 0; i < n_metr; ++i) {
      if (i == 0) {
        cout << "Metropolis...\n";
      }
      double acc = metropolis_update(cfg.data(), e2, rng, shape);
      cout << "\t" << (i+1) << " / " << n_metr << ", acc = " << acc << "\n";
      if (i == n_metr-1) {
        cout << "TIME metropolis_update "
             << (steady_clock::now() - _start).count() / BIL << "\n";
      }
    }

    // re-zero cfg periodically, if periodic BCs
    if (!shape->cper && (i+1) % SHIFT_FREQ == 0) {
      rezero_cfg(cfg);
    }

    // histogram cluster sizes
    if (i >= 0) {
      int _cluster_tot = 0;
      for (auto s : cluster_sizes) {
        cluster_histogram[s]++;
        _cluster_tot += s;
      }
      assert(_cluster_tot == shape->vol);
    }
      
    // measurements
    if (i >= 0 && (i+1) % n_skip_meas == 0) {
      cout << (i+1) << " / " << n_iter << "\n";
      double E = measure_E(cfg.data(), shape);
      double M = measure_M(cfg.data(), shape);
      double MT = measure_MT(cfg.data(), shape);
      double MC = measure_MC(cfg.data(), shape);
      double hsq = measure_hsq(cfg.data(), shape);
      vector<double> Cl_mom = measure_Cl_mom(cfg.data(), shape, e2);
      vector<cdouble> Ch_mom_0 = measure_Ch_mom(cfg.data(), p0, shape);
      vector<cdouble> Ch_mom_1 = measure_Ch_mom(cfg.data(), p1, shape);
      int meas_ind = ((i+1) / n_skip_meas) - 1;
      E_hist[meas_ind] = E;
      M_hist[meas_ind] = M;
      MT_hist[meas_ind] = MT;
      MC_hist[meas_ind] = MC;
      hsq_hist[meas_ind] = hsq;
      assert(Cl_mom_hist.size() >= meas_ind*Cl_mom.size());
      std::copy(
          Cl_mom.begin(), Cl_mom.end(),
          Cl_mom_hist.begin() + meas_ind*Cl_mom.size());
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

template <typename T>
void write_array_to_file(const vector<T>& arr, ostream &os) {
  for (T val : arr) {
    os.write(reinterpret_cast<char*>(&val), sizeof(val));
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
  args::ValueFlag<int> flag_n_skip_meas(
      parser, "n_skip_meas", "Number of iters between measurements",
      {"n_skip_meas"}, args::Options::Required);
  args::ValueFlag<int> flag_n_metr(
      parser, "n_metr", "Number of metropolis hits per iter",
      {"n_metr"}, 0);
  args::ValueFlag<int> flag_seed(
      parser, "seed", "RNG seed", {"seed"}, args::Options::Required);
  args::ValueFlag<double> flag_e2(
      parser, "e2", "Coupling squared", {"e2"}, args::Options::Required);
  args::ValueFlag<int> flag_L(
      parser, "L", "Lattice side length", {"L"}, args::Options::Required);
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
  const int n_skip_meas = args::get(flag_n_skip_meas);
  const int n_metr = args::get(flag_n_metr);
  const unsigned long seed = args::get(flag_seed);
  const string out_prefix = args::get(flag_out_prefix);
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

  const int n_meas = n_iter / n_skip_meas;
  vector<double> E_hist(n_meas);
  vector<double> M_hist(n_meas);
  vector<double> MT_hist(n_meas);
  vector<double> MC_hist(n_meas);
  vector<double> hsq_hist(n_meas);
  vector<double> Cl_mom_hist(2 * L * n_meas);
  vector<cdouble> Ch_mom_hist(L * n_meas);
  vector<cdouble> Ch_mom1_hist(L * n_meas);
  vector<int> cluster_histogram(shape.vol, 0);
  run_cluster(
      e2, n_iter, n_therm, n_skip_meas, n_metr, rng, &shape,
      E_hist, M_hist, MT_hist, MC_hist, hsq_hist, Cl_mom_hist,
      Ch_mom_hist, Ch_mom1_hist, cluster_histogram);

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
  {
    ofstream f(out_prefix + "_Cl_mom.dat", ios::binary);
    write_array_to_file(Cl_mom_hist, f);
  }
  {
    ofstream f(out_prefix + "_Ch_mom.dat", ios::binary);
    write_array_to_file(Ch_mom_hist, f);
  }
  {
    ofstream f(out_prefix + "_Ch_mom1.dat", ios::binary);
    write_array_to_file(Ch_mom1_hist, f);
  }
  {
    ofstream f(out_prefix + "_clust_hist.dat", ios::binary);
    write_array_to_file(cluster_histogram, f);
  }

  return 0;
}
