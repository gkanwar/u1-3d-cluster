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
#include "latt_ops.h"
#include "measurements.h"
#include "metropolis.h"
#include "my_rand.h"
#include "args.hxx"

constexpr int MAX_N2 = 36;
constexpr int SHIFT_FREQ = 100;

using namespace std;
using namespace std::chrono;
constexpr double BIL = 1000000000;


vector<wavevector> precompute_wavevectors(int max_n2, const latt_shape* shape) {
  cout << "Precomputing wavevectors smaller than " << max_n2 << "\n";
  vector<wavevector> wavevectors;
  const int max_abs_n = (int)std::sqrt(max_n2);
  assert(ND == 3);
  const int L0 = shape->dims[0];
  const int L1 = shape->dims[1];
  assert(L0 % 2 == 0 && L1 % 2 == 0);
  for (int n0 = -max_abs_n; n0 <= max_abs_n; ++n0) {
    for (int n1 = -max_abs_n; n1 <= max_abs_n; ++n1) {
      if (n0*n0 + n1*n1 <= max_n2) {
        cout << "... (" << n0 << "," << n1 << ") ";
        cout << "(" << n0+L0/2 << "," << n1 << ") ";
        cout << "(" << n0 << "," << n1+L1/2 << ") ";
        cout << "(" << n0+L0/2 << "," << n1+L1/2 << ")\n";
        wavevectors.push_back(wavevector{.p = {n0, n1}});
        wavevectors.push_back(wavevector{.p = {n0+L0/2, n1}});
        wavevectors.push_back(wavevector{.p = {n0, n1+L1/2}});
        wavevectors.push_back(wavevector{.p = {n0+L0/2, n1+L1/2}});
      }
    }
  }
  return wavevectors;
}


void run_cluster(
    float_t e2, int n_iter, int n_therm, int n_skip_meas, int n_bin_meas,
    int n_metr, my_rand& rng, const latt_shape* shape,
    vector<double> &E_hist, vector<double> &M_hist,
    vector<double> &MT_hist, vector<double> &MC_hist,
    vector<double> &hsq_hist, vector<double> &Cl_mom_hist,
    vector<vector<cdouble>> &Ch_mom_hist, vector<int> &cluster_histogram,
    const vector<wavevector> &wavevectors) {

  vector<cfg_t> cfg = make_init_hx(shape);

  uniform_int_distribution<int> site_dist =
      uniform_int_distribution<int>(0, shape->vol-1);
  uniform_int_distribution<int> offset_dist =
      uniform_int_distribution<int>(-1, 1);

  assert(wavevectors.size() == Ch_mom_hist.size());
  assert(cluster_histogram.size() == (unsigned) shape->vol);

  double bin_E, bin_M, bin_MT, bin_MC, bin_hsq;
  vector<double> bin_Cl_mom(2*shape->dims[ND-1]);
  vector<vector<cdouble>> bin_Ch_mom(Ch_mom_hist.size());
  auto reset_bins = [&]() {
    bin_E = bin_M = bin_MT = bin_MC = bin_hsq = 0.0;
    std::fill(bin_Cl_mom.begin(), bin_Cl_mom.end(), 0.0);
    for (auto& bin_Ch_mom_n : bin_Ch_mom) {
      bin_Ch_mom_n.resize(shape->dims[ND-1]);
      std::fill(bin_Ch_mom_n.begin(), bin_Ch_mom_n.end(), 0.0);
    }
  };
  reset_bins();

  std::chrono::time_point<steady_clock> _tot_start;

  for (int i = -n_therm; i < n_iter; ++i) {
    if (i == 0) _tot_start = steady_clock::now();
    
    // cluster update
    cfg_t cfg_star = 0;
    if (!shape->cper) {
      const int site = site_dist(rng); // rng() % shape->vol
      cfg_star = cfg[site];
      if (!shape->staggered) {
        cfg_star += offset_dist(rng) / 2.0;
      }
    }
    
    auto _start = steady_clock::now();
    const vector<int> cluster_sizes =
        flip_clusters(cfg.data(), cfg_star, e2, rng, shape);
    auto _cluster_time = (steady_clock::now() - _start).count() / BIL;

    // metropolis update
    _start = steady_clock::now();
    for (int i = 0; i < n_metr; ++i) {
      // double acc =
      metropolis_update(cfg.data(), e2, rng, shape);
      // cout << "\t" << (i+1) << " / " << n_metr << ", acc = " << acc << "\n";
    }
    auto _metr_time = (steady_clock::now() - _start).count() / BIL;

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
      _start = steady_clock::now();
      bin_E += measure_E(cfg.data(), shape);
      bin_M += measure_M(cfg.data(), shape);
      bin_MT += measure_MT(cfg.data(), shape);
      bin_MC += measure_MC(cfg.data(), shape);
      bin_hsq += measure_hsq(cfg.data(), shape);
      vector<double> Cl_mom = measure_Cl_mom(cfg.data(), shape, e2);
      assert(Cl_mom.size() == bin_Cl_mom.size());
      for (unsigned i = 0; i < Cl_mom.size(); ++i) {
        bin_Cl_mom[i] += Cl_mom[i];
      }
      auto _basic_meas_time = (steady_clock::now() - _start).count() / BIL;
      _start = steady_clock::now();
      measure_Ch_all_mom(cfg.data(), wavevectors, shape, bin_Ch_mom);
      // for (unsigned i = 0; i < wavevectors.size(); ++i) {
      //   assert(ND == 3);
      //   const wavevector& wv = wavevectors[i];
      //   const vector<double> p{
      //     2*PI*wv.p[0] / shape->dims[0], 2*PI*wv.p[1] / shape->dims[1]
      //   };
      //   vector<cdouble>& bin_Ch_mom_n = bin_Ch_mom[i];
      //   vector<cdouble> Ch_mom_n = measure_Ch_mom(cfg.data(), p, shape);
      //   assert(Ch_mom_n.size() == bin_Ch_mom_n.size());
      //   for (unsigned i = 0; i < Ch_mom_n.size(); ++i) {
      //     bin_Ch_mom_n[i] += Ch_mom_n[i];
      //   }
      // }
      auto _chmom_meas_time = (steady_clock::now() - _start).count() / BIL;
      

      if ((i+1) % (n_skip_meas*n_bin_meas) == 0) {
        int meas_ind = ((i+1) / (n_skip_meas*n_bin_meas)) - 1;
        E_hist[meas_ind] = bin_E / n_bin_meas;
        M_hist[meas_ind] = bin_M / n_bin_meas;
        MT_hist[meas_ind] = bin_MT / n_bin_meas;
        MC_hist[meas_ind] = bin_MC / n_bin_meas;
        hsq_hist[meas_ind] = bin_hsq / n_bin_meas;
        assert(Cl_mom_hist.size() >= (meas_ind+1)*bin_Cl_mom.size());
        for (unsigned j = 0; j < bin_Cl_mom.size(); ++j) {
          unsigned ind = meas_ind*bin_Cl_mom.size() + j;
          Cl_mom_hist[ind] = bin_Cl_mom[j] / n_bin_meas;
        }
        // std::copy(
        //     bin_Cl_mom.begin(), bin_Cl_mom.end(),
        //     Cl_mom_hist.begin() + meas_ind*bin_Cl_mom.size());
        for (unsigned i = 0; i < wavevectors.size(); ++i) {
          assert(ND == 3);
          vector<cdouble>& Ch_mom_n_hist = Ch_mom_hist[i];
          const vector<cdouble>& bin_Ch_mom_n = bin_Ch_mom[i];
          assert(Ch_mom_n_hist.size() >= (meas_ind+1)*bin_Ch_mom_n.size());
          for (unsigned j = 0; j < bin_Ch_mom_n.size(); ++j) {
            unsigned ind = meas_ind*bin_Ch_mom_n.size() + j;
            Ch_mom_n_hist[ind] = bin_Ch_mom_n[j] / (double)n_bin_meas;
          }
          // std::copy(
          //     bin_Ch_mom_n.begin(), bin_Ch_mom_n.end(),
          //     Ch_mom_n_hist.begin() + meas_ind*bin_Ch_mom_n.size());
        }
        reset_bins();

        /// TIMING INFO
        // if ((i+1) % 1000 == 0)
        {
          cout << "TIME flip_clusters " << _cluster_time << "\n";
          cout << "TIME metropolis_update " << _metr_time << "\n";
          cout << "TIME basic_meas " << _basic_meas_time/n_skip_meas
               << " amortized\n";
          cout << "TIME chmom_meas " << _chmom_meas_time/n_skip_meas
               << " amortized\n";
          double _tot_time = (steady_clock::now() - _tot_start).count() / BIL;
          cout << "EST TIME "
               << _tot_time << " of " << (_tot_time * n_iter)/(i+1) << "\n";
        }
      }
    }
    
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
  args::ValueFlag<int> flag_n_bin_meas(
      parser, "n_bin_meas", "Number of measurements to bin together",
      {"n_bin_meas"}, args::Options::Required);
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
  const float_t e2 = args::get(flag_e2);
  const int n_iter = args::get(flag_n_iter);
  const int n_therm = args::get(flag_n_therm);
  const int n_skip_meas = args::get(flag_n_skip_meas);
  const int n_bin_meas = args::get(flag_n_bin_meas);
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
  const vector<wavevector> wavevectors = precompute_wavevectors(MAX_N2, &shape);

  my_rand rng(seed);

  const int n_meas = n_iter / (n_skip_meas*n_bin_meas);
  vector<double> E_hist(n_meas);
  vector<double> M_hist(n_meas);
  vector<double> MT_hist(n_meas);
  vector<double> MC_hist(n_meas);
  vector<double> hsq_hist(n_meas);
  vector<double> Cl_mom_hist(2 * L * n_meas);
  vector<vector<cdouble>> Ch_mom_hist;
  for (unsigned i = 0; i < wavevectors.size(); ++i) {
    Ch_mom_hist.emplace_back(L * n_meas);
  }
  vector<int> cluster_histogram(shape.vol, 0);
  run_cluster(
      e2, n_iter, n_therm, n_skip_meas, n_bin_meas, n_metr, rng, &shape,
      E_hist, M_hist, MT_hist, MC_hist, hsq_hist, Cl_mom_hist,
      Ch_mom_hist, cluster_histogram, wavevectors);

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
  for (unsigned i = 0; i < wavevectors.size(); ++i) {
    assert(ND == 3);
    const wavevector& wv = wavevectors[i];
    const vector<cdouble>& Ch_mom_hist_n = Ch_mom_hist[i];
    string n_str = int_to_fname_str(wv.p[0]) + "_" + int_to_fname_str(wv.p[1]);
    ofstream f(out_prefix + "_Ch_mom_" + n_str + ".dat", ios::binary);
    write_array_to_file(Ch_mom_hist_n, f);
  }
  {
    ofstream f(out_prefix + "_clust_hist.dat", ios::binary);
    write_array_to_file(cluster_histogram, f);
  }

  return 0;
}
