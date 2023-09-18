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
constexpr float_t SIGMA_BIAS = 0.01;
#ifndef LATT_SIZE
#error "LATT_SIZE must be set"
#else
constexpr int L = LATT_SIZE;
#endif
#ifdef CPER
constexpr bool cper = true;
#else
constexpr bool cper = false;
#endif
#ifdef STAG
constexpr bool stag = true;
#else
constexpr bool stag = false;
#endif
#if(ND == 3)
constexpr latt_shape _shape{
  .dims = {L, L, L},
  .strides = {L*L, L, 1},
  .blocks = {L*L*L, L*L, L},
  .vol = L*L*L,
  .cper = cper,
  .staggered = stag
};
constexpr const latt_shape* shape = &_shape;
#else
#error "Specialized for ND 3"
#endif

using namespace std;
using namespace std::chrono;
constexpr double BIL = 1000000000;

// Measure plaq from Wilson loop snake sim. Forward plaq
// is on the dual link between (x,0,t) and (x,1,t) while backward plaq is on the
// dual link between (x,0,t-1) and (x,1,t-1).
pair<double, double> measure_wloop_plaqs(
    const cfg_t* cfg, float_t e2,
    const latt_shape* shape, const WloopShape& wloop) {
  assert(wloop.t < shape->dims[ND-1] && wloop.t > 0);
  const int x_fwd = get_site_idx(wloop.x, 0, wloop.t, shape);
  const int y_fwd = get_site_idx(wloop.x, 1, wloop.t, shape);
  const float_t hx_fwd = cfg[x_fwd];
  const float_t hy_fwd = cfg[y_fwd];
  double fwd_plaq = exp(-e2*(hy_fwd - hx_fwd + 0.5));
  const int x_bwd = get_site_idx(wloop.x, 0, wloop.t-1, shape);
  const int y_bwd = get_site_idx(wloop.x, 1, wloop.t-1, shape);
  const float_t hx_bwd = cfg[x_bwd];
  const float_t hy_bwd = cfg[y_bwd];
  double bwd_plaq = exp(e2*(hy_bwd - hx_bwd + 0.5));
  return make_pair(fwd_plaq, bwd_plaq);
}

static std::uniform_int_distribution<int> offset_dist =
    std::uniform_int_distribution<int>(0, 1);
static std::uniform_real_distribution<double> unif_dist =
    std::uniform_real_distribution<double>(0.0, 1.0);

vector<double> compute_Wt_weights(
    const cfg_t* cfg, float_t e2,
    const latt_shape* shape, const WloopShape& wloop) {
  double accum_log_w = 0.0;
  vector<double> ws(shape->dims[ND-1]+1, 0);
  ws[0] = 1.0;
  double tot_w = 1.0;
  for (int t = 0; t < shape->dims[ND-1]; ++t) {
    const int x = get_site_idx(wloop.x, 0, t, shape);
    const int y = get_site_idx(wloop.x, 1, t, shape);
    const double hx = cfg[x];
    const double hy = cfg[y];
    // Delta S = e2/2(hy - hx + 1)^2 - e2/2(hy - hx)^2
    //         = e2(hy - hx + 1/2)
    accum_log_w += -e2*(hy - hx + 0.5) + SIGMA_BIAS;
    ws[t+1] = exp(accum_log_w);
    tot_w += ws[t+1];
  }
  for (int t = 0; t <= shape->dims[ND-1]; ++t) {
    ws[t] /= tot_w;
  }
  return ws;
}

// Update the Wilson loop shape in the background of a config
bool metropolis_update_wloop(
    const cfg_t* cfg, float_t e2, my_rand &rng,
    const latt_shape* shape, WloopShape& wloop) {
  const int dt = 2*offset_dist(rng) - 1;
  if (wloop.t + dt < 0 || wloop.t + dt > shape->dims[ND-1]) {
    return false;
  }
  const int plaq_t = (dt > 0) ? wloop.t : (wloop.t-1);
  const int x = get_site_idx(wloop.x, 0, plaq_t, shape);
  const int y = get_site_idx(wloop.x, 1, plaq_t, shape);
  const float_t hx = cfg[x];
  const float_t hy = cfg[y];
  const double weight = exp(dt * (-e2*(hy - hx + 0.5) + SIGMA_BIAS));
  if (unif_dist(rng) < weight) {
    wloop.t += dt;
    return true;
  }
  else {
    return false;
  }
}

void heatbath_update_wloop(
    const cfg_t* cfg, float_t e2, my_rand &rng,
    const latt_shape* shape, WloopShape& wloop) {
  const vector<double> ws = compute_Wt_weights(cfg, e2, shape, wloop);
  const double r = unif_dist(rng);
  double tot_w = 0.0;
  for (int t = 0; t <= shape->dims[ND-1]; ++t) {
    tot_w += ws[t];
    if (r <= tot_w) {
      wloop.t = t;
      break;
    }
  }
}

void update_Wt_hist_v1(
    vector<double> &Wt_hist, [[maybe_unused]] const cfg_t* cfg,
    [[maybe_unused]] float_t e2, [[maybe_unused]] const latt_shape* shape,
    const WloopShape& wloop) {
  Wt_hist[wloop.t] += 1;
}

void update_Wt_hist_v2(
    vector<double> &Wt_hist, const cfg_t* cfg, float_t e2,
    const latt_shape* shape, const WloopShape& wloop) {
  const vector<double> ws = compute_Wt_weights(cfg, e2, shape, wloop);
  for (int t = 0; t <= shape->dims[ND-1]; ++t) {
    Wt_hist[t] += ws[t];
  }
}

//// TEST:
static std::bernoulli_distribution binary_dist;
static std::uniform_real_distribution<double> unif_dist_2 =
    std::uniform_real_distribution<double>(0.0, 1.0);
inline static int offset_dist_2(my_rand &rng) {
  return 2*binary_dist(rng)-1;
}
inline int shift_fwd_3(int idx, int ax, const latt_shape* shape) {
  int full_block_idx = idx - (idx % shape->blocks[ax]);
  int new_idx = ((idx + shape->strides[ax]) % shape->blocks[ax]) + full_block_idx;
  return new_idx;
}
inline int shift_bwd_3(int idx, int ax, const latt_shape* shape) {
  int full_block_idx = idx - (idx % shape->blocks[ax]);
  int new_idx = ((idx + (shape->dims[ax]-1)*shape->strides[ax]) % shape->blocks[ax]) + full_block_idx;
  return new_idx;
}
float_t metropolis_update_with_wloop_v2(
    vector<cfg_t>& cfg, float_t e2, my_rand &rng,
    const WloopShape& wloop) {
  // TODO: Why faster to initiate the shape here?
  constexpr latt_shape _shape{
    .dims = {L, L, L},
    .strides = {L*L, L, 1},
    .blocks = {L*L*L, L*L, L},
    .vol = L*L*L,
    .cper = cper,
    .staggered = stag
  };
  const latt_shape* shape = &_shape;
  float_t acc = 0;
  for (int x = 0; x < shape->vol; ++x) {
    const float_t hx = cfg[x];
    const float_t hx_p = hx + offset_dist_2(rng);
    float_t neighbors = 0.0;
    for (int i = 0; i < ND; ++i) {
      /*
      const auto [x_fwd, sx_fwd] = shift_site_idx(x, 1, i, shape);
      const auto [x_bwd, sx_bwd] = shift_site_idx(x, -1, i, shape);
      float_t hx_fwd = sx_fwd * cfg[x_fwd];
      float_t hx_bwd = sx_bwd * cfg[x_bwd];
      */
      /// FORNOW
      float_t hx_fwd = cfg[shift_fwd_3(x, i, shape)];
      float_t hx_bwd = cfg[shift_bwd_3(x, i, shape)];
      // dislocations from Wilson loop
      const int x0 = compute_comp(x, 0, shape);
      const int x1 = compute_comp(x, 1, shape);
      const int x2 = compute_comp(x, 2, shape);
      if (i == 1 && (x0 < wloop.x || (x0 == wloop.x && x2 < wloop.t))) {
        if (x1 == 0) {
          hx_fwd += 1;
        }
        else if (x1 == 1) {
          hx_bwd -= 1;
        }
      }
      neighbors += hx_fwd + hx_bwd;
    }
    const float_t dS = e2 * (ND*(sq(hx_p) - sq(hx)) + (hx - hx_p)*neighbors);
    if (unif_dist_2(rng) < exp(-dS)) {
      cfg[x] = hx_p;
      acc += 1;
    }
  }
  return acc / shape->vol;
}


void run_wloop_sim(
    float_t e2, int n_iter, int n_therm, int n_bin_meas,
    my_rand& rng, WloopShape wloop,
    vector<double> &E_hist, vector<double> &M_hist,
    vector<double> &MT_hist, vector<double> &MC_hist,
    vector<double> &hsq_hist, vector<double> &Wt_hist) {
    // const latt_shape* shape,
    // vector<double> &Pf_hist, vector<double> &Pb_hist

  vector<cfg_t> cfg = make_init_hx(shape);
  // double Pf_bin = 0.0;
  // double Pb_bin = 0.0;
  vector<double> Wt_hist_bin(shape->dims[ND-1]+1);

  auto _tot_start = steady_clock::now();

  for (int i = -n_therm; i < n_iter; ++i) {
    // metropolis update
    auto _start = steady_clock::now();
    double acc = metropolis_update_with_wloop_v2(cfg, e2, rng, wloop);
    bool updated = true;
    heatbath_update_wloop(cfg.data(), e2, rng, shape, wloop);
    if ((i+1) % 1000 == 0) {
      cout << (i+1) << " / " << n_iter << "\n";
      cout << "Metropolis...\n";
      cout << "\tacc = " << acc << "\n";
      cout << "\twloop updated: " << updated << "\n";
      cout << "TIME metropolis update "
           << (steady_clock::now() - _start).count() / BIL << "\n";
      double _tot_time = (steady_clock::now() - _tot_start).count() / BIL;
      cout << "EST TIME "
           << _tot_time << " of " << (_tot_time * (n_therm+n_iter))/(n_therm+i+1) << "\n";
    }
      
    // re-zero cfg periodically, if periodic BCs
    if (!shape->cper && (i+1) % SHIFT_FREQ == 0) {
      rezero_cfg(cfg);
    }

    // measurements
    if (i >= 0) {
      // const auto [fwd_plaq, bwd_plaq] =
      //     measure_wloop_plaqs(cfg.data(), e2, shape, wloop);
      // Pf_bin += fwd_plaq / n_bin_meas;
      // Pb_bin += bwd_plaq / n_bin_meas;

      update_Wt_hist_v2(Wt_hist_bin, cfg.data(), e2, shape, wloop);
      
      if ((i+1) % n_bin_meas == 0) {
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
        // std::copy(
        //     Wt_hist_bin.begin(),
        //     Wt_hist_bin.end(),
        //     Wt_hist.begin() + meas_ind * Wt_hist_bin.size());
        // std::fill(Wt_hist_bin.begin(), Wt_hist_bin.end(), 0);
        for (unsigned j = 0; j < Wt_hist_bin.size(); ++j) {
          Wt_hist[meas_ind * Wt_hist_bin.size() + j] = Wt_hist_bin[j] / n_bin_meas;
          Wt_hist_bin[j] = 0.0;
        }
        // Pf_hist[meas_ind] = Pf_bin;
        // Pb_hist[meas_ind] = Pb_bin;
        // Pf_bin = 0.0;
        // Pb_bin = 0.0;
        
      }
    }
    
  }

  cout << "TIME total "
       << (steady_clock::now() - _tot_start).count() / BIL << "\n";
}


int main(int argc, char** argv) {
  stringstream desc;
  desc << "U(1) Wilson loop (Snake) [L=" << L << "]";
  args::ArgumentParser parser(desc.str());
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
  // args::ValueFlag<int> flag_L(
  //     parser, "L", "Lattice side length", {"L"}, args::Options::Required);
  args::ValueFlag<int> flag_wloop_x(
      parser, "wloop_x", "Wloop spatial separation", {"x"}, args::Options::Required);
  args::ValueFlag<int> flag_wloop_t(
      parser, "wloop_t", "Wloop kink time extent", {"t"}, args::Options::Required);
  args::ValueFlag<string> flag_out_prefix(
      parser, "out_prefix", "Output file prefix", {"out_prefix"},
      args::Options::Required);
  // args::Flag flag_cper(parser, "cper", "C-periodic BCs", {"cper"});
  // args::Flag flag_stag(parser, "stag", "Staggered model", {"stag"});

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

  // const int L = args::get(flag_L);
  const float_t e2 = args::get(flag_e2);
  const int n_iter = args::get(flag_n_iter);
  const int n_therm = args::get(flag_n_therm);
  const int n_bin_meas = args::get(flag_n_bin_meas);
  const unsigned long seed = args::get(flag_seed);
  const string out_prefix = args::get(flag_out_prefix);
  // const bool cper = args::get(flag_cper);
  // const bool stag = args::get(flag_stag);
  const int wloop_x = args::get(flag_wloop_x);
  const int wloop_t = args::get(flag_wloop_t);
  const WloopShape wloop { .x = wloop_x, .t = wloop_t };
  cout << "Running with lattice size L=" << L << ".\n";
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

  // array<int,3> dims = { L, L, L };
  // latt_shape shape = make_latt_shape(&dims.front(), cper, stag);

  my_rand rng(seed);

  const int n_meas = n_iter / n_bin_meas;
  vector<double> E_hist(n_meas);
  vector<double> M_hist(n_meas);
  vector<double> MT_hist(n_meas);
  vector<double> MC_hist(n_meas);
  vector<double> hsq_hist(n_meas);
  // vector<double> Pf_hist(n_meas);
  // vector<double> Pb_hist(n_meas);
  vector<double> Wt_hist(n_meas * (shape->dims[ND-1]+1));
  run_wloop_sim(
      e2, n_iter, n_therm, n_bin_meas, rng, wloop,
      E_hist, M_hist, MT_hist, MC_hist, hsq_hist, Wt_hist); // Pf_hist, Pb_hist);

  double E = sum_array(E_hist.data(), E_hist.size()) / E_hist.size();
  cout << "Mean E/V = " << (E/shape->vol) << "\n";
  double M = sum_array(M_hist.data(), M_hist.size()) / M_hist.size();
  cout << "Mean M = " << M << "\n";
  double MT = sum_array(MT_hist.data(), MT_hist.size()) / MT_hist.size();
  cout << "Mean MT = " << MT << "\n";
  double MC = sum_array(MC_hist.data(), MC_hist.size()) / MC_hist.size();
  cout << "Mean MC = " << MC << "\n";

  // double Pf = sum_array(Pf_hist.data(), Pf_hist.size()) / Pf_hist.size();
  // double Pb = sum_array(Pb_hist.data(), Pb_hist.size()) / Pb_hist.size();
  // cout << "sigma(Pf) = " << -log(Pf) << ", sigma(Pb) = " << log(Pb) << "\n";
  vector<double> check_Wt_hist(shape->dims[ND-1]+1);
  int tot = 0;
  for (int i = 0; i < n_meas; ++i) {
    for (int j = 0; j <= shape->dims[ND-1]; ++j) {
      double count = Wt_hist[i * (shape->dims[ND-1]+1) + j];
      check_Wt_hist[j] += count;
      tot += count;
    }
  }
  cout << "Wt hist =";
  for (int j = 0; j <= shape->dims[ND-1]; ++j) {
    cout << " " << check_Wt_hist[j] / ((double)tot);
  }
  cout << "\n";
  cout << "sigma =";
  for (int j = 0; j < shape->dims[ND-1]; ++j) {
    cout << " ";
    cout << log(check_Wt_hist[j] / ((double)check_Wt_hist[j+1])) + SIGMA_BIAS;
  }
  cout << "\n";

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
  //   ofstream f(out_prefix + "_Pf.dat", ios::binary);
  //   write_array_to_file(Pf_hist, f);
  // }
  // {
  //   ofstream f(out_prefix + "_Pb.dat", ios::binary);
  //   write_array_to_file(Pb_hist, f);
  // }
  {
    ofstream f(out_prefix + "_Wt_hist.dat", ios::binary);
    write_array_to_file(Wt_hist, f);
  }

  return 0;
}
