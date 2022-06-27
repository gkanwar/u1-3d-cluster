#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>

#include "config.h"
#include "util.h"
#include "args.hxx"

using namespace std;

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line) {
  if (cudaSuccess != err) {
    fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line,
            (int)err, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

extern "C" int* alloc_and_init_cfg(int L, dim3 grid_shape, dim3 block_shape);
extern "C" void free_cfg(int* d_cfg);
extern "C" curandState* init_rng(unsigned long seed, dim3 grid_shape, dim3 block_shape);
extern "C" void free_rng(curandState* rng_state);
extern "C" void run_metropolis(
    int* d_cfg, double e2, int L, int n_iter, int n_therm, int n_skip_meas,
    dim3 grid_shape, dim3 block_shape, curandState* rng_state,
    double* E_hist, double* MC_hist);

void write_array_to_file(const vector<double>& arr, ostream &os) {
  for (double val : arr) {
    os.write(reinterpret_cast<char*>(&val), sizeof(val));
  }
}

int main(int argc, char** argv) {
  args::ArgumentParser parser("U(1) metropolis on CUDA");
  args::ValueFlag<int> flag_n_iter(
      parser, "n_iter", "Number of Metropolis iters", {"n_iter"}, args::Options::Required);
  args::ValueFlag<int> flag_n_therm(
      parser, "n_therm", "Number of thermalization iters", {"n_therm"}, args::Options::Required);
  args::ValueFlag<int> flag_n_skip_meas(
      parser, "n_skip_meas", "Number of iters between measurements", {"n_skip_meas"}, args::Options::Required);
  args::ValueFlag<int> flag_seed(parser, "seed", "RNG seed", {"seed"}, args::Options::Required);
  args::ValueFlag<double> flag_e2(parser, "e2", "Coupling squared", {"e2"}, args::Options::Required);
  args::ValueFlag<int> flag_L(parser, "L", "Lattice side length", {"L"}, args::Options::Required);
  args::ValueFlag<string> flag_out_prefix(parser, "out_prefix", "Output file prefix", {"out_prefix"}, args::Options::Required);

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
  const int thread_block_size = 4;
  const string out_prefix = args::get(flag_out_prefix);

  dim3 block_shape(thread_block_size, thread_block_size, thread_block_size);
  int grid_size = L / (thread_block_size * THREAD_L);
  dim3 grid_shape(grid_size, grid_size, grid_size);

  curandState* rng_state = init_rng(seed, grid_shape, block_shape);
  int* d_cfg = alloc_and_init_cfg(L, grid_shape, block_shape);
  vector<double> E_hist(n_iter / n_skip_meas);
  vector<double> MC_hist(n_iter / n_skip_meas);
  run_metropolis(d_cfg, e2, L, n_iter, n_therm, n_skip_meas, grid_shape, block_shape, rng_state,
                 E_hist.data(), MC_hist.data());
  checkCudaErrors(cudaDeviceSynchronize());

  /// TMP:
  // vector<int> cfg(L*L*L);
  // checkCudaErrors(cudaMemcpy(
  //     (void*)cfg.data(), (void*)d_cfg, L*L*L*sizeof(int),
  //     cudaMemcpyDeviceToHost));
  // cout << "TEST CFG\n";
  // for(int x = 0; x < L; ++x) {
  //   for(int y = 0; y < L; ++y) {
  //     cout << cfg[x*L + y] << " ";
  //   }
  //   cout << "\n";
  // }
  // cout << "\n";
  
  free_cfg(d_cfg);
  free_rng(rng_state);

  cout << E_hist[E_hist.size()-1] << "\n";
  cout << MC_hist[MC_hist.size()-1] << "\n";

  double E = sum_array(E_hist.data(), E_hist.size()) / E_hist.size();
  cout << "Mean E/V = " << E/(L*L*L) << "\n";
  double MC = sum_array(MC_hist.data(), MC_hist.size()) / MC_hist.size();
  cout << "Mean MC = " << MC << "\n";

  ofstream f1(out_prefix + "_E.dat", ios::binary);
  write_array_to_file(E_hist, f1);
  ofstream f2(out_prefix + "_MC.dat", ios::binary);
  write_array_to_file(MC_hist, f2);

  return 0;
}
