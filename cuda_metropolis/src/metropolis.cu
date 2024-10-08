/* -*- mode: c++ -*- */

#ifndef METROPOLIS_KERNEL_CU
#define METROPOLIS_KERNEL_CU

#include <cassert>
#include <cstdio>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_profiler_api.h>

#include "config.h"
#include "util.h"

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line) {
  if (cudaSuccess != err) {
    fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line,
            (int)err, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

__device__ double sq(double x) {
  return x*x;
}

__device__ double get_rand_float(curandState* thread_state) {
  return curand_uniform(thread_state);
}
__device__ int get_rand_int(curandState* thread_state, int max) {
  return (int) max * (1.0 - curand_uniform(thread_state));
}


__device__ int get_thread_id() {
  int blockId = blockIdx.x + blockIdx.y * gridDim.x
      + blockIdx.z * gridDim.x * gridDim.y;
  int threadId = blockId * blockDim.x * blockDim.y * blockDim.z
      + threadIdx.z * blockDim.x * blockDim.y
      + threadIdx.y * blockDim.x + threadIdx.x;
  return threadId;
}

__global__ void init_curand(curandState* state, unsigned long seed) {
  int idx = get_thread_id();
  curand_init(seed, idx, 0, &state[idx]);
}

__device__ inline unsigned get_idx(unsigned x, unsigned y, unsigned z, unsigned L) {
  return x*L*L + y*L + z;
}

__device__ double get_local_action(
    double h, double h_xp, double h_xm, double h_yp, double h_ym, double h_zp,
    double h_zm, double e2) {
  return (e2/2.0) * (
      sq(h - h_xp) +
      sq(h - h_xm) +
      sq(h - h_yp) +
      sq(h - h_ym) +
      sq(h - h_zp) +
      sq(h - h_zm) );
}

__global__ void metropolis_kernel(
    int* cfg, curandState* rng_state, double e2, int parity, unsigned L) {
  // thread index has x running fastest, z slowest, whereas our coordinates have
  // x running slowest, z fastest... so we swap labels at this point to match
  const unsigned int z0 = (blockIdx.x * blockDim.x + threadIdx.x) * THREAD_L;
  const unsigned int y0 = (blockIdx.y * blockDim.y + threadIdx.y) * THREAD_L;
  const unsigned int x0 = (blockIdx.z * blockDim.z + threadIdx.z) * THREAD_L;
  const int threadId = get_thread_id();
  curandState* thread_rng_state = &rng_state[threadId];
  for (unsigned int x = x0; x < x0+THREAD_L; ++x) {
    for (unsigned int y = y0; y < y0+THREAD_L; ++y) {
      // NOTE: we assume THREAD_L is even, so (x0,y0,z0) is an EVEN
      // site, meaning we can start at (x0,y0,z0+parity) and advance by 2.
      for (unsigned int z = z0+parity; z < z0+THREAD_L; z += 2) {
        int cfg_site = cfg[get_idx(x, y, z, L)];
        const double h_xp = cfg[get_idx((x+1)%L, y, z, L)] / 2.0;
        const double h_xm = cfg[get_idx((x+L-1)%L, y, z, L)] / 2.0;
        const double h_yp = cfg[get_idx(x, (y+1)%L, z, L)] / 2.0;
        const double h_ym = cfg[get_idx(x, (y+L-1)%L, z, L)] / 2.0;
        const double h_zp = cfg[get_idx(x, y, (z+1)%L, L)] / 2.0;
        const double h_zm = cfg[get_idx(x, y, (z+L-1)%L, L)] / 2.0;
        double S = get_local_action(cfg_site/2.0, h_xp, h_xm, h_yp, h_ym, h_zp, h_zm, e2);
        for (unsigned i = 0; i < N_METROPOLIS_HITS; ++i) {
          const int dcfg = 4*get_rand_int(thread_rng_state, 2) - 2;
          const double new_S = get_local_action(
              (cfg_site+dcfg)/2.0, h_xp, h_xm, h_yp, h_ym, h_zp, h_zm, e2);
          if (get_rand_float(thread_rng_state) < exp(-new_S + S)) {
            cfg_site += dcfg;
            S = new_S;
          }
        }
        cfg[get_idx(x, y, z, L)] = cfg_site;
      }
    }
  }
}

__global__ void metropolis_snake_kernel(
    int* cfg, int* wloop, curandState* rng_state, double e2, int parity, unsigned L) {
  // thread index has x running fastest, z slowest, whereas our coordinates have
  // x running slowest, z fastest... so we swap labels at this point to match
  const unsigned int z0 = (blockIdx.x * blockDim.x + threadIdx.x) * THREAD_L;
  const unsigned int y0 = (blockIdx.y * blockDim.y + threadIdx.y) * THREAD_L;
  const unsigned int x0 = (blockIdx.z * blockDim.z + threadIdx.z) * THREAD_L;
  const int threadId = get_thread_id();
  curandState* thread_rng_state = &rng_state[threadId];
  const int wloop_x = wloop[0];
  const int wloop_t = wloop[1];
  for (unsigned int x = x0; x < x0+THREAD_L; ++x) {
    for (unsigned int y = y0; y < y0+THREAD_L; ++y) {
      // NOTE: we assume THREAD_L is even, so (x0,y0,z0) is an EVEN
      // site, meaning we can start at (x0,y0,z0+parity) and advance by 2.
      for (unsigned int z = z0+parity; z < z0+THREAD_L; z += 2) {
        int cfg_site = cfg[get_idx(x, y, z, L)];
        const double h_xp = cfg[get_idx((x+1)%L, y, z, L)] / 2.0;
        const double h_xm = cfg[get_idx((x+L-1)%L, y, z, L)] / 2.0;
        double h_yp = cfg[get_idx(x, (y+1)%L, z, L)] / 2.0;
        double h_ym = cfg[get_idx(x, (y+L-1)%L, z, L)] / 2.0;
        const double h_zp = cfg[get_idx(x, y, (z+1)%L, L)] / 2.0;
        const double h_zm = cfg[get_idx(x, y, (z+L-1)%L, L)] / 2.0;
        if (x < wloop_x || x == wloop_x && z < wloop_t) {
          if (y == 0) {
            h_yp += 1;
          }
          else if (y == 1) {
            h_ym -= 1;
          }
        }
        double S = get_local_action(cfg_site/2.0, h_xp, h_xm, h_yp, h_ym, h_zp, h_zm, e2);
        for (unsigned i = 0; i < N_METROPOLIS_HITS; ++i) {
          const int dcfg = 4*get_rand_int(thread_rng_state, 2) - 2;
          const double new_S = get_local_action(
              (cfg_site+dcfg)/2.0, h_xp, h_xm, h_yp, h_ym, h_zp, h_zm, e2);
          if (get_rand_float(thread_rng_state) < exp(-new_S + S)) {
            cfg_site += dcfg;
            S = new_S;
          }
        }
        cfg[get_idx(x, y, z, L)] = cfg_site;
      }
    }
  }
}

__device__ void compute_Wt_weights(
    const int* cfg, int wloop_x, double* ws, double e2, unsigned L) {
  double accum_log_w = 0.0;
  ws[0] = 1.0;
  double tot_w = 1.0;
  for (int z = 0; z < L; ++z) {
    const unsigned int x = get_idx(wloop_x, 0, z, L);
    const unsigned int y = get_idx(wloop_x, 1, z, L);
    const double hx = cfg[x] / 2.0;
    const double hy = cfg[y] / 2.0;
    accum_log_w += -e2*(hy - hx + 0.5) + SIGMA_BIAS;
    ws[z+1] = exp(accum_log_w);
    tot_w += ws[z+1];
  }
  for (int z = 0; z <= L; ++z) {
    ws[z] /= tot_w;
  }
}

__global__ void heatbath_wloop_kernel(
    const int* cfg, int* wloop, double* Wt_weights,
    curandState* rng_state, double e2, unsigned L) {
  compute_Wt_weights(cfg, wloop[0], Wt_weights, e2, L);
  const int threadId = get_thread_id();
  curandState* thread_rng_state = &rng_state[threadId];
  const double r = get_rand_float(thread_rng_state);
  double tot_w = 0.0;
  for (int z = 0; z <= L; ++z) {
    tot_w += Wt_weights[z];
    if (tot_w >= r) {
      wloop[1] = z;
      break;
    }
  }
}

__global__ void update_Wt_hist(
    const int* cfg, const int* wloop, double* Wt_hist, double* Wt_weights,
    double e2, unsigned L) {
  compute_Wt_weights(cfg, wloop[0], Wt_weights, e2, L);
  for (int z = 0; z <= L; ++z) {
    Wt_hist[z] += Wt_weights[z];
  }
}
    


__global__ void measure_E(const int* cfg, double* E, unsigned L) {
  const unsigned int z0 = (blockIdx.x * blockDim.x + threadIdx.x) * THREAD_L;
  const unsigned int y0 = (blockIdx.y * blockDim.y + threadIdx.y) * THREAD_L;
  const unsigned int x0 = (blockIdx.z * blockDim.z + threadIdx.z) * THREAD_L;
  double E_local = 0.0;
  for (unsigned int x = x0; x < x0+THREAD_L; ++x) {
    for (unsigned int y = y0; y < y0+THREAD_L; ++y) {
      for (unsigned int z = z0; z < z0+THREAD_L; ++z) {
        const double h = cfg[get_idx(x, y, z, L)] / 2.0;
        const double h_xp = cfg[get_idx((x+1)%L, y, z, L)] / 2.0;
        const double h_yp = cfg[get_idx(x, (y+1)%L, z, L)] / 2.0;
        const double h_zp = cfg[get_idx(x, y, (z+1)%L, L)] / 2.0;
        // E[get_idx(x, y, z, L)] = (
	E_local += (
            sq(h - h_xp) +
            sq(h - h_yp) +
            sq(h - h_zp) );
      }
    }
  }
  atomicAdd(E, E_local);
}

__global__ void measure_OC(const int* cfg, double* OC, unsigned L) {
  const unsigned int z0 = (blockIdx.x * blockDim.x + threadIdx.x) * THREAD_L;
  const unsigned int y0 = (blockIdx.y * blockDim.y + threadIdx.y) * THREAD_L;
  const unsigned int x0 = (blockIdx.z * blockDim.z + threadIdx.z) * THREAD_L;
  double OC_local = 0.0;
  for (unsigned int x = x0; x < x0+THREAD_L; ++x) {
    for (unsigned int y = y0; y < y0+THREAD_L; ++y) {
      for (unsigned int z = z0; z < z0+THREAD_L; ++z) {
        const double h = cfg[get_idx(x, y, z, L)] / 2.0;
        const double h1 = cfg[get_idx((x+1)%L, y, z, L)] / 2.0;
        const double h2 = cfg[get_idx(x, (y+1)%L, z, L)] / 2.0;
        const double h3 = cfg[get_idx(x, y, (z+1)%L, L)] / 2.0;
        const double h12 = cfg[get_idx((x+1)%L, (y+1)%L, z, L)] / 2.0;
        const double h23 = cfg[get_idx(x, (y+1)%L, (z+1)%L, L)] / 2.0;
        const double h13 = cfg[get_idx((x+1)%L, y, (z+1)%L, L)] / 2.0;
        const double h123 = cfg[get_idx((x+1)%L, (y+1)%L, (z+1)%L, L)] / 2.0;
        const double h_bar = (h + h1 + h2 + h3 + h12 + h23 + h13 + h123) / 8.0;
	const int par = (x + y + z) % 2 == 0 ? 1 : -1;
        // OC[get_idx(x, y, z, L)] = (
	OC_local += par * (
            sq(h - h_bar) + sq(h12 - h_bar) +
            sq(h23 - h_bar) + sq(h13 - h_bar)
            - sq(h1 - h_bar) - sq(h2 - h_bar)
            - sq(h3 - h_bar) - sq(h123 - h_bar) );
      }
    }
  }
  atomicAdd(OC, OC_local);
}

__global__ void init_cfg(int* cfg, unsigned L) {
  const unsigned int z0 = (blockIdx.x * blockDim.x + threadIdx.x) * THREAD_L;
  const unsigned int y0 = (blockIdx.y * blockDim.y + threadIdx.y) * THREAD_L;
  const unsigned int x0 = (blockIdx.z * blockDim.z + threadIdx.z) * THREAD_L;
  for (unsigned int x = x0; x < x0+THREAD_L; ++x) {
    for (unsigned int y = y0; y < y0+THREAD_L; ++y) {
      for (unsigned int z = z0; z < z0+THREAD_L; ++z) {
        cfg[get_idx(x, y, z, L)] = (x+y+z) % 2;
      }
    }
  }
}

__global__ void clear_meas(double* x) {
  *x = 0.0;
}

extern "C" int* alloc_and_init_cfg(int L, dim3 grid_shape, dim3 block_shape) {
  int* d_cfg = NULL;
  checkCudaErrors(cudaMalloc((void**)&d_cfg, L*L*L*sizeof(int)));
  init_cfg<<<grid_shape, block_shape>>>(d_cfg, L);
  return d_cfg;
}

__global__ void init_wloop(int* d_wloop, int wloop_x, int wloop_t) {
  d_wloop[0] = wloop_x;
  d_wloop[1] = wloop_t;
}

extern "C" int* alloc_and_init_wloop(int wloop_x, int L) {
  int* d_wloop = NULL;
  checkCudaErrors(cudaMalloc((void**)&d_wloop, 2*sizeof(int)));
  init_wloop<<<1,1>>>(d_wloop, wloop_x, L/2);
  return d_wloop;
}

extern "C" curandState* init_rng(unsigned long seed, dim3 grid_shape, dim3 block_shape) {
  curandState* rng_state = NULL;
  int block_size = block_shape.x * block_shape.y * block_shape.z;
  int grid_size = grid_shape.x * grid_shape.y * grid_shape.z;
  checkCudaErrors(cudaMalloc(
      (void**)&rng_state,
      block_size * grid_size * sizeof(curandState)));
  init_curand<<<grid_shape, block_shape>>>(rng_state, seed);
  return rng_state;
}

extern "C" void free_cfg(int* d_cfg) {
  if (d_cfg) {
    checkCudaErrors(cudaFree(d_cfg));
  }
}

extern "C" void free_rng(curandState* rng_state) {
  if (rng_state) {
    checkCudaErrors(cudaFree(rng_state));
  }
}

void copy_dev_to_host(double* dest, const double* d_src, unsigned n) {
  checkCudaErrors(cudaMemcpy(
      (void*)dest, (void*)d_src, n*sizeof(double),
      cudaMemcpyDeviceToHost));
}

extern "C" void run_metropolis(
    int* d_cfg, double e2, int L, int n_iter, int n_therm, int n_skip_meas,
    dim3 grid_shape, dim3 block_shape, curandState* rng_state,
    double* E_hist, double* MC_hist) {

  assert(L == grid_shape.x * block_shape.x * THREAD_L);
  assert(L == grid_shape.y * block_shape.y * THREAD_L);
  assert(L == grid_shape.z * block_shape.z * THREAD_L);
  double* d_tmp_E = NULL;
  double* d_tmp_OC = NULL;
  // checkCudaErrors(cudaMalloc((void**)&d_tmp_E, L*L*L*sizeof(double)));
  // checkCudaErrors(cudaMalloc((void**)&d_tmp_OC, L*L*L*sizeof(double)));
  checkCudaErrors(cudaMalloc((void**)&d_tmp_E, sizeof(double)));
  checkCudaErrors(cudaMalloc((void**)&d_tmp_OC, sizeof(double)));
  // double* tmp_E =  (double*) malloc(L*L*L*sizeof(double));
  // double* tmp_OC =  (double*) malloc(L*L*L*sizeof(double));
  double E;
  double MC;

  cudaProfilerStart();
  for (int i = -n_therm; i < n_iter; ++i) {

    if ((i+1) % 1000 == 0) {
      printf("%d / %d\n", i+1, n_iter);
    }

    metropolis_kernel<<<grid_shape, block_shape>>>(d_cfg, rng_state, e2, 0, L);
    metropolis_kernel<<<grid_shape, block_shape>>>(d_cfg, rng_state, e2, 1, L);

    if (i >= 0 && (i+1) % n_skip_meas == 0) {
      // V1 (slow!)
      // // measure all arrays on device
      // measure_E<<<grid_shape, block_shape>>>(d_cfg, d_tmp_E, L);
      // measure_OC<<<grid_shape, block_shape>>>(d_cfg, d_tmp_OC, L);
      // checkCudaErrors(cudaDeviceSynchronize());

      // // copy to host and reduce
      // copy_dev_to_host(tmp_E, d_tmp_E, L*L*L);
      // copy_dev_to_host(tmp_OC, d_tmp_OC, L*L*L);
      // double E = sum_field(tmp_E, L);
      // double MC = sum_field_staggered(tmp_OC, L);

      // V2 (faster): reduce on device
      clear_meas<<<1,1>>>(d_tmp_E);
      clear_meas<<<1,1>>>(d_tmp_OC);
      measure_E<<<grid_shape, block_shape>>>(d_cfg, d_tmp_E, L);
      measure_OC<<<grid_shape, block_shape>>>(d_cfg, d_tmp_OC, L);
      checkCudaErrors(cudaDeviceSynchronize());
      copy_dev_to_host(&E, d_tmp_E, 1);
      copy_dev_to_host(&MC, d_tmp_OC, 1);

      // log history
      int meas_ind = ((i+1) / n_skip_meas) - 1;
      E_hist[meas_ind] = E;
      MC_hist[meas_ind] = MC;
    }
  }
  cudaProfilerStop();

  // free(tmp_E);
  // free(tmp_OC);
  checkCudaErrors(cudaFree(d_tmp_E));
  checkCudaErrors(cudaFree(d_tmp_OC));
}



extern "C" void run_snake_metropolis(
    int* d_cfg, int* d_wloop, double e2, int L,
    int n_iter, int n_therm, int n_skip_meas,
    dim3 grid_shape, dim3 block_shape, curandState* rng_state,
    double* E_hist, double* MC_hist, double* Wt_hist) {

  assert(L == grid_shape.x * block_shape.x * THREAD_L);
  assert(L == grid_shape.y * block_shape.y * THREAD_L);
  assert(L == grid_shape.z * block_shape.z * THREAD_L);
  double* d_tmp_E = NULL;
  double* d_tmp_OC = NULL;
  double* d_tmp_Wt = NULL;
  double* d_Wt_weights = NULL;
  checkCudaErrors(cudaMalloc((void**)&d_tmp_E, sizeof(double)));
  checkCudaErrors(cudaMalloc((void**)&d_tmp_OC, sizeof(double)));
  checkCudaErrors(cudaMalloc((void**)&d_tmp_Wt, (L+1)*sizeof(double)));
  checkCudaErrors(cudaMemset(d_tmp_Wt, 0, (L+1)*sizeof(double)));
  checkCudaErrors(cudaMalloc((void**)&d_Wt_weights, (L+1)*sizeof(double)));
  double E;
  double MC;
  double* tmp_Wt = (double*) malloc((L+1)*sizeof(double));


  cudaProfilerStart();
  for (int i = -n_therm; i < n_iter; ++i) {

    if ((i+1) % 1000 == 0) {
      printf("%d / %d\n", i+1, n_iter);
    }

    metropolis_snake_kernel<<<grid_shape, block_shape>>>(
        d_cfg, d_wloop, rng_state, e2, 0, L);
    metropolis_snake_kernel<<<grid_shape, block_shape>>>(
        d_cfg, d_wloop, rng_state, e2, 1, L);
    heatbath_wloop_kernel<<<1,1>>>(d_cfg, d_wloop, d_Wt_weights, rng_state, e2, L);
    

    if (i >= 0) {
      update_Wt_hist<<<1,1>>>(d_cfg, d_wloop, d_tmp_Wt, d_Wt_weights, e2, L);
      
      if ((i+1) % n_skip_meas == 0) {
        clear_meas<<<1,1>>>(d_tmp_E);
        clear_meas<<<1,1>>>(d_tmp_OC);
        measure_E<<<grid_shape, block_shape>>>(d_cfg, d_tmp_E, L);
        measure_OC<<<grid_shape, block_shape>>>(d_cfg, d_tmp_OC, L);
        checkCudaErrors(cudaDeviceSynchronize());
        copy_dev_to_host(&E, d_tmp_E, 1);
        copy_dev_to_host(&MC, d_tmp_OC, 1);
        copy_dev_to_host(tmp_Wt, d_tmp_Wt, L+1);
        checkCudaErrors(cudaMemset(d_tmp_Wt, 0, (L+1)*sizeof(double)));

        // log history
        int meas_ind = ((i+1) / n_skip_meas) - 1;
        E_hist[meas_ind] = E;
        MC_hist[meas_ind] = MC;
        for (int j = 0; j < L+1; ++j) {
          Wt_hist[(L+1)*meas_ind + j] = tmp_Wt[j] / (double)n_skip_meas;
        }
      }
    }
  }
  cudaProfilerStop();

  // free(tmp_E);
  // free(tmp_OC);
  free(tmp_Wt);

  checkCudaErrors(cudaFree(d_tmp_E));
  checkCudaErrors(cudaFree(d_tmp_OC));
  checkCudaErrors(cudaFree(d_tmp_Wt));
  checkCudaErrors(cudaFree(d_Wt_weights));
}


#endif
