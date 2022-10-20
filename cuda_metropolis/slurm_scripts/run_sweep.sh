#!/bin/bash

# ALL_E2=( 0.30 0.35 0.40 0.45 0.50 0.55 0.60 )
# ALL_SEEDS=( 839398 910403 821142 302683 459054 668529 597297 )
ALL_E2=( 0.50 0.55 0.60 )
ALL_SEEDS=( 459054 668529 597297 )
L=80

for i in "${!ALL_E2[@]}"; do
    e2="${ALL_E2[i]}"
    seed="${ALL_SEEDS[i]}"
    ./src/u1_3d_cuda --n_iter=50000 --n_therm=500 --n_skip_meas=100 \
                 --seed=$seed --e2=$e2 --L=$L \
                 --out_prefix raw_obs/obs_trace_${e2}_L${L}_cuda
done
