#!/bin/bash

export L=96
export n_iter=1000000
export n_therm=10000
export n_skip_meas=100

ALL_E2=($(seq 0.30 0.05 1.80))
ALL_SEED=($(seq 830060 831060))
for i in "${!ALL_E2[@]}"; do
    e2="${ALL_E2[i]}"
    seed="${ALL_SEED[i]}"
    sbatch --time 00:50:00 --export=ALL,e2=${e2},seed=${seed} run_one_job.sh
done
