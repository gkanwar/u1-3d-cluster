#!/bin/bash

export L=192
export n_iter=10000
export n_therm=1000
export n_skip_meas=10

ALL_E2=($(seq 0.30 0.10 2.00))
ALL_SEED=($(seq 58323 59323))
for i in "${!ALL_E2[@]}"; do
    e2="${ALL_E2[i]}"
    seed="${ALL_SEED[i]}"
    echo "Firing e2=${e2} seed=${seed}"
    sbatch --time 06:00:00 --export=ALL,e2=${e2},seed=${seed} run_one_job.sh
done
