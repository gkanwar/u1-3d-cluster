#!/bin/bash

export L=64
export n_iter=100000
export n_therm=10000
export n_skip_meas=100
export n_metr=3
export cper=yes

ALL_E2=($(seq 0.30 0.05 2.00))
ALL_SEED=($(seq 723976 724976))
for i in "${!ALL_E2[@]}"; do
    e2="${ALL_E2[i]}"
    seed="${ALL_SEED[i]}"
    echo "Firing e2=${e2} seed=${seed}"
    sbatch --time 20:00:00 --export=ALL,e2=${e2},seed=${seed} run_one_job.sh
done
