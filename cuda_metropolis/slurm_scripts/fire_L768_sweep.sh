#!/bin/bash

export L=768
export n_iter=250000
export n_therm=2500
export n_skip_meas=10


ALL_E2=($(seq 0.30 0.05 0.50))
ALL_SEED=($(seq 720354 721354))
for i in "${!ALL_E2[@]}"; do
    e2="${ALL_E2[i]}"
    seed="${ALL_SEED[i]}"
    sbatch --time 24:00:00 --export=ALL,e2=${e2},seed=${seed} slurm_scripts/run_one_job.sh
done
