#!/bin/bash

set -x

export n_iter=1000000
export n_therm=10000
export n_skip_meas=100
export out_dir=raw_obs_bias0.01_mL8

ALL_L=(64 96 128 192 256)
ALL_E2=(1.68 1.50 1.39 1.24 1.18)
# XF = 5/8 L
ALL_XF=(40 60 80 120 160)
ALL_TIME=("00:15:00" "00:30:00" "01:45:00" "02:30:00" "06:00:00")
ALL_SEED=($(seq 161762 171762))
i=0
for j in "${!ALL_L[@]}"; do
    e2="${ALL_E2[j]}"
    L="${ALL_L[j]}"
    xf="${ALL_XF[j]}"
    wtime="${ALL_TIME[j]}"
    for dx in $(seq -3 0); do
	x=$((xf+dx))
	echo "e2=${e2} x=${x} seed=${seed} wtime=${wtime}"
	seed="${ALL_SEED[i]}"
	sbatch --time "${wtime}" \
	    --export=ALL,e2=${e2},x=${x},seed=${seed},L=${L} \
	    slurm_scripts/run_one_wloop_job.sh
	((i++))
    done
done
