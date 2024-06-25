#!/bin/bash

set -x

export n_iter=3000000
export n_therm=10000
export n_skip_meas=1000
export out_dir=raw_obs_bias0.01_mL6

ALL_L=(64 96 128 192 256)
ALL_E2=(1.35 1.29 1.23 1.12 1.09)
# XF = 5/8 L
ALL_XF=(40 60 80 120 160)
ALL_TIME=("02:00:00" "05:00:00" "16:00:00" "24:00:00" "24:00:00")
# ALL_SEED=($(seq 467339 477339))
ALL_SEED=($(seq 901497 911497))
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
	if [[ "${L}" == "256" ]]; then
	sbatch --time "${wtime}" \
	    --export=ALL,e2=${e2},x=${x},seed=${seed},L=${L} \
	    slurm_scripts/run_one_wloop_job.sh
	fi
	((i++))
    done
done
