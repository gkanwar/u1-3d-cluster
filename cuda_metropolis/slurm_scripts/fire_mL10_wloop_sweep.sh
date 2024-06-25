#!/bin/bash

set -x

export n_iter=3000000
export n_therm=10000
export n_skip_meas=1000
export out_dir=raw_obs_bias0.01_mL10

ALL_L=(64 96 128 192 256)
ALL_E2=(1.92 1.65 1.49 1.33 1.25)
# XF = 5/8 L
ALL_XF=(40 60 80 120 160)
ALL_TIME=("02:00:00" "05:00:00" "16:00:00" "24:00:00" "24:00:00")
# ALL_TIME=("00:15:00" "00:30:00" "01:45:00" "02:30:00" "06:00:00")
# ALL_SEED=($(seq 701970 711970))
ALL_SEED=($(seq 548305 558305))
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
