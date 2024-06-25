#!/bin/bash

set -x

export n_iter=5000000
export n_therm=10000
export n_skip_meas=1000

ALL_ML=(6 8 10)
ALL_E2=(1.09 1.18 1.25)
ALL_SEED=($(seq 497592 507592))
i=0
for r in {1..2}; do
for j in "${!ALL_ML[@]}"; do
    e2="${ALL_E2[j]}"
    ML="${ALL_ML[j]}"
    L=256
    xf=160 # XF = 5/8 L
    wtime="${ALL_TIME[j]}"
    export out_dir=raw_obs_bias0.01_mL${ML}
    export tag=stag_r${r}
    for dx in $(seq -3 0); do
	x=$((xf+dx))
	echo "e2=${e2} x=${x} seed=${seed} wtime=${wtime}"
	seed="${ALL_SEED[i]}"
	sbatch --time "24:00:00" \
	    --export=ALL,e2=${e2},x=${x},seed=${seed},L=${L} \
	    slurm_scripts/run_one_wloop_job.sh
	((i++))
    done
done
done
