#!/bin/bash

export L=64
export n_iter=1000000
export n_therm=10000
export n_bin_meas=100

ALL_E2=($(seq 0.6 0.1 0.9))
ALL_X=($(seq 0 32))
ALL_SEED=($(seq 197601 198601))

i=0
for e2 in "${ALL_E2[@]}"; do
    for x in "${ALL_X[@]}"; do
	seed="${ALL_SEED[i]}"
	echo "Firing e2=${e2} x=${x} seed=${seed}"
	sbatch --time 24:00:00 --export=ALL,e2=${e2},seed=${seed},x=${x} run_one_job.sh
	((i++))
    done
done