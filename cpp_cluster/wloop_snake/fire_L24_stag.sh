#!/bin/bash

export L=24
export n_iter=50000000
export n_therm=100000
export n_bin_meas=5000

ALL_E2=($(seq 0.40 0.10 1.00))
ALL_X=($(seq 6 11))
ALL_SEED=($(seq 261454 262454))

i=0
for e2 in "${ALL_E2[@]}"; do
    for x in "${ALL_X[@]}"; do
	seed="${ALL_SEED[i]}"
	echo "Firing e2=${e2} x=${x} seed=${seed}"
	sbatch --time 48:00:00 --export=ALL,e2=${e2},seed=${seed},x=${x} run_one_job.sh
	((i++))
    done
done
