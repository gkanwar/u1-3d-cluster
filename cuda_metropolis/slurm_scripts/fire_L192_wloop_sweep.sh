#!/bin/bash

export L=192
export n_iter=10000000
export n_therm=10000
export n_skip_meas=1000

ALL_E2=($(seq 0.6 0.1 1.4))
# ALL_E2+=($(seq 1.5 0.5 3.0))
# ALL_X=($(seq 45 48))
ALL_X=($(seq 93 96))
ALL_X+=($(seq 69 72))
ALL_X+=($(seq 61 64))
ALL_SEED=($(seq 34772 35772))
i=0
for e2 in "${ALL_E2[@]}"; do
    for x in "${ALL_X[@]}"; do
	seed="${ALL_SEED[i]}"
	sbatch --time 24:00:00 --export=ALL,e2=${e2},x=${x},seed=${seed} \
	    slurm_scripts/run_one_wloop_job.sh
	((i++))
    done
done
