#!/bin/bash

export L=96
export n_iter=1000000
export n_therm=10000
export n_bin_meas=100

ALL_E2=($(seq 0.6 0.1 1.4))
ALL_X=($(seq 0 48))
ALL_SEED=($(seq 31014 32014))
out_dir=wloop_snake/raw_obs_v2_bias0.01

i=0
for e2 in "${ALL_E2[@]}"; do
    for x in "${ALL_X[@]}"; do
	seed="${ALL_SEED[i]}"
        fname="raw_obs_v2_bias0.01/dyn_wloop_v2_stag_L${L}_${e2}_x${x}_Wt_hist.dat"
        if [[ -f "$fname" ]]; then
            echo "Skipping e2=${e2} x=${x}"
        else
	    echo "Firing e2=${e2} x=${x} seed=${seed}"
	    sbatch --time 48:00:00 \
                   --export=ALL,e2=${e2},seed=${seed},x=${x},out_dir=${out_dir} \
                   run_one_job.sh
        fi
	((i++))
    done
done
