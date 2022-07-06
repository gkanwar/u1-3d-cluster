#!/bin/bash

L=30
n_iter=10000
n_therm=1000
n_skip_meas=10

ALL_E2=($(seq 0.10 0.10 2.00))
ALL_SEED=($(seq 678426 679426))
j=0
for i in "${!ALL_E2[@]}"; do
    e2="${ALL_E2[i]}"
    seed="${ALL_SEED[i]}"
    j=$(( (j+1) % 4 ))
    ../cpp_cluster/src/u1_3d_cluster \
        --n_iter=${n_iter} --n_therm=${n_therm} --n_skip_meas=${n_skip_meas} \
        --L=${L} --e2=${e2} --seed=${seed} \
        --out_prefix=data_tej/obs_trace_${e2}_L${L}_cluster >/dev/null &
    if [[ "${j}" == "0" ]]; then
        wait
    fi
done
