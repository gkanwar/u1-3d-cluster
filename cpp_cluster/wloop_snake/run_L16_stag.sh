#!/bin/bash

NITER=5000000
NTHERM=10000
NBIN=5000

L=16

for e2 in 0.40 0.50 0.60 0.70 0.80 0.90 1.00; do
    for x in 3 4 5 6 7 8 9 10 11; do
        ../src/u1_3d_wloop \
            --n_iter=${NITER} --n_therm=${NTHERM} --n_bin_meas=${NBIN} \
            --seed=423${t} --e2=${e2} --L=${L} --x=${x} --t=3 \
            --out_prefix=raw_obs/dyn_wloop_stag_L${L}_${e2}_x${x} \
            --stag | tail -n10 &
    done
    wait
done

# wait
