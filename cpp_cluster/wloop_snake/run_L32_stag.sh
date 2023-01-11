#!/bin/bash

NITER=5000000
NTHERM=10000
NBIN=500

L=32
x=6

# 0.40 0.50 0.60 0.70 0.80 0.90 1.00
for e2 in 1.10 1.20 1.30 1.40 1.50 2.00 2.50 3.00 3.50 4.00; do
    for t in 3 4 5; do
        ../src/u1_3d_wloop \
            --n_iter=${NITER} --n_therm=${NTHERM} --n_bin_meas=${NBIN} \
            --seed=523${t} --e2=${e2} --L=${L} --x=${x} --t=${t} \
            --out_prefix=raw_obs/raw_obs_stag_L${L}_${e2}_x${x}_t${t} \
            --stag | tail -n10
    done &
done
wait
