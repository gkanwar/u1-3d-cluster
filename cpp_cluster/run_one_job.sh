#!/bin/bash
#SBATCH --job-name=u1_3d_cluster
#SBATCH --partition=wilson
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

if [ -z "${e2+x}" ]; then
    echo "Must set e2"
    exit
fi
if [ -z "${L+x}" ]; then
    echo "Must set L"
    exit
fi

if [ -z "${n_iter+x}" ]; then
    echo "Must set n_iter"
    exit
fi
if [ -z "${n_therm+x}" ]; then
    echo "Must set n_therm"
    exit
fi
if [ -z "${n_skip_meas+x}" ]; then
    echo "Must set n_skip_meas"
    exit
fi
if [ -z "${seed+x}": ]; then
    echo "Must set seed"
    exit
fi


cd /space4/kanwar/quantum_link/u1/cpp_cluster
srun --ntasks-per-node=1 \
    ./u1_3d_cluster --n_iter=${n_iter} --n_therm=${n_therm} --n_skip_meas=${n_skip_meas} \
    --seed=${seed} --e2=${e2} --L=${L} \
    --out_prefix=obs_trace_${e2}_L${L}_cluster
