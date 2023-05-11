#!/bin/bash
#SBATCH --job-name=u1_3d_wloop
#SBATCH --partition=wilson
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/space4/kanwar/quantum_link/u1/cpp_cluster/slurm_logs/slurm-%j.out

if [ -z "${e2+x}" ]; then
    echo "Must set e2"
    exit
fi
if [ -z "${L+x}" ]; then
    echo "Must set L"
    exit
fi
if [ -z "${x+x}" ]; then
    echo "Must set x"
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
if [ -z "${n_bin_meas+x}" ]; then
    echo "Must set n_bin_meas"
    exit
fi
if [ -z "${seed+x}" ]; then
    echo "Must set seed"
    exit
fi

if [ -z "${unstag+x}" ]; then
    STAG_FLAG='--stag'
    STAG_TAG='stag'
else
    STAG_FLAG=''
    STAG_TAG='unstag'
fi

if [ -z "${out_dir+x}" ]; then
    out_dir=wloop_snake/raw_obs_v2
fi


cd /space4/kanwar/quantum_link/u1/cpp_cluster
# --L=${L}
# ${STAG_FLAG}
srun --ntasks-per-node=1 \
    ./src/u1_3d_wloop_L${L} --n_iter=${n_iter} --n_therm=${n_therm} --n_bin_meas=${n_bin_meas} \
    --seed=${seed} --e2=${e2} --x=${x} --t=3 \
    --out_prefix=${out_dir}/dyn_wloop_v2_${STAG_TAG}_L${L}_${e2}_x${x}
