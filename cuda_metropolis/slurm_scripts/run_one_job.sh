#!/bin/bash
#SBATCH --job-name=u1_3d_cluster
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --qos=job_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.out

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


cd ${HOME}/u1_3d_cluster/cuda_metropolis
module load CUDA
srun --ntasks-per-node=1 \
    ./src/u1_3d_cuda --n_iter=${n_iter} --n_therm=${n_therm} --n_skip_meas=${n_skip_meas} \
    --seed=${seed} --e2=${e2} --L=${L} \
    --out_prefix=raw_obs/obs_trace_${e2}_L${L}_cuda
