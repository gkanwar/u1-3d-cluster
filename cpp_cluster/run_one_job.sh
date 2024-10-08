#!/bin/bash
#SBATCH --job-name=u1_3d_cluster
#SBATCH --partition=wilson
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/space4/kanwar/quantum_link/u1/cpp_cluster/slurm_logs/slurm-%j.out
#SBATCH --mem-per-cpu 2000M

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
if [ -z "${n_bin_meas+x}" ]; then
    echo "Must set n_bin_meas"
    exit
fi
if [ -z "${seed+x}" ]; then
    echo "Must set seed"
    exit
fi

if [ -z "${n_metr+x}" ]; then
    NMETR_FLAG=''
else
    NMETR_FLAG="--n_metr=${n_metr}"
fi

if [ -z "${cper+x}" ]; then
    CPER_FLAG=''
    CPER_TAG=''
else
    CPER_FLAG='--cper'
    CPER_TAG='_cper'
fi

if [ -z "${unstag+x}" ]; then
    STAG_FLAG='--stag'
    STAG_TAG=''
else
    STAG_FLAG=''
    STAG_TAG='_unstag'
fi

if [ -z "${out_dir+x}" ]; then
    out_dir=raw_obs
fi


cd /space4/kanwar/quantum_link/u1/cpp_cluster
mkdir -p ${out_dir}
srun --ntasks-per-node=1 \
     ./src/u1_3d_cluster --n_iter=${n_iter} --n_therm=${n_therm} \
     --n_skip_meas=${n_skip_meas} --n_bin_meas=${n_bin_meas} \
     --seed=${seed} --e2=${e2} --L=${L} ${CPER_FLAG} ${STAG_FLAG} ${NMETR_FLAG} \
     --out_prefix=${out_dir}/obs_trace_${e2}_L${L}_cluster${CPER_TAG}${STAG_TAG}
