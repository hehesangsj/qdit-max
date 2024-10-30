#!/bin/bash
set -x

CONFIG=${1}

# GPUS=${2:-8}
# GPUS_PER_NODE=${3:-8}
GPUS=${2:-1}
GPUS_PER_NODE=${3:-1}
PARTITION=${4:-"INTERN3"}
QUOTA_TYPE=${5:-"reserved"}
JOB_NAME=${6:-"vl_sj"}

CPUS_PER_TASK=${CPUS_PER_TASK:-10}

if [ $GPUS -lt 8 ]; then
    NODES=1
else
    NODES=$((GPUS / GPUS_PER_NODE))
fi

# SRUN_ARGS=${SRUN_ARGS:-" --jobid=3722476"} # 3768157 3768158 3789766 -w HOST-10-140-66-41  3636795

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=32424    


QUANT_FLAGS="--wbits 4 --abits 8 \
            --act_group_size 128 --weight_group_size 128 \
            --quant_method max \
            --calib_data_path ../cali_data/cali_data_256.pth"
            # --use_gptq \
SAMPLE_FLAGS="--batch-size 1 --num-fid-samples 10000 --num-sampling-steps 50 --cfg-scale 1.5 --image-size 256 --seed 0 --results-dir ../results/gptq"
export PYTHONUNBUFFERED=1

srun -p ${PARTITION} \
  --job-name=${JOB_NAME} \
  --gres=gpu:${GPUS_PER_NODE} \
  --nodes=${NODES} \
  --ntasks=${GPUS} \
  --ntasks-per-node=${GPUS_PER_NODE} \
  --cpus-per-task=${CPUS_PER_TASK} \
  --kill-on-bad-exit=1 \
  --quotatype=${QUOTA_TYPE} \
  ${SRUN_ARGS} \
  python evaluator.py pretrained_models/VIRTUAL_imagenet256_labeled.npz results/gptq/001-qdit_w4a8/DiT-XL-2-pretrained-size-256-vae-mse-cfg-1.5-seed-0.npz
#   python -u quant_main.py $QUANT_FLAGS $SAMPLE_FLAGS