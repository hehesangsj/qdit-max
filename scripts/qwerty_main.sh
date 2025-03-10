#!/bin/bash
set -x

CONFIG=${1}

# GPUS=${2:-32}
# GPUS_PER_NODE=${3:-8}
GPUS=${2:-1}
GPUS_PER_NODE=${3:-1}
PARTITION=${4:-"INTERN2"}
QUOTA_TYPE=${5:-"reserved"}
JOB_NAME=${6:-"vl_sj"}

CPUS_PER_TASK=${CPUS_PER_TASK:-10}

if [ $GPUS -lt 8 ]; then
    NODES=1
else
    NODES=$((GPUS / GPUS_PER_NODE))
fi

# SRUN_ARGS=${SRUN_ARGS:-" --jobid=3794248"} # 3768157 3768158 3789766 -w HOST-10-140-66-41  3636795

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=32425


QUANT_FLAGS="--wbits 4 --abits 8 \
            --act_group_size 128 --weight_group_size 128 --use_gptq \
            --quant_method max \
            --calib_data_path ./cali_data/cali_data_256.pth"
SAMPLE_FLAGS="--global-batch-size 32 --num-fid-samples 10000 --num-sampling-steps 50 --cfg-scale 1.5 --image-size 256 --seed 0"
EVAL_FLAGS="pretrained_models/VIRTUAL_imagenet256_labeled.npz results/qwerty/017-DiT-XL-2/DiT-XL-2-DiT-XL-2-256x256-size-256-vae-mse-cfg-0.0-seed-0-new.npz"
QWERTY_FLAGS="--image-size 256 --ckpt pretrained_models/DiT-XL-2-256x256.pt --mode gen --results-dir results/qwerty --start_block 0 \
--qwerty-ckpt results/qwerty/014-DiT-XL-2/checkpoints/ckpt.pt --seed 4"
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
  python qwerty/main_qwerty.py $QUANT_FLAGS $SAMPLE_FLAGS $QWERTY_FLAGS
#   python evaluator.py $EVAL_FLAGS

