#!/bin/bash
# ------------------------------------------------------------------
# Qwen2.5-VL: 编码 query (在指定 GPU 上运行)
# 用法: bash encode_query_qwen.sh <GPU_ID> <MODEL> <DATASET> [BATCH_SIZE]
# ------------------------------------------------------------------
set -e

if [ "$#" -lt 3 ]; then
    echo "用法: $0 <GPU_ID> <MODEL> <DATASET> [BATCH_SIZE]"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$1
MODEL=$2
DATASET=$3
BATCH_SIZE=${4:-2}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

eval "$(conda shell.bash hook)"
conda activate reproduce

# 路径配置 — Qwen
# https://huggingface.co/datasets/NTT-hil-insight/OpenDocVQA
QUERY_PATH=/mnt1/open_source/datas/realign/OpenDocVQA

# https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
QWEN=/mnt1/open_source/models/Qwen2.5-VL-7B-Instruct

EMBEDDING_OUTPUT_DIR="${REPO_DIR}/emb/${MODEL}"
LORA="${REPO_DIR}/outputs/${MODEL}"

mkdir -p "${EMBEDDING_OUTPUT_DIR}"

echo "[Query] GPU=${CUDA_VISIBLE_DEVICES} Dataset=${DATASET} BatchSize=${BATCH_SIZE}"

python -m vdocrag.vdocretriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path ${QWEN} \
  --lora_name_or_path ${LORA} \
  --lora \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --encode_is_query \
  --per_device_eval_batch_size ${BATCH_SIZE} \
  --query_max_len 256 \
  --dataset_name ${QUERY_PATH} \
  --dataset_config ${DATASET} \
  --dataset_split test \
  --encode_output_path "${EMBEDDING_OUTPUT_DIR}/query-${DATASET}.pkl"

echo "[Query] ${DATASET} 编码完成。"
