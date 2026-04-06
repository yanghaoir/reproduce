#!/bin/bash
# ------------------------------------------------------------------
# 编码单个 corpus 分片 (在指定 GPU 上运行)
# 用法: bash encode_shard.sh <GPU_ID> <MODEL> <DATASET> <NUM_SHARDS> <SHARD_INDEX> [BATCH_SIZE]
# ------------------------------------------------------------------
set -e

if [ "$#" -lt 5 ]; then
    echo "用法: $0 <GPU_ID> <MODEL> <DATASET> <NUM_SHARDS> <SHARD_INDEX> [BATCH_SIZE]"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$1
MODEL=$2
DATASET=$3
NUM_SHARDS=$4
SHARD_INDEX=$5
BATCH_SIZE=${6:-2}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate reproduce

# 路径配置
# https://huggingface.co/datasets/NTT-hil-insight/OpenDocVQA-Corpus
CORPUS_PATH=/mnt1/open_source/datas/realign/OpenDocVQA-Corpus

# https://huggingface.co/microsoft/Phi-3-vision-128k-instruct
PHI3=/mnt1/open_source/datas/realign/Phi3

EMBEDDING_OUTPUT_DIR="${REPO_DIR}/emb/${MODEL}"
LORA="${REPO_DIR}/outputs/${MODEL}"

mkdir -p "${EMBEDDING_OUTPUT_DIR}"

echo "[Shard ${SHARD_INDEX}/${NUM_SHARDS}] GPU=${CUDA_VISIBLE_DEVICES} Dataset=${DATASET} BatchSize=${BATCH_SIZE}"

python -m vdocrag.vdocretriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path ${PHI3} \
  --lora_name_or_path ${LORA} \
  --lora \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --per_device_eval_batch_size ${BATCH_SIZE} \
  --corpus_name "${CORPUS_PATH}/${DATASET}" \
  --corpus_path "${CORPUS_PATH}/${DATASET}/*.parquet" \
  --corpus_split train \
  --dataset_number_of_shards ${NUM_SHARDS} \
  --dataset_shard_index ${SHARD_INDEX} \
  --encode_output_path "${EMBEDDING_OUTPUT_DIR}/corpus.${DATASET}.${SHARD_INDEX}.pkl"

echo "[Shard ${SHARD_INDEX}/${NUM_SHARDS}] 完成。"
