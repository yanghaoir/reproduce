#!/bin/bash
# ------------------------------------------------------------------
# 检索 + 评估 (Search & Eval)
# 用法: bash search_eval.sh <GPU_ID> <MODEL> <DATASET>
# docvqa 使用 depth=700，其他使用 depth=1000
# ------------------------------------------------------------------
set -e

if [ "$#" -lt 3 ]; then
    echo "用法: $0 <GPU_ID> <MODEL> <DATASET>"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$1
MODEL=$2
DATASET=$3

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate reproduce

# 路径配置
# https://huggingface.co/datasets/NTT-hil-insight/OpenDocVQA
QUERY_PATH=/mnt1/open_source/datas/realign/OpenDocVQA

EMBEDDING_OUTPUT_DIR="${REPO_DIR}/emb/${MODEL}"
RESULT_DIR="${REPO_DIR}/result/${MODEL}"

mkdir -p "${RESULT_DIR}"

# 根据数据集选择 depth
if [ "$DATASET" == "docvqa" ]; then
    DEPTH=700
else
    DEPTH=1000
fi

echo "[Eval] Dataset=${DATASET} Depth=${DEPTH}"

# 1. Retrieval (Search)
python -m vdocrag.vdocretriever.driver.search \
    --query_reps "${EMBEDDING_OUTPUT_DIR}/query-${DATASET}.pkl" \
    --document_reps "${EMBEDDING_OUTPUT_DIR}/corpus.${DATASET}.*.pkl" \
    --depth ${DEPTH} \
    --batch_size 64 \
    --save_text \
    --save_ranking_to "${EMBEDDING_OUTPUT_DIR}/rank.${DATASET}.${DATASET}.txt"

# 2. Convert retrieval results (.txt) to .trec file
python -m vdocrag.utils.format.convert_result_to_trec \
    --input "${EMBEDDING_OUTPUT_DIR}/rank.${DATASET}.${DATASET}.txt" \
    --output "${EMBEDDING_OUTPUT_DIR}/rank.${DATASET}.${DATASET}.trec" \
    --remove_query

# 3. Create ground-truth retrieval results
python -m vdocrag.utils.format.convert_qas_to_trec_qrels \
    --dataset_name ${QUERY_PATH} \
    --dataset_config ${DATASET} \
    --output "${EMBEDDING_OUTPUT_DIR}/qrels.${DATASET}.txt"

# 4. Evaluate with pyserini
echo "=================================================="
echo "  评估结果: ${DATASET} (depth=${DEPTH})"
echo "=================================================="

python -m pyserini.eval.trec_eval -c \
    -mrecip_rank \
    -mmap \
    -mrecall.1,5,10 \
    -mndcg_cut.1,5,10 \
    "${EMBEDDING_OUTPUT_DIR}/qrels.${DATASET}.txt" \
    "${EMBEDDING_OUTPUT_DIR}/rank.${DATASET}.${DATASET}.trec" \
    | tee -a "${RESULT_DIR}/eval.txt"

echo ""
echo "[Eval] ${DATASET} 评估完成。"
echo "==================================================" >> "${RESULT_DIR}/eval.txt"
echo "Dataset: ${DATASET}  Depth: ${DEPTH}  Model: ${MODEL}" >> "${RESULT_DIR}/eval.txt"
echo "==================================================" >> "${RESULT_DIR}/eval.txt"
