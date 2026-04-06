#!/bin/bash
# ------------------------------------------------------------------
# Qwen2.5-VL 多卡并行评估主脚本
#
# 功能：
#   1. 将 corpus 编码按 GPU 数量自动分片，多卡并行编码
#   2. 编码完成后在单卡上编码 query
#   3. 执行检索 + 评估（复用 search_eval.sh）
#
# 用法：
#   bash sh/eval_qwen.sh <MODEL> <GPU_IDS> [DATASETS]
#
# 参数：
#   MODEL     - 模型名称 (必须)
#   GPU_IDS   - 逗号分隔的 GPU ID，如 0,1,2,3 或 5,7 (必须)
#   DATASETS  - 可选，逗号分隔的数据集名称，如 docvqa,slidevqa
#               支持方括号写法: [docvqa,slidevqa]
#               默认评估所有数据集: docvqa,slidevqa,arxivqa,infovqa,plotqa,chartqa
#
# 示例：
#   bash sh/eval_qwen.sh qwen-kl 0,1,2,3
#   bash sh/eval_qwen.sh qwen-kl 1,2,3,5,7 docvqa,slidevqa
#   bash sh/eval_qwen.sh qwen-kl 3
# ------------------------------------------------------------------
set -e

# ===================== 信号处理：Ctrl+C 时清理所有后台子进程 =====================
BG_PIDS=()

cleanup() {
    echo ""
    echo "  收到终止信号，正在清理后台进程..."
    for PID in "${BG_PIDS[@]}"; do
        if kill -0 "$PID" 2>/dev/null; then
            pkill -TERM -P "$PID" 2>/dev/null
            kill -TERM "$PID" 2>/dev/null
        fi
    done
    sleep 1
    for PID in "${BG_PIDS[@]}"; do
        if kill -0 "$PID" 2>/dev/null; then
            pkill -KILL -P "$PID" 2>/dev/null
            kill -KILL "$PID" 2>/dev/null
        fi
    done
    echo "  所有后台进程已清理。"
    exit 130
}

trap cleanup INT TERM

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ===================== 参数解析 =====================
if [ "$#" -lt 2 ]; then
    echo "错误: 参数不足。"
    echo "用法: bash $0 <MODEL> <GPU_IDS> [DATASETS]"
    echo ""
    echo "示例:"
    echo "  bash $0 qwen-kl 0,1,2,3"
    echo "  bash $0 qwen-kl 1,2,3,5,7 docvqa,slidevqa"
    exit 1
fi

MODEL=$1
GPU_IDS_STR=$2

DEFAULT_DATASETS="docvqa,slidevqa,arxivqa,infovqa,plotqa,chartqa"
DATASETS_RAW=${3:-${DEFAULT_DATASETS}}
DATASETS_STR=$(echo "${DATASETS_RAW}" | tr -d '[]')

IFS=',' read -ra GPU_IDS <<< "${GPU_IDS_STR}"
NUM_GPUS=${#GPU_IDS[@]}

IFS=',' read -ra DATASETS <<< "${DATASETS_STR}"

# Qwen 7B 显存更大，默认 batch size 更小
BATCH_SIZE=2

# ===================== 打印配置 =====================
echo "=============================================="
echo "  Qwen2.5-VL 多卡并行评估"
echo "=============================================="
echo "  模型:     ${MODEL}"
echo "  GPU:      ${GPU_IDS_STR} (共 ${NUM_GPUS} 张)"
echo "  数据集:   ${DATASETS_STR}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Conda 环境: vdoc"
echo "=============================================="
echo ""

# ===================== 准备目录 =====================
EMBEDDING_OUTPUT_DIR="${REPO_DIR}/emb/${MODEL}"
RESULT_DIR="${REPO_DIR}/result/${MODEL}"
LOG_DIR="${REPO_DIR}/log/eval-${MODEL}"
mkdir -p "${EMBEDDING_OUTPUT_DIR}" "${RESULT_DIR}" "${LOG_DIR}"

# ===================== 统计 =====================
TOTAL_DATASETS=${#DATASETS[@]}
CURRENT=0
FAILED_DATASETS=()

# ===================== 主循环 =====================
for DATASET in "${DATASETS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [${CURRENT}/${TOTAL_DATASETS}] 数据集: ${DATASET}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # -------- 清理旧的 embedding 文件 --------
    echo "  清理旧的 ${DATASET} embedding 文件..."
    rm -f "${EMBEDDING_OUTPUT_DIR}/corpus.${DATASET}."*.pkl
    rm -f "${EMBEDDING_OUTPUT_DIR}/query-${DATASET}.pkl"

    # -------- Step 1: 多卡并行 Corpus 编码 --------
    echo "  [Step 1/3] Corpus 编码 (${NUM_GPUS} 分片并行)..."
    PIDS=()
    SHARD_FAILED=0

    for ((i=0; i<NUM_GPUS; i++)); do
        GPU_ID=${GPU_IDS[$i]}
        LOG_FILE="${LOG_DIR}/${DATASET}.corpus.shard${i}.log"
        > "${LOG_FILE}"

        bash "${SCRIPT_DIR}/encode_shard_qwen.sh" \
            "${GPU_ID}" "${MODEL}" "${DATASET}" "${NUM_GPUS}" "${i}" "${BATCH_SIZE}" \
            > "${LOG_FILE}" 2>&1 &

        PIDS+=($!)
        BG_PIDS+=($!)
        echo "    分片 ${i}/${NUM_GPUS} -> GPU ${GPU_ID} (PID: ${PIDS[-1]})"
    done

    # 监控 shard 0 的进度条，同时等待所有分片完成
    MONITOR_LOG="${LOG_DIR}/${DATASET}.corpus.shard0.log"
    while true; do
        ALL_DONE=1
        for PID in "${PIDS[@]}"; do
            if kill -0 "$PID" 2>/dev/null; then
                ALL_DONE=0
                break
            fi
        done

        if [ -f "${MONITOR_LOG}" ]; then
            PROGRESS=$(tr '\r' '\n' < "${MONITOR_LOG}" | grep -oP '\s*\d+%\|[^|]+\|\s*\d+/\d+\s*\[.*?\]' | tail -1)
            if [ -n "$PROGRESS" ]; then
                printf "\r    [Shard 0 进度]%s" "$PROGRESS"
            fi
        fi

        [ ${ALL_DONE} -eq 1 ] && break
        sleep 3
    done
    if [ -f "${MONITOR_LOG}" ]; then
        PROGRESS=$(tr '\r' '\n' < "${MONITOR_LOG}" | grep -oP '\s*\d+%\|[^|]+\|\s*\d+/\d+\s*\[.*?\]' | tail -1)
        if [ -n "$PROGRESS" ]; then
            printf "\r    [Shard 0 进度]%s\n" "$PROGRESS"
        else
            printf "\n"
        fi
    else
        printf "\n"
    fi

    # 检查各分片退出状态
    for ((i=0; i<${#PIDS[@]}; i++)); do
        if ! wait ${PIDS[$i]}; then
            echo "    ✗ 分片 ${i} 编码失败! 查看日志: ${LOG_DIR}/${DATASET}.corpus.shard${i}.log"
            SHARD_FAILED=1
        fi
    done

    if [ ${SHARD_FAILED} -ne 0 ]; then
        echo "  ✗ Corpus 编码失败，跳过 ${DATASET}"
        FAILED_DATASETS+=("${DATASET}")
        continue
    fi
    echo "  ✓ 所有 Corpus 分片编码完成。"

    # -------- Step 2: Query 编码 (单卡) --------
    echo "  [Step 2/3] Query 编码 (GPU ${GPU_IDS[0]})..."
    QUERY_LOG="${LOG_DIR}/${DATASET}.query.log"

    if ! bash "${SCRIPT_DIR}/encode_query_qwen.sh" \
        "${GPU_IDS[0]}" "${MODEL}" "${DATASET}" "${BATCH_SIZE}" \
        > "${QUERY_LOG}" 2>&1; then
        echo "  ✗ Query 编码失败! 查看日志: ${QUERY_LOG}"
        FAILED_DATASETS+=("${DATASET}")
        continue
    fi
    echo "  ✓ Query 编码完成。"

    # -------- Step 3: 检索 + 评估 (复用 search_eval.sh) --------
    echo "  [Step 3/3] 检索 & 评估 (GPU ${GPU_IDS[0]})..."
    EVAL_LOG="${LOG_DIR}/${DATASET}.eval.log"

    if ! bash "${SCRIPT_DIR}/search_eval.sh" \
        "${GPU_IDS[0]}" "${MODEL}" "${DATASET}" \
        2>&1 | tee "${EVAL_LOG}"; then
        echo "  ✗ 评估失败! 查看日志: ${EVAL_LOG}"
        FAILED_DATASETS+=("${DATASET}")
        continue
    fi
    echo "  ✓ ${DATASET} 评估完成。"
    echo ""
done

# ===================== 汇总 =====================
echo ""
echo "=============================================="
echo "  Qwen2.5-VL 评估完成汇总"
echo "=============================================="
echo "  模型:   ${MODEL}"
echo "  GPU:    ${GPU_IDS_STR}"
echo "  总数据集: ${TOTAL_DATASETS}"
echo "  成功: $((TOTAL_DATASETS - ${#FAILED_DATASETS[@]}))"

if [ ${#FAILED_DATASETS[@]} -gt 0 ]; then
    echo "  失败: ${#FAILED_DATASETS[@]} (${FAILED_DATASETS[*]})"
    echo "  日志目录: ${LOG_DIR}"
    echo "=============================================="
    exit 1
else
    echo "  失败: 0"
    echo ""
    echo "  结果文件: ${RESULT_DIR}/eval.txt"
    echo "  日志目录: ${LOG_DIR}"
    echo "=============================================="
fi
