#!/bin/bash
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"
# shellcheck source=config/dir_config.sh
source "${REPO_DIR}/config/dir_config.sh"

eval "$(conda shell.bash hook)"
conda activate reproduce

unset CUDA_VISIBLE_DEVICES

# ============================================================
# 基础配置
# ============================================================
export GPU=0,1,2,3
export MODEL=realign-phi3v

export MODEL_PATH="${VDOC_PHI3_MODEL_DIR}"
export LORA_PATH="${VDOC_PHI3_LORA_PRETRAINED_DIR}"
export CORPUS_PATH="${VDOC_TRAIN_CORPUS_PATH}"
export DATASET_PATH="${VDOC_TRAIN_DATASET_PATH}"
export OUTPUT_DIR=outputs/${MODEL}



# KL散度loss权重，用于image batch的知识蒸馏
export KL_LOSS_WEIGHT=0.2

# 遗留代码，无需改动
export IMAGE_SAMPLE_STRATEGY=random
export TRAINING_MODE=mixed

echo "============================================================"
echo "训练配置"
echo "============================================================"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "LORA_PATH: ${LORA_PATH}"
echo "DATASET_PATH: ${DATASET_PATH}"
echo "CORPUS_PATH: ${CORPUS_PATH}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo ""
echo "训练模式: ${TRAINING_MODE}"
echo "KL Loss权重: ${KL_LOSS_WEIGHT}"
echo "图像采样策略: ${IMAGE_SAMPLE_STRATEGY}"
echo "============================================================"

# ============================================================
# 开始训练
# ============================================================
deepspeed --include localhost:${GPU} --master_port 12345 --module vdocrag.vdocretriever.driver.train \
  --deepspeed deepspeed/ds_zero3_config.json \
  --output_dir ${OUTPUT_DIR} \
  --model_name_or_path ${MODEL_PATH} \
  --lora \
  --lora_name_or_path ${LORA_PATH} \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --save_strategy epoch \
  --dataset_path ${DATASET_PATH} \
  --corpus_path ${CORPUS_PATH} \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size 16 \
  --gradient_checkpointing \
  --train_group_size 1 \
  --learning_rate 1e-4 \
  --query_max_len 256 \
  --answer_max_len 256 \
  --num_train_epochs 5 \
  --logging_steps 10 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 4 \
  --report_to none \
  --training_mode ${TRAINING_MODE} \
  --kl_loss_weight ${KL_LOSS_WEIGHT} \
  --image_sample_strategy ${IMAGE_SAMPLE_STRATEGY}

echo ""
echo "训练完成！"
echo "输出目录: ${OUTPUT_DIR}"

