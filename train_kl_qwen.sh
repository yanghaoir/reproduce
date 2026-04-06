#!/bin/bash
# Qwen2.5-VL 训练脚本（KL散度 + 对比学习）
# 基于 train_kl.sh，适配 Qwen2.5-VL-7B-Instruct

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
eval "$(conda shell.bash hook)"
conda activate reproduce

unset CUDA_VISIBLE_DEVICES

# ============================================================
# 基础配置
# ============================================================
export GPU=1
export MODEL=lamda-0.2-1epoch_qwen
export MODEL_PATH=/mnt1/open_source/models/Qwen2.5-VL-7B-Instruct
# Qwen 从头训练，不加载预训练 LoRA
export LORA_PATH=""
export DATASET_PATH=/mnt1/yanghao/data/NTT-hil-insight/buchong
export CORPUS_PATH=/mnt1/yanghao/data/NTT-hil-insight/OpenDocVQA-Corpus/data
export OUTPUT_DIR=outputs/${MODEL}

# ============================================================
# 训练模式配置
# ============================================================
export TRAINING_MODE=mixed
export KL_LOSS_WEIGHT=0.2
export IMAGE_SAMPLE_STRATEGY=random

echo "============================================================"
echo "Qwen2.5-VL 训练配置"
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
deepspeed --include localhost:${GPU} --master_port 12346 --module vdocrag.vdocretriever.driver.train \
  --deepspeed deepspeed/ds_zero3_config.json \
  --output_dir ${OUTPUT_DIR} \
  --model_name_or_path ${MODEL_PATH} \
  --save_strategy "no" \
  --lora \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
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
