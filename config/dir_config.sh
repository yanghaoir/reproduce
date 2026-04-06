# =============================================================================
# 数据与模型目录
# =============================================================================

# --- Phi-3 Vision（encode 里的 PHI3、Phi-3 训练里的 MODEL_PATH）
# 下载: https://huggingface.co/microsoft/Phi-3-vision-128k-instruct
export VDOC_PHI3_MODEL_DIR="/mnt1/open_source/datas/realign/Phi3"

# --- VDocRetriever Phi-3 LoRA 预训练（仅 Phi-3 训练里的 LORA_PATH）
# 下载: https://huggingface.co/NTT-hil-insight/VDocRetriever-Phi3-vision-pretrained
export VDOC_PHI3_LORA_PRETRAINED_DIR="/mnt1/open_source/datas/realign/VDocRetriever-Phi3-vision-pretrained"

# --- Qwen2.5-VL-7B（encode_qwen 里的 QWEN、Qwen 训练里的 MODEL_PATH）
# 下载: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
export VDOC_QWEN_MODEL_DIR="/mnt1/open_source/models/Qwen2.5-VL-7B-Instruct"

# --- Qwen 训练：可选 LoRA 基座路径（当前为空表示不加载预训练 LoRA）
export VDOC_QWEN_LORA_PATH=""

# --- OpenDocVQA 查询集根目录（encode_query* / search_eval 里的 QUERY_PATH）
# 下载: https://huggingface.co/datasets/NTT-hil-insight/OpenDocVQA
export VDOC_OPEN_DOC_VQA_DIR="/mnt1/open_source/datas/realign/OpenDocVQA"

# --- OpenDocVQA-Corpus 仓库根（encode_shard* 里的 CORPUS_PATH；各数据集为子目录）
# 下载: https://huggingface.co/datasets/NTT-hil-insight/OpenDocVQA-Corpus
export VDOC_OPEN_DOC_VQA_CORPUS_ROOT="/mnt1/open_source/datas/realign/OpenDocVQA-Corpus"

# --- 训练：--dataset_path / --corpus_path（Phi-3 与 Qwen 共用）
# dataset 可为相对仓库根目录的路径；corpus 一般为 OpenDocVQA-Corpus 的 data 子目录
# 下载: https://huggingface.co/datasets/NTT-hil-insight/OpenDocVQA-Corpus/tree/main/data
export VDOC_TRAIN_DATASET_PATH="train_data"
export VDOC_TRAIN_CORPUS_PATH="/mnt1/open_source/datas/realign/OpenDocVQA-Corpus/data"
