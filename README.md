# Reproduce

多模态文档检索训练与评估的复现说明。

## 环境

进入项目根目录后创建并激活 Conda 环境（Python 3.10）：

```bash
cd reproduce
conda create -n reproduce python=3.10 -y
conda activate reproduce
```

安装依赖与可编辑包：

```bash
pip install -r requirements.txt
pip install -e .
```

**flash-attn**：与 CUDA / PyTorch 版本绑定较紧，`pip` 源码编译容易失败。若安装报错，请改用与当前 CUDA、PyTorch 匹配的预编译 wheel（或按官方文档从源码安装）。

## 训练

日志目录若不存在请先创建：

```bash
mkdir -p log
```

Phi-3 Vision 训练示例：

```bash
bash train_kl_phi3v.sh > log/realign-phi3v.log 2>&1
```

Qwen 训练示例（按需启用）：

```bash
bash train_kl_qwen.sh > log/realign-qwen.log 2>&1
```

## 评估

评估脚本中第二个参数为 GPU ID 列表（逗号分隔）。以下为四卡示例，可按机器实际情况增删或改为单卡（例如 `0`）。

```bash
bash sh/eval.sh realign-phi3v 0,1,2,3
bash sh/eval_qwen.sh realign-qwen 0,1,2,3
```
