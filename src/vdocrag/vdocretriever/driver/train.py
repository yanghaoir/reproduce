import logging
import os
import sys
import torch
import wandb

from transformers import AutoTokenizer
from transformers import AutoProcessor 

from transformers import (
    HfArgumentParser,
    set_seed,
)

from vdocrag.vdocretriever.arguments import ModelArguments, DataArguments, \
    VDocRetrieverTrainingArguments as TrainingArguments
from vdocrag.vdocretriever.dataset import TrainDataset
from vdocrag.vdocretriever.collator import TrainCollator
from vdocrag.vdocretriever.modeling import VDocRetriever
from vdocrag.vdocretriever.trainer import VDocRetrieverTrainer as Trainer

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    model_name = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    is_qwen = 'qwen' in model_name.lower()

    processor_kwargs = dict(cache_dir=model_args.cache_dir, trust_remote_code=True)
    if is_qwen:
        processor_kwargs.update(min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)

    processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
    tokenizer = processor.tokenizer

    if is_qwen and tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f"Set Qwen pad_token_id to eos_token_id: {tokenizer.eos_token_id}")

    if training_args.bf16:
        torch_dtype = torch.bfloat16
    elif training_args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # 此处为修改的代码
    # 传入data_args到模型build方法，以获取kl_loss_weight等参数
    model = VDocRetriever.build(
        model_args,
        training_args,
        data_args=data_args,
        cache_dir=model_args.cache_dir,
        trust_remote_code=True,
        torch_dtype=torch_dtype, 
        _attn_implementation='eager' if data_args.pretrain else 'flash_attention_2',
    )

    train_dataset = TrainDataset(data_args)
    collator = TrainCollator(data_args, tokenizer, processor)

    trainer_cls = Trainer

    # 此处为修改的代码
    # 传入data_args到trainer，以访问training_mode等参数
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        data_args=data_args
    )
    train_dataset.trainer = trainer

    trainer.train()  # TODO: resume training
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
