import os
import sys
import torch
import transformers
from datasets import load_dataset
import json
from transformers import AutoModelForCausalLM, AutoTokenizer  # 有些模型可以不能用Auto
from dataclasses import dataclass, field


os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 关闭并行化tokenize避免死锁


@dataclass
class ModelArguments:
    base_model: str = field(default=None)  # 使用的基础大模型


@dataclass
class DataArguments:
    data_path: str = field(default="./example_dataset/alpaca_data_gpt4.json")
    train_on_inputs: bool = field(default=True, metadata={"help": "将模型输入纳入训练"})
    add_eos_token: bool = field(default=False, metadata={"help": "添加结束符，用于强化模型停止输出能力"})
    cutoff_len: int = field(default=256, metadata={"help": "截断长度，切断超出该长度的Token，该值需要小于模型的Token上限，否则可能报错"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    per_device_train_batch_size: int = field(default=4, metadata={"help": "每次计算的batch"})
    gradient_accumulation_steps: int = field(default=32, metadata={"help": "进行梯度累积的次数"})
    warmup_steps: int = field(default=100, metadata={"help": "线性学习率变化，详见 https://stackoverflow.com/questions/55933867/what-does-learning-rate-warm-up-mean"})
    num_train_epochs: int = field(default=3, metadata={"help": "训练的轮数"})
    learning_rate: float = field(default=3e-4)
    bf16: bool = field(default=True, metadata={"help": "模型使用的数据结构，A系列及以后的设备(A100 A800 H800 RTX3090+)建议使用bfloat16，其他设备(V100 P40)使用float16"})  # 要与模型load时一致，如果load时使用torch.float16这里应该是fp16=True
    logging_steps: int = field(default=10, metadata={"help": "每隔多少步打印一次loss"})
    val_set_size: int = field(default=1000)
    save_strategy: str = field(default="epoch")
    evaluation_strategy: str = field(default="no")
    eval_steps: int = field(default=None)  # 每多少步进行一次验证
    save_steps: int = field(default=None)  # 每多少步存储一次checkpoint
    output_dir: str = field(default="./checkpoint", metadata={"help": "模型存储路径"})
    save_total_limit: int = field(default=10, metadata={"help": "最多存储多少个checkpoints"})
    group_by_length: bool = field(default=False, metadata={"help": "将数据按Token长度进行打包。由于在同一个mini_batch中，必须等待所有的数据推断完毕才可以停止，因此存在较大的长度差异会导致算力浪费，但相似长度的数据可能具有相似的内容，导致训练不稳定。"})


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()
    train_args.evaluation_strategy = "steps" if train_args.val_set_size > 0 else "no"


    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"base_model: {model_args.base_model}\n"
        f"data_path: {data_args.data_path}\n"
        f"output_dir: {train_args.output_dir}\n"
        f"batch_size: {train_args.per_device_train_batch_size*train_args.gradient_accumulation_steps}\n"
        f"micro_batch_size: {train_args.per_device_train_batch_size}\n"
        f"num_epochs: {train_args.num_train_epochs}\n"
        f"learning_rate: {train_args.learning_rate}\n"
        f"cutoff_len: {data_args.cutoff_len}\n"
        f"val_set_size: {train_args.val_set_size}\n"
        f"train_on_inputs: {data_args.train_on_inputs}\n"
        f"add_eos_token: {data_args.add_eos_token}\n"
        f"group_by_length: {train_args.group_by_length}\n"
    )
    assert (
        model_args.base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    tokenizer = AutoTokenizer.from_pretrained(model_args.base_model)  # 构建Tokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_args.base_model,
        torch_dtype=torch.bfloat16 if train_args.bf16 else torch.float16,  # 模型使用的数据结构，A系列及以后的设备(A100 A800 H800 RTX3090+)建议使用bfloat16，其他设备(V100 P40)使用float16
        attn_implementation="flash_attention_2",
        # device_map="auto",   # !!当使用ZeRO3级别的并行优化时不允许指定device_map,需要删除本行
        # 模型放置设备，详见 https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained.device_map
    )  # 默认情况下使用auto即可，使用可见的所有卡

    tokenizer.pad_token_id = (
        0  # pad id，绝大多数填充id均为0，但不排除有例外
    )

    tokenizer.padding_side = "left"  # 向右对齐，允许具体应用场景按batch进行推断（可选）

    # 用于将单个样例进行tokenize
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,  # 超出max_length的数据进行截断
            max_length=data_args.cutoff_len,
            padding=False,  # 不填充，训练时使用 data_collator 进行填充
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < data_args.cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()  # decoding模型的labels与输出一致
        return result

    # 处理数据集
    def generate_and_tokenize_prompt(data_point):
        full = [
            {"role": "user", "content": data_point["instruction"]},
            {"role": "assistant", "content": data_point["output"]},
        ]
        user = [
            {"role": "user", "content": data_point["instruction"]},
        ]
        full_prompt = tokenizer.apply_chat_template(full, tokenize=False, add_generation_prompt=True)
        user_prompt = tokenizer.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
        tokenized_full_prompt = tokenize(full_prompt)
        tokenized_full_prompt["labels"] = tokenized_full_prompt["input_ids"].copy()
        if not data_args.train_on_inputs:
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=data_args.add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if data_args.add_eos_token:
                user_prompt_len -= 1
            # 将不参与训练的答案label置为负数，pytorch不计算负数的loss
            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        return tokenized_full_prompt

    if data_args.data_path.endswith(".json") or data_args.data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_args.data_path)
    elif data_args.data_path.endswith(".parquet"):
        data = load_dataset('parquet', data_files=data_args.data_path)
    else:
        data = load_dataset(data_args.data_path)

    if train_args.val_set_size > 0:  # 切分测试集
        train_val = data["train"].train_test_split(
            test_size=train_args.val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt, num_proc=32)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt, num_proc=32)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt, num_proc=32)
        val_data = None

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=train_args,
        data_collator=transformers.DataCollatorForSeq2Seq(  # 用于动态padding
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    trainer.train()

    model.save_pretrained(train_args.output_dir)


if __name__ == "__main__":
    train()
