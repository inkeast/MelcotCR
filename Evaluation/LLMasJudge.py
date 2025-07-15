import os
import random
import sys
import pandas as pd
import fire
import torch
from transformers import GenerationConfig, AutoModel, AutoTokenizer
from tqdm import tqdm
import jsonlines
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import re


prompt_template = """Analyze the given review comment and the standard review comment. \
Determine if they highlight the same core issue(s), even if phrased differently. \
Focus on the substance of the critique, not exact wording. \
If they address identical problems (e.g., methodology flaws, data limitations, clarity issues), output Review Consistent. \
If they diverge in the issues raised (e.g., one critiques methodology while the other focuses on formatting), output Review Inconsistent.

standard review comment:
```
{}
```

given review comment:
```
{}
```"""

def process_output(outputs):
    for i in range(len(outputs)):
        outputs[i] = outputs[i].outputs[0].text
    return outputs


def main(
    base_model: str = 'huggyllama/llama-13b',
    test_dataset: str = "Data/LongCoTTestDatasetShort.parquet", # The test dataset to use.
    target_line_name: str = "review_comment"
):
    df = pd.read_parquet(test_dataset)
    pattern_comment = re.compile(r'.*<comment>(.*)</comment>', re.DOTALL)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    prompts = []

    for i, row in df.iterrows():
        stand_review = row["review_comment"]
        given_review = pattern_comment.search(row[target_line_name])

        if given_review:
            given_review = given_review.group(1)

        prompt = prompt_template.format(stand_review, given_review)

        user = [
            {"role": "user", "content": prompt},
        ]
        prompts.append(tokenizer.apply_chat_template(user, tokenize=False, add_generation_prompt=True))

    llm = LLM(model=base_model, max_model_len=10000, gpu_memory_utilization=0.9, tensor_parallel_size=8)

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=4000
    )

    outputs = llm.generate(
        prompts,
        sampling_params
    )

    outputs = process_output(outputs)

    line_name = target_line_name + "_LLM_Evaluate"

    if line_name in df.columns:
        df[line_name] = outputs
    else:
        df.insert(2, line_name, outputs)

    if test_dataset.endswith(".xlsx"):
        df.to_excel(test_dataset)
    elif test_dataset.endswith(".parquet"):
        df.to_parquet(test_dataset)


if __name__ == "__main__":
    fire.Fire(main)


