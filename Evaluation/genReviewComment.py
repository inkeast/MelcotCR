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

prompt_origin = """Review the given code and provide a constructive code review comment.\nThe code/(diff hunk) is: '{} '"""

review_prompt = """Please review the code thoroughly and provide structured feedback using the following XML-style format for each identified issue:  

For each code issue:  
1. Specify exact code location
2. Include four mandatory elements in comments:  
   - Technical analysis of the issue  
   - Explanation of the rationale for determining the issue in the specified code 
   - Potential impact assessment  
   - Actionable solution proposal  

Example Format:  
```
<location>
   ```python
   xxx
   ```
</location>
<comment>
   - Analysis: xxx
   - Professional Explanation: xxx
   - Issue Identification: xxx
   - Recommendation: xxx
</comment>
```
```

Code:
```{}
{}
```"""

prompt_new = """You are a Senior Software Engineer performing critical code review. You have 15+ years of experience in diagnosing complex systems and identifying subtle code defects. Execute this task with forensic-level scrutiny.

**Task**  
Analyze the provided code to identify exact locations of the defect through systematic scrutiny and then comment the exact defect. Please follow the process below:

1. **Systematic Code Dissection**  
   <think>  
   - Begin with code summarization: Explain the core functionality in 2-3 sentences  
   - Identify key code logic and critical execution paths  
   - Analyze recent diffs (if available) for regression patterns  
   - Examine error (or exception) handling and edge case coverage  
   - Check resource management (memory, connections, etc.)  
   - Evaluate API/dependency usage correctness  
   - Cross-reference with common vulnerability patterns (OWASP, CWE)  
   </think>

2. **Defect Localization**  
   <location>  
   Paste EXACT problematic code snippet (1-5 lines maximum)  
   </location>

3. **Structured Report**  
   <comment>  
   **Technical analysis**:  
   - Explain the nature of the defect  
   - Reference specific language semantics/APIs involved  

   **Root cause rationale**:  
   - Evidence chain demonstrating defect viability 
   - Reference standards/violated best practices  (CERT/ISO/IEC, etc)

   **Impact assessment**:  
   - Current consequences  
   - Potential worst-case scenarios  

   **Corrective proposal**:  
   - Specific code-level fix  
   - Alternative approaches (if applicable)  
   - Recommended prevention patterns  
   </comment>

**Example Output Structure**:  
```xml
<think>  
[Detailed analytical process showing the logical progression of identifying the defect]  
</think>  

<location>  
[Exact problematic code copy/pasted]  
</location>  

<comment>  
**Technical analysis**:  
...  
**Root cause rationale**:  
...  
**Impact assessment**:  
...  
**Corrective proposal**:  
...  
</comment>
```
Code:
```{}
{}
```
"""


def process_output(outputs):
    for i in range(len(outputs)):
        outputs[i] = outputs[i].outputs[0].text
    return outputs


def main(
    base_model: str = 'huggyllama/llama-13b',
    test_dataset: str = "Data/LongCoTTestDatasetShort.parquet", # The test dataset to use.
    lora: str = "",
    out_file: str = "",
    line_name: str = "",
):
    df = pd.read_parquet(test_dataset)

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    prompts = []

    for i, row in df.iterrows():
        if "LCot" in line_name:
            prompt = row["instruction"]
        elif "Prompt_New" in line_name:
            prompt = prompt_new.format(row["language"],row["diff"])
        elif "Origin" in line_name:
            prompt = prompt_origin.format(row["diff"])
        else:
            prompt = review_prompt.format(row["language"],row["diff"])
        user = [
            {"role": "user", "content": prompt},
        ]
        prompts.append(tokenizer.apply_chat_template(user, tokenize=False, add_generation_prompt=True))

    if lora:
        llm = LLM(model=base_model, enable_lora=True, max_model_len=8000, gpu_memory_utilization=0.9, tensor_parallel_size=1)

    else:
        llm = LLM(model=base_model, max_model_len=10000, gpu_memory_utilization=0.9, tensor_parallel_size=1)

    sampling_params = SamplingParams(
        temperature=0.2,
        max_tokens=4000
    )

    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest("t",1,lora) if lora else None,
    )

    outputs = process_output(outputs)

    if line_name in df.columns:
        df[line_name] = outputs
    else:
        df.insert(2, line_name, outputs)

    if out_file.endswith(".xlsx"):
        df.to_excel(out_file)
    elif out_file.endswith(".parquet"):
        df.to_parquet(out_file)


if __name__ == "__main__":
    fire.Fire(main)


