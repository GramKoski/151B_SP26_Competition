import argparse
import os
import json

import re
import sys
from pathlib import Path
from typing import Optional


parser = argparse.ArgumentParser()
parser.add_argument("--gpu-id", type=int, required=True)
parser.add_argument("--chunk-id", type=int, required=True)
parser.add_argument("--num-chunks", type=int, required=True)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
DATA_PATH = "data/private.jsonl"
OUTPUT_PATH = f"results/private_chunk_{args.chunk_id}.jsonl"

with open(DATA_PATH) as f:
    data = [json.loads(line) for line in f]

n = len(data)
chunk_size = (n + args.num_chunks - 1) // args.num_chunks
start = args.chunk_id * chunk_size
end = min(start + chunk_size, n)
chunk = data[start:end]

SYSTEM_PROMPT_MATH = (
    "You are a mathematician solving a competition problem under strict constraints.\n\n"
    "CRITICAL RULES:\n"
    "1. You have LIMITED TOKENS. Be maximally efficient.\n"
    "2. Skip unnecessary explanations. Show ONLY the essential calculation steps.\n"
    "3. Do NOT verify your work multiple times. Calculate once, box the answer.\n"
    "4. If stuck after 3 steps, make your best guess and move on.\n\n"
    "Put your final answer inside \\boxed{}. Multiple answers separated by commas: \\boxed{3, 7}.\n"
    "Exact expressions are fine (e.g., \\boxed{323*(325+1)/7}). Never round unless requested."
    "Leave answer unsimplified as possible. The grader will accept many different numerically equivalent solutions, so don't worry about getting you answer into a specific form."
)

SYSTEM_PROMPT_MCQ = (
    "You are solving a multiple-choice math problem under strict token limits.\n\n"
    "STRATEGY:\n"
    "1. Read all answer choices FIRST.\n"
    "2. Use PROCESS OF ELIMINATION and EDUCATED GUESSING.\n"
    "3. If a choice looks obviously wrong, eliminate it immediately.\n"
    "4. If calculation is complex, estimate or use dimensional analysis.\n"
    "5. Pick the most reasonable answer within 2-3 steps of reasoning.\n\n"
    "Output ONLY the letter inside \\boxed{}, e.g., \\boxed{C}.\n"
    "Do NOT write multiple paragraphs. Do NOT verify. Just choose the best answer quickly."
    "Choose the MCQ answer that is closest. Do not worry about the answer being exact."
)


def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a question."""
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question



tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

llm = LLM(
    model=MODEL_ID,
    dtype="float16",
    enable_prefix_caching=False,
    gpu_memory_utilization=0.9,
    max_model_len=8192,
    trust_remote_code=True,
    max_num_seqs=2,
)

sampling_params = SamplingParams(
    max_tokens=4096,
    temperature=0.0,
    top_p=1.0,
    top_k=-1,
    min_p=0.0,
    presence_penalty=0.0,
    repetition_penalty=1.0,
)

print(f"GPU {args.gpu_id}: Processing chunk {args.chunk_id} - [{start}:{end})")

prompts = []
for item in chunk:
    system, user = build_prompt(item["question"], item.get("options"))
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user",   "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompts.append(prompt_text)

# Generate
outputs = llm.generate(prompts, sampling_params=sampling_params)

out_path = Path(OUTPUT_PATH)
out_path.parent.mkdir(parents=True, exist_ok=True)

with open(out_path, "w") as f:
    for item, out in zip(chunk, outputs):
        response = out.outputs[0].text.strip()
        record = {
            "id": item["id"],
            "is_mcq": item.get("options") is not None,
            "response": response,
        }
        f.write(json.dumps(record) + "\n")

print(f"GPU {args.gpu_id}: Saved chunk {args.chunk_id} - [{start}:{end}] responses to {out_path}")