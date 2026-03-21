import json
import re
from typing import List, Dict, Any

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


MODEL_DIR = "YOUR_MODEL_DIR"
INPUT_FILE = "CORPUS_FILE"
OUTPUT_FILE = "OUTPUT_DIR"
BATCH_SIZE = 8


def load_data(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def build_prompt(domain: str) -> str:
    return (
        f"You are a web analysis expert. Please score the source authority of the given domain based on the following criteria, with a rating range of 0 to 9.\n\n"
        f"Source authority refers to the credibility, professionalism, official certification, or public recognition of the domain itself.\n\n"
        f"**DOMAIN**: {domain}\n"
        f"Based on the above criteria, rate the source authority of this DOMAIN (0-9 points).\n"
        f"Output only: {{\"score\": X}} where X is your evaluated score (0, 1, 2, 3, 4, 5, 6, 7, 8, or 9). Do not add any explanation.\n"
        f"Your answer: "
    )


def extract_score(text: str) -> int:
    text = text.strip()
    try:
        if match := re.search(r'\{[^}]*"score"\s*:\s*(\d+)[^}]*\}', text):
            score = int(match.group(1))
            if 0 <= score <= 9:
                return score
    except Exception:
        pass
    if numbers := re.findall(r'\b([0-9])\b', text):
        return int(numbers[0])
    return -1


def main():
    items = load_data(INPUT_FILE)
    llm = LLM(model=MODEL_DIR, tokenizer=MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=128)

    results = []
    for i in tqdm(range(0, len(items), BATCH_SIZE)):
        batch = items[i : i + BATCH_SIZE]
        messages = [[{"role": "user", "content": build_prompt(item["domain"])}] for item in batch]
        prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in messages]
        outputs = llm.generate(prompts, sampling_params)

        for item, output in zip(batch, outputs):
            text = output.outputs[0].text if output.outputs else ""
            pred = extract_score(text)
            results.append({"domain": item["domain"], "true_score": item["pagerank"], "pred_score": pred, "llm_output": text if pred != -1 else ""})

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
