import json
import random
import re
from typing import List, Dict, Any

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL_DIR = "YOUR_MODEL_DIR"
EVAL_PAIR_FILE = "LISTWISE_FILE"
OUTPUT_FILE = "YOUR_MODEL_DIR"
BATCH_SIZE = 16


def build_prompt(domains: List[str]) -> str:
    return (
        f"You are a web analysis expert. Please score the source authority of each domain "
        f"in the provided list based on the following criteria, with a rating range of 0 to 9.\n"
        f"Source authority refers to the credibility, professionalism, official certification, or public recognition of the domain itself.\n"
        f"Consider all domains together when assigning scores to ensure consistency and relative calibration.\n\n"
        f"**DOMAINS**:\n"
        + "\n".join([f"[{i}] {domain}" for i, domain in enumerate(domains)])
        + f"\n\nYour output MUST be a single, valid JSON object. The JSON object must map each domain's numerical identifier (as a string) to its integer score (0-9).\n"
        f"Example format: {{\"0\": 7, \"1\": 4, ..., \"9\": 8}}. Do not add any explanation.\n"
        f"Your answer: "
    )


def extract_scores(response: str, num_domains: int) -> List[int]:
    text = response.strip()
    if pos := text.rfind("Your answer:"):
        text = text[pos + 12:].strip()
    if match := re.search(r'\{[^{}]*\}', text, re.DOTALL):
        try:
            scores = {int(re.sub(r'\D', '', str(k))): float(v) for k, v in json.loads(match.group()).items()}
            if len(scores) == num_domains and set(scores.keys()) == set(range(num_domains)):
                return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        except Exception:
            pass
    ranking = list(range(num_domains))
    random.shuffle(ranking)
    return ranking


def main():
    llm = LLM(model=MODEL_DIR, tokenizer=MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=256)

    with open(EVAL_PAIR_FILE, 'r', encoding='utf-8') as f:
        items = [json.loads(line) for line in f]

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for i in tqdm(range(0, len(items), BATCH_SIZE)):
            batch = items[i:i + BATCH_SIZE]
            prompts = [tokenizer.apply_chat_template([{"role": "user", "content": build_prompt(item["domains"])}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for item in batch]
            outputs = llm.generate(prompts, sampling_params)
            for item, output in zip(batch, outputs):
                text = output.outputs[0].text if output.outputs else ""
                pred = extract_scores(text, len(item["domains"]))
                out_f.write(json.dumps({"domains": item["domains"], "pageranks": item["pageranks"], "ground_truth": item["ground_truth"], "pred": pred, "llm_output": text}, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main()
