import json
import random
import itertools
from collections import defaultdict

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


MODEL_DIR = "YOUR_MODEL_DIR"
EVAL_FILE = "PAIRWISE_FILE"
OUTPUT_FILE = "OUTPUT_FILE"
BATCH_SIZE = 16
NUM_ANCHORS = 5


def build_prompt(domain1, domain2):
    return (
        f"You are a web analysis expert. Please score the source authority of each of the two given domains "
        f"based on the following criteria, with a rating range of 0 to 9.\n\n"
        f"Source authority refers to the credibility, professionalism, official certification, or public recognition of the domain itself.\n"
        f"Consider both domains together when assigning scores to ensure consistency.\n\n"
        f"**DOMAINS**:\n"
        f"Domain A: {domain1}\n"
        f"Domain B: {domain2}\n\n"
        f"Your output MUST be a single, valid JSON object in the format: {{\"A\": X, \"B\": X}}, where each X is an integer score from 0 to 9. "
        f"Example format: {{\"A\": 7, \"B\": 4}}. Do not add any explanation.\n"
        f"Your answer: "
    )


def extract_scores(response):
    text = response.strip()
    if pos := text.rfind("Your answer:"):
        text = text[pos + 12:].strip()
    if match := __import__("re").search(r'\{[^{}]*\}', text, __import__("re").DOTALL):
        try:
            d = json.loads(match.group())
            sa, sb = float(d.get("A", d.get("a"))), float(d.get("B", d.get("b")))
            return (sa, sb) if sa is not None and sb is not None else (None, None)
        except Exception:
            pass
    return None, None


def run_inference(tasks, llm, tokenizer, sampling_params):
    results = []
    for i in range(0, len(tasks), BATCH_SIZE):
        batch = tasks[i:i + BATCH_SIZE]
        prompts = [tokenizer.apply_chat_template([{"role": "user", "content": build_prompt(t["domain_a"], t["domain_b"])}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for t in batch]
        outputs = llm.generate(prompts, sampling_params)
        for t, out in zip(batch, outputs):
            raw = out.outputs[0].text if out.outputs else ""
            sa, sb = extract_scores(raw)
            results.append((t["idx_a"], t["idx_b"], sa, sb, raw))
    return results


def main():
    llm = LLM(model=MODEL_DIR, tokenizer=MODEL_DIR, tensor_parallel_size=4, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=64)

    with open(EVAL_FILE, 'r', encoding='utf-8') as f:
        items = [json.loads(line) for line in f]

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for item in tqdm(items):
            domains, n = item["domains"], len(domains)
            anchors = random.sample(range(n), NUM_ANCHORS)
            non_anchors = [i for i in range(n) if i not in anchors]

            tasks = [{"idx_a": ni, "idx_b": ai, "domain_a": domains[ni], "domain_b": domains[ai]} for ni in non_anchors for ai in anchors]
            tasks += [{"idx_a": ai, "idx_b": aj, "domain_a": domains[ai], "domain_b": domains[aj]} for ai, aj in itertools.combinations(anchors, 2)]

            scores = defaultdict(list)
            records = []
            for idx_a, idx_b, sa, sb, raw in run_inference(tasks, llm, tokenizer, sampling_params):
                if sa is not None and sb is not None:
                    scores[idx_a].append(sa)
                    scores[idx_b].append(sb)
                records.append({"domain_a": domains[idx_a], "idx_a": idx_a, "domain_b": domains[idx_b], "idx_b": idx_b, "score_a": sa, "score_b": sb, "llm_output": raw})

            avg = {i: float(__import__("numpy").mean(scores[i])) if scores[i] else 0.0 for i in range(n)}
            pred = sorted(range(n), key=lambda x: avg[x], reverse=True)

            out_f.write(json.dumps({"domains": domains, "pageranks": item["pageranks"], "ground_truth": item["ground_truth"], "anchor_indices": anchors, "avg_scores": avg, "pred": pred, "comparisons": records}, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main()
