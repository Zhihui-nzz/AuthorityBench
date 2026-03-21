import json
import random
import itertools
import re
from collections import defaultdict

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


MODEL_DIR = "YOUR_MODEL_DIR"
EVAL_FILE = "EVAL_FILE"
OUTPUT_FILE = "OUTPUT_FILE"
BATCH_SIZE = 16
NUM_ANCHORS = 5


def build_prompt(domain1, domain2):
    return (
        f"You are a web analysis expert. Please determine which of the two given domains "
        f"is more authoritative as an information source based on the following criteria.\n\n"
        f"Source authority refers to the credibility, professionalism, official certification, or public recognition of the domain itself.\n"
        f"Consider both domains together when making the comparison.\n\n"
        f"**DOMAINS**:\n"
        f"Domain A: {domain1}\n"
        f"Domain B: {domain2}\n\n"
        f"Which domain is more authoritative? Only return 'A' or 'B'. Do not add any explanation.\n"
        f"Your answer: "
    )


def extract_winner(response):
    if not response:
        return None
    text = response.strip().upper()
    first = text[0] if text else ""
    if first in ("A", "B"):
        return first
    if match := re.search(r'\b([AB])\b', text):
        return match.group(1)
    return None


def compare(i, j, anchor_set, anchor_wins, non_anchor_wins, direct):
    i_is_anchor = i in anchor_set
    j_is_anchor = j in anchor_set

    if not i_is_anchor and not j_is_anchor:
        return non_anchor_wins.get(i, 0) > non_anchor_wins.get(j, 0)
    elif i_is_anchor and j_is_anchor:
        return anchor_wins.get(i, 0) > anchor_wins.get(j, 0)
    else:
        if (i, j) in direct:
            return direct[(i, j)]
        elif (j, i) in direct:
            return not direct[(j, i)]
        else:
            score_i = anchor_wins.get(i, non_anchor_wins.get(i, 0))
            score_j = anchor_wins.get(j, non_anchor_wins.get(j, 0))
            return score_i > score_j


def bubble_sort(indices, anchor_set, anchor_wins, non_anchor_wins, direct):
    arr = list(indices)
    n = len(arr)
    for i in range(n):
        for j in range(n - i - 1):
            if not compare(arr[j], arr[j + 1], anchor_set, anchor_wins, non_anchor_wins, direct):
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def main():
    llm = LLM(model=MODEL_DIR, tokenizer=MODEL_DIR, tensor_parallel_size=4, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=16)

    __import__("os").makedirs(__import__("os").path.dirname(OUTPUT_FILE), exist_ok=True)

    completed = set()
    if __import__("os").path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        if "list_id" in (obj := json.loads(line)):
                            completed.add(obj["list_id"])
                    except Exception:
                        pass

    with open(EVAL_FILE, 'r', encoding='utf-8') as f:
        items = [json.loads(line) for line in f]

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as out_f:
        for idx, item in enumerate(tqdm(items)):
            list_id = item.get("list_id", idx)
            if list_id in completed:
                continue

            domains, n = item["domains"], len(domains)
            anchors = random.sample(range(n), NUM_ANCHORS)
            anchor_set = set(anchors)
            non_anchors = [i for i in range(n) if i not in anchor_set]

            tasks = [{"idx_a": ni, "idx_b": ai, "domain_a": domains[ni], "domain_b": domains[ai], "type": "na_vs_a"}
                     for ni in non_anchors for ai in anchors]
            tasks += [{"idx_a": ai, "idx_b": aj, "domain_a": domains[ai], "domain_b": domains[aj], "type": "a_vs_a"}
                      for ai, aj in itertools.combinations(anchors, 2)]

            results = []
            for i in range(0, len(tasks), BATCH_SIZE):
                batch = tasks[i:i + BATCH_SIZE]
                prompts = [tokenizer.apply_chat_template([{"role": "user", "content": build_prompt(t["domain_a"], t["domain_b"])}],
                                                         tokenize=False, add_generation_prompt=True, enable_thinking=False) for t in batch]
                outputs = llm.generate(prompts, sampling_params)
                for t, out in zip(batch, outputs):
                    raw = out.outputs[0].text if out.outputs else ""
                    winner = extract_winner(raw)
                    results.append((t, winner, raw))

            direct = {}
            anchor_wins = defaultdict(int)
            non_anchor_wins = defaultdict(int)
            records = []

            for task, winner, raw in results:
                idx_a, idx_b = task["idx_a"], task["idx_b"]
                if winner:
                    i_wins = (winner == "A")
                    direct[(idx_a, idx_b)] = i_wins
                    direct[(idx_b, idx_a)] = not i_wins

                    if task["type"] == "na_vs_a":
                        non_anchor_wins[idx_a] += 1 if i_wins else 0
                    elif task["type"] == "a_vs_a":
                        anchor_wins[idx_a] += 1 if i_wins else 0
                        anchor_wins[idx_b] += 1 if not i_wins else 0

                records.append({"domain_a": task["domain_a"], "idx_a": idx_a, "domain_b": task["domain_b"],
                               "idx_b": idx_b, "winner": winner, "llm_output": raw, "type": task["type"]})

            for ni in non_anchors:
                if ni not in non_anchor_wins:
                    non_anchor_wins[ni] = 0
            for ai in anchors:
                if ai not in anchor_wins:
                    anchor_wins[ai] = 0

            pred = bubble_sort(range(n), anchor_set, anchor_wins, non_anchor_wins, direct)

            out_f.write(json.dumps({
                "list_id": list_id, "domains": domains, "pageranks": item["pageranks"],
                "ground_truth": item["ground_truth"], "anchor_indices": anchors,
                "anchor_wins_vs_anchors": {str(k): v for k, v in anchor_wins.items()},
                "non_anchor_wins_vs_anchors": {str(k): v for k, v in non_anchor_wins.items()},
                "pred": pred, "comparisons": records
            }, ensure_ascii=False) + '\n')
            out_f.flush()


if __name__ == "__main__":
    main()
