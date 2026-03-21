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


def build_pairwise_prompt(entity1, entity2):
    return (
        f"You are a Wikipedia notability expert and a music industry expert. Please determine which of the two given entities is more authoritative based on the following criteria.\n\n"
        f"Entity authority refers to the public recognition, fame, historical significance, or widespread knowledge of the entity (person, place, organization, etc.).\n"
        f"Consider both entities together when assigning scores to ensure consistency.\n\n"
        f"**QUESTION CONTEXT**: These entities are the subjects of questions like 'Who is the performer of the song ...'\n\n"
        f"**TARGET ENTITIES**:\n"
        f"Entity A: \"{entity1}\"\n"
        f"Entity B: \"{entity2}\"\n\n"
        f"Which entity is more authoritative? Only return 'A' or 'B'. Do not add any explanation.\n"
        f"Your answer: "
    )


def extract_ab_answer(response):
    if not response:
        return None
    text = response.strip()
    first = text[0].upper() if text else ""
    if first in ("A", "B"):
        return first
    if m := re.search(r"\b([AB])\b", text.upper()):
        return m.group(1)
    return None


def run_batch_inference(task_list, llm, tokenizer, sampling_params):
    results = []
    for i in range(0, len(task_list), BATCH_SIZE):
        batch = task_list[i:i + BATCH_SIZE]
        prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": build_pairwise_prompt(task["entity_a"], task["entity_b"])}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False
        ) for task in batch]
        for task, out in zip(batch, llm.generate(prompts, sampling_params)):
            raw = out.outputs[0].text if out.outputs else ""
            winner = extract_ab_answer(raw)
            results.append((winner, raw, winner is None))
    return results


def anchor_compare(i, j, anchor_set, anchor_wins, non_anchor_wins, direct_results):
    if i not in anchor_set and j not in anchor_set:
        return non_anchor_wins.get(i, 0) > non_anchor_wins.get(j, 0)
    if i in anchor_set and j in anchor_set:
        return anchor_wins.get(i, 0) > anchor_wins.get(j, 0)
    if (i, j) in direct_results:
        return direct_results[(i, j)]
    if (j, i) in direct_results:
        return not direct_results[(j, i)]
    return anchor_wins.get(i, non_anchor_wins.get(i, 0)) > anchor_wins.get(j, non_anchor_wins.get(j, 0))


def bubble_sort_by_anchor(indices, anchor_set, anchor_wins, non_anchor_wins, direct_results):
    arr = list(indices)
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if not anchor_compare(arr[j], arr[j + 1], anchor_set, anchor_wins, non_anchor_wins, direct_results):
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def main():
    llm = LLM(model=MODEL_DIR, tokenizer=MODEL_DIR, max_num_seqs=256, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0)

    os = __import__("os")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    completed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        if "list_id" in (obj := json.loads(line)):
                            completed_ids.add(obj["list_id"])
                    except json.JSONDecodeError:
                        pass

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
        for item_idx, item in enumerate(tqdm(items)):
            list_id = item.get("list_id", item_idx)
            if list_id in completed_ids:
                continue

            entities = item["question_entity"]
            n = len(entities)
            anchor_indices = random.sample(range(n), NUM_ANCHORS)
            anchor_set = set(anchor_indices)
            non_anchor_indices = [i for i in range(n) if i not in anchor_set]

            na_vs_a_tasks = [{"idx_a": ni, "entity_a": entities[ni], "idx_b": ai, "entity_b": entities[ai], "type": "non_anchor_vs_anchor"}
                              for ni in non_anchor_indices for ai in anchor_indices]
            a_vs_a_tasks = [{"idx_a": ai, "entity_a": entities[ai], "idx_b": aj, "entity_b": entities[aj], "type": "anchor_vs_anchor"}
                             for ai, aj in itertools.combinations(anchor_indices, 2)]

            infer_results = run_batch_inference(na_vs_a_tasks + a_vs_a_tasks, llm, tokenizer, sampling_params)

            direct_results, non_anchor_wins, anchor_wins, comparison_records = {}, defaultdict(int), defaultdict(int), []
            for task, (winner, raw, failed) in zip(na_vs_a_tasks + a_vs_a_tasks, infer_results):
                idx_a, idx_b = task["idx_a"], task["idx_b"]
                if not failed:
                    i_wins = winner == "A"
                    direct_results[(idx_a, idx_b)] = i_wins
                    direct_results[(idx_b, idx_a)] = not i_wins
                    if task["type"] == "non_anchor_vs_anchor" and i_wins:
                        non_anchor_wins[idx_a] += 1
                    elif task["type"] == "anchor_vs_anchor":
                        if i_wins:
                            anchor_wins[idx_a] += 1
                        else:
                            anchor_wins[idx_b] += 1

                comparison_records.append({
                    "entity_a": task["entity_a"], "idx_a": idx_a,
                    "entity_b": task["entity_b"], "idx_b": idx_b,
                    "winner": winner, "extraction_failed": failed, "llm_output": raw, "type": task["type"]
                })

            pred = bubble_sort_by_anchor(range(n), anchor_set, anchor_wins, non_anchor_wins, direct_results)

            out_f.write(json.dumps({
                "list_id": list_id, "question_entity": entities, "score": item["score"],
                "ground_truth": item["ground_truth"], "anchor_indices": anchor_indices,
                "anchor_wins_vs_anchors": {str(k): v for k, v in anchor_wins.items()},
                "non_anchor_wins_vs_anchors": {str(k): v for k, v in non_anchor_wins.items()},
                "pred": pred, "comparisons": comparison_records
            }, ensure_ascii=False) + "\n")
            out_f.flush()


if __name__ == "__main__":
    main()
