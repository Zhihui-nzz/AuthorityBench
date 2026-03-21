import json
import random
import itertools
import re
from collections import defaultdict

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


MODEL_DIR = "YOUR_MODEL_DIR"
EVAL_FILE = "LISTWISE_FILE"
OUTPUT_FILE = "OUTPUT_FILE"
BATCH_SIZE = 16
NUM_ANCHORS = 5


def build_prompt(entity1, entity2):
    return (
        f"You are a Wikipedia notability expert and a pop-culture sports expert. Please score the entity authority of each of the two given basketball entities "
        f"based on the following criteria, with a rating range of 0 to 9.\n\n"
        f"Entity authority refers to the public recognition, fame, historical significance, or widespread knowledge of the entity (person, place, organization, etc.).\n"
        f"Consider both entities together when assigning scores to ensure consistency.\n\n"
        f"QUESTION CONTEXT: These entities are the subjects of questions like 'Where is the birthplace of the basketball player...'\n\n"
        f"**TARGET ENTITIES**:\n"
        f"Entity A: {entity1}\n"
        f"Entity B: {entity2}\n\n"
        f"Your output MUST be a single, valid JSON object in the format: {{\"A\": X, \"B\": Y}}, where X and Y are integer scores from 0 to 9. "
        f"Example format: {{\"A\": 7, \"B\": 4}}. Do not add any explanation.\n"
        f"Your answer: "
    )


def extract_scores(response):
    text = response.strip()
    if pos := text.rfind("Your answer:"):
        text = text[pos + 12:].strip()
    if match := re.search(r'\{[^{}]*\}', text, re.DOTALL):
        try:
            d = json.loads(match.group())
            sa, sb = float(d.get("A", d.get("a"))), float(d.get("B", d.get("b")))
            if sa is not None and sb is not None and 0 <= sa <= 9 and 0 <= sb <= 9:
                return sa, sb
        except Exception:
            pass
    return None, None


def main():
    llm = LLM(model=MODEL_DIR, tokenizer=MODEL_DIR, max_num_seqs=256, trust_remote_code=True, tensor_parallel_size=4, dtype="bfloat16")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=64)

    __import__("os").makedirs(__import__("os").path.dirname(OUTPUT_FILE), exist_ok=True)

    completed = set()
    if __import__("os").path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    completed.add(json.loads(line).get("item_id", -1))

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
        pending = [(idx, item) for idx, item in enumerate(items) if idx not in completed]
        for idx, item in tqdm(pending):
            entities = item["question_entity"]
            n = len(entities)
            anchors = random.sample(range(n), NUM_ANCHORS)
            non_anchors = [i for i in range(n) if i not in anchors]

            tasks = [{"idx_a": ni, "idx_b": ai, "entity_a": entities[ni], "entity_b": entities[ai]}
                     for ni in non_anchors for ai in anchors]
            tasks += [{"idx_a": ai, "idx_b": aj, "entity_a": entities[ai], "entity_b": entities[aj]}
                      for ai, aj in itertools.combinations(anchors, 2)]

            # Batch inference
            results = []
            for i in range(0, len(tasks), BATCH_SIZE):
                batch = tasks[i:i + BATCH_SIZE]
                prompts = [tokenizer.apply_chat_template([{"role": "user", "content": build_prompt(t["entity_a"], t["entity_b"])}],
                                                         tokenize=False, add_generation_prompt=True, enable_thinking=False) for t in batch]
                outputs = llm.generate(prompts, sampling_params)
                for t, out in zip(batch, outputs):
                    raw = out.outputs[0].text if out.outputs else ""
                    sa, sb = extract_scores(raw)
                    results.append((t, sa, sb, raw))

            # Aggregate scores
            scores = defaultdict(list)
            records = []
            for task, sa, sb, raw in results:
                if sa is not None and sb is not None:
                    scores[task["idx_a"]].append(sa)
                    scores[task["idx_b"]].append(sb)
                records.append({"entity_a": task["entity_a"], "idx_a": task["idx_a"], "entity_b": task["entity_b"],
                               "idx_b": task["idx_b"], "score_a": sa, "score_b": sb, "llm_output": raw})

            avg = {i: float(__import__("numpy").mean(scores[i])) if scores[i] else 0.0 for i in range(n)}
            pred = sorted(range(n), key=lambda x: avg[x], reverse=True)

            out_f.write(json.dumps({
                "item_id": idx, "question_entity": entities, "score": item.get("score"),
                "ground_truth": item["ground_truth"], "anchor_indices": anchors,
                "avg_scores": avg, "pred": pred, "comparisons": records
            }, ensure_ascii=False) + "\n")
            out_f.flush()


if __name__ == "__main__":
    main()
