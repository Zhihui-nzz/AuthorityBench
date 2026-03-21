import json
import os
import itertools
from collections import defaultdict
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import re

MODEL_DIR = "YOUR_MODEL_DIR"
EVAL_FILE = "PAIRWISE_FILE
OUTPUT_FILE = "OUTPUT_DIR"
BATCH_SIZE = 8
MAX_TEXT_LENGTH = 500


def build_pairwise_prompt(domain1: str, text1: str, domain2: str, text2: str) -> str:
    t1 = (text1[:MAX_TEXT_LENGTH] + "...") if text1 and len(text1) > MAX_TEXT_LENGTH else (text1 or "[No content available]")
    t2 = (text2[:MAX_TEXT_LENGTH] + "...") if text2 and len(text2) > MAX_TEXT_LENGTH else (text2 or "[No content available]")

    prompt = (
        f"You are a web analysis expert. Please score the source authority of each of the two given domains "
        f"based on the following criteria, with a rating range of 0 to 9.\n\n"
        f"Source authority refers to the credibility, professionalism, official certification, or public recognition of the domain itself and its content.\n"
        f"Consider both domains and their contents together when assigning scores to ensure consistency.\n\n"
        f"**DOMAINS**:\n"
        f"Domain A: {domain1}\n"
        f"Content A: {t1}\n\n"
        f"Domain B: {domain2}\n"
        f"Content B: {t2}\n\n"
        f"Your output MUST be a single, valid JSON object in the format: {{\"A\": X, \"B\": X}}, where each X is an integer score from 0 to 9. "
        f"Example format: {{\"A\": 7, \"B\": 4}}. Do not add any explanation.\n"
        f"Your answer: "
    )
    return prompt


def extract_pairwise_scores(response_raw: str):
    response = response_raw.strip()

    marker = "Your answer: "
    pos = response.rfind(marker)
    if pos != -1:
        response = response[pos + len(marker):].strip()

    json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
    if json_match:
        try:
            d = json.loads(json_match.group())
            sa = float(d.get("A", d.get("a", None)))
            sb = float(d.get("B", d.get("b", None)))
            if sa is not None and sb is not None:
                return sa, sb
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    return None, None


def run_batch_inference(task_list, llm, tokenizer, sampling_params):
    results = []
    for i in range(0, len(task_list), BATCH_SIZE):
        batch = task_list[i:i + BATCH_SIZE]
        prompts = []

        for task in batch:
            p = build_pairwise_prompt(task["domain_a"], task["text_a"], task["domain_b"], task["text_b"])
            msg = [{"role": "user", "content": p}]
            chat_p = tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            prompts.append(chat_p)

        outputs = llm.generate(prompts, sampling_params)

        for task, out in zip(batch, outputs):
            raw = out.outputs[0].text if out.outputs else ""
            sa, sb = extract_pairwise_scores(raw)
            failed = (sa is None or sb is None)
            results.append((sa, sb, raw, failed))

    return results


def main():
    llm = LLM(model=MODEL_DIR, tokenizer=MODEL_DIR, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=256,
    )

    with open(EVAL_FILE, 'r', encoding='utf-8') as f:
        listwise_items = [json.loads(line) for line in f]

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for item in listwise_items:
            domains = item["domains"]
            doc_texts = item.get("doc_texts", [""] * len(domains))
            pageranks = item["pageranks"]
            ground_truth = item["ground_truth"]
            n = len(domains)

            all_tasks = []
            for i, j in itertools.combinations(range(n), 2):
                all_tasks.append({
                    "idx_a": i, "domain_a": domains[i], "text_a": doc_texts[i],
                    "idx_b": j, "domain_b": domains[j], "text_b": doc_texts[j]
                })

            infer_results = run_batch_inference(all_tasks, llm, tokenizer, sampling_params)

            score_lists = defaultdict(list)
            comparison_records = []

            for task, (sa, sb, raw, failed) in zip(all_tasks, infer_results):
                if not failed:
                    score_lists[task["idx_a"]].append(sa)
                    score_lists[task["idx_b"]].append(sb)

                comparison_records.append({
                    "domain_a": task["domain_a"], "idx_a": task["idx_a"],
                    "domain_b": task["domain_b"], "idx_b": task["idx_b"],
                    "score_a": sa, "score_b": sb,
                    "extraction_failed": failed,
                    "llm_output": raw
                })

            avg_scores = {
                idx: float(sum(scores) / len(scores)) if scores else 0.0
                for idx, scores in score_lists.items()
            }

            pred = sorted(range(n), key=lambda x: avg_scores[x], reverse=True)

            result = {
                "domains": domains,
                "pageranks": pageranks,
                "ground_truth": ground_truth,
                "avg_scores": avg_scores,
                "pred": pred,
                "comparisons": comparison_records,
            }

            out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
            out_f.flush()

    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
