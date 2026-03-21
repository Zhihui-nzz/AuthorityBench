import json
import os
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL_DIR = "YOUR_MODEL_DIR"
EVAL_PAIR_FILE = "EVAL_FILE"
OUTPUT_FILE = "OUTPUT_DIR"
BATCH_SIZE = 8

# Extract scoring answer from model response
def extract_scoring_answer(response_raw: str, num_domains: int = 10):
    """
    Extract JSON scoring from model output and return ranked indices.
    Returns (ranking_list, failed_flag).
    """
    response = response_raw.strip()

    marker = "Your answer:"
    marker_pos = response.rfind(marker)
    if marker_pos != -1:
        response = response[marker_pos + len(marker):].strip()

    json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
    if json_match:
        try:
            score_dict = json.loads(json_match.group())
            scores = {
                int(re.sub(r'\D', '', str(k))): float(v)
                for k, v in score_dict.items()
            }
            if len(scores) == num_domains and set(scores.keys()) == set(range(num_domains)):
                # Sort by score descending and return original indices
                return sorted(scores.keys(), key=lambda x: scores[x], reverse=True), False
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

    random_ranking = list(range(num_domains))
    random.shuffle(random_ranking)
    return random_ranking, True


def build_scoring_prompt(domains, doc_texts):
    """Build prompt for listwise domain authority scoring."""
    truncated_texts = [
        text[:500] + "..." if len(text) > 500 else text
        for text in doc_texts
    ]

    prompt = (
        f"You are a web analysis expert. Please score the source authority of each domain "
        f"in the provided list based on the following criteria, with a rating range of 0 to 9.\n"
        f"Source authority refers to the credibility, professionalism, official certification, or public recognition of the domain itself and its content.\n"
        f"Consider all domains and their contents together when assigning scores to ensure consistency and relative calibration.\n\n"
        f"**DOMAINS & CONTENTS**:\n"
    )
    for i, domain in enumerate(domains):
        prompt += f"[{i}] {domain}: {truncated_texts[i]}\n"

    prompt += (
        f"\n"
        f"Your output MUST be a single, valid JSON object. The JSON object must map each domain's numerical identifier (as a string) to its integer score (0-9).\n"
        f"Example format: {{\"0\": 7, \"1\": 4, ..., \"9\": 8}}. Do not add any explanation.\n"
        f"Your answer: "
    )
    return prompt


def main():
    llm = LLM(model=MODEL_DIR, tokenizer=MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=256,
    )

    with open(EVAL_PAIR_FILE, 'r', encoding='utf-8') as f:
        listwise_items = [json.loads(line) for line in f]

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for i in range(0, len(listwise_items), BATCH_SIZE):
            batch = listwise_items[i:i + BATCH_SIZE]

            prompts = []
            for item in batch:
                prompt = build_scoring_prompt(item["domains"], item["doc_texts"])
                messages = [{"role": "user", "content": prompt}]
                chat_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                prompts.append(chat_prompt)

            outputs = llm.generate(prompts, sampling_params)

            for item, output in zip(batch, outputs):
                response_raw = output.outputs[0].text if output.outputs else ""
                pred, failed = extract_scoring_answer(response_raw, len(item["domains"]))

                result = {
                    "domains": item["domains"],
                    "pageranks": item["pageranks"],
                    "ground_truth": item["ground_truth"],
                    "pred": pred,
                    "extraction_failed": failed,
                    "llm_output": response_raw
                }

                out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                out_f.flush()

    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
