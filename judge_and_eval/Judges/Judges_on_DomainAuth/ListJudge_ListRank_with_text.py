import json
import os
import re
import random
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL_DIR = "YOUR_MODEL_DIR"
EVAL_PAIR_FILE = "EVAL_FILE"
OUTPUT_FILE = "OUTPUT_DIR"
BATCH_SIZE = 8


def build_ranking_prompt(domains, doc_texts):
    truncated_texts = [
        text[:500] + "..." if len(text) > 500 else text
        for text in doc_texts
    ]

    prompt = (
        f"You are a web analysis expert that can rank domains based on their authority.\n\n"
        f"I will provide you with 10 domains and their web contents, each indicated by a numerical identifier [].\n"
        f"Rank the domains based on their source authority (credibility, professionalism, official certification, or public recognition).\n"
    )

    for i in range(10):
        prompt += f"[{i}] {domains[i]}: {truncated_texts[i]}\n"

    prompt += (
        f"\nRank the 10 domains above based on their authority (highest to lowest).\n"
        f"All the domains should be included and listed using identifiers, in descending order of authority.\n"
        f"The output format should be [] > [], e.g., [4] > [0].\n"
        f"Only respond with the ranking results, do not say any word or explain.\n"
        f"Your answer: "
    )

    return prompt


def extract_ranking_answer(response_raw: str, num_domains: int = 10):
    response = response_raw.strip()

    bracket_matches = re.findall(r'\[(\d+)\]', response)
    if bracket_matches:
        try:
            ranking = [int(x) for x in bracket_matches]
            if (len(ranking) == num_domains and
                all(0 <= idx < num_domains for idx in ranking) and
                set(ranking) == set(range(num_domains))):
                return ranking
        except ValueError:
            pass

    all_numbers = re.findall(r'\b\d+\b', response)
    if len(all_numbers) >= num_domains:
        try:
            ranking = [int(x) for x in all_numbers[:num_domains]]
            if (len(ranking) == num_domains and
                all(0 <= idx < num_domains for idx in ranking) and
                set(ranking) == set(range(num_domains))):
                return ranking
        except ValueError:
            pass

    lines = response.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line:
            line_numbers = re.findall(r'\b\d+\b', line)
            if len(line_numbers) >= num_domains:
                try:
                    ranking = [int(x) for x in line_numbers[:num_domains]]
                    if (len(ranking) == num_domains and
                        all(0 <= idx < num_domains for idx in ranking) and
                        set(ranking) == set(range(num_domains))):
                        return ranking
                except ValueError:
                    pass

    random_ranking = list(range(num_domains))
    random.shuffle(random_ranking)
    return random_ranking


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

    results = []
    for i in range(0, len(listwise_items), BATCH_SIZE):
        batch = listwise_items[i:i + BATCH_SIZE]

        prompts = []
        for item in batch:
            prompt = build_ranking_prompt(item["domains"], item["doc_texts"])
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
            pred = extract_ranking_answer(response_raw, len(item["domains"]))

            results.append({
                "domains": item["domains"],
                "pageranks": item["pageranks"],
                "ground_truth": item["ground_truth"],
                "pred": pred,
                "llm_output": response_raw
            })

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
