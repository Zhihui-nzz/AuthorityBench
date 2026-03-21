import json
import os
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL_DIR = "YOUR_MODEL_DIR"
INPUT_FILE = "EVAL_FILE"
OUTPUT_FILE = "OUTPUT_DIR"
BATCH_SIZE = 8


def build_prompt(domain: str, doc_text: str) -> str:
    truncated_text = doc_text[:500] if len(doc_text) > 500 else doc_text

    prompt = (
        f"You are a web analysis expert. Please score the source authority of the given domain based on the following criteria, with a rating range of 0 to 9.\n\n"
        f"Source authority refers to the credibility, professionalism, official certification, or public recognition of the domain itself.\n"
        f"**DOMAIN**: {domain}\n"
        f"**Webpage Content**: {truncated_text}\n"
        f"Based on the domain name, their web content and the above criteria, rate the source authority of this DOMAIN (0-9 points).\n"
        f"Output only: {{\"score\": X}} where X is your evaluated score (0, 1, 2, 3, 4, 5, 6, 7, 8, or 9). Do not add any explanation.\n"
        f"Your answer: "
    )
    return prompt


def extract_score(text: str) -> int:
    text = text.strip()

    # Try to extract JSON format: {"score": X}
    json_match = re.search(r'\{[^}]*"score"\s*:\s*(\d+)[^}]*\}', text)
    if json_match:
        score = int(json_match.group(1))
        if 0 <= score <= 9:
            return score

    numbers = re.findall(r'\b([0-9])\b', text)
    if numbers:
        return int(numbers[0])

    return -1


def main():
    pointwise_items = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pointwise_items.append(json.loads(line))

    llm = LLM(model=MODEL_DIR, tokenizer=MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=256
    )

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i in range(0, len(pointwise_items), BATCH_SIZE):
            batch = pointwise_items[i:i + BATCH_SIZE]

            raw_prompts = [
                build_prompt(item.get("domain", ""), item.get("doc_text", ""))
                for item in batch
            ]

            messages_batch = [[{"role": "user", "content": prompt}] for prompt in raw_prompts]
            prompts = [
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                for messages in messages_batch
            ]

            outputs = llm.generate(prompts, sampling_params)

            for item, output in zip(batch, outputs):
                response_raw = output.outputs[0].text if output.outputs else ""
                pred = extract_score(response_raw)

                result = {
                    "domain": item.get("domain", ""),
                    "url": item.get("url", ""),
                    "docid": item.get("docid", ""),
                    "true_score": item["pagerank"],
                    "pred_score": pred,
                    "llm_output": response_raw
                }

                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
