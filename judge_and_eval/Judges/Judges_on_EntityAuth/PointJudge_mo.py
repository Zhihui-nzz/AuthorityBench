import json
import re

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


MODEL_DIR = "YOUR_MODEL_DIR"
INPUT_FILE = "INPUT_FILE"
OUTPUT_FILE = "OUTPUT_FILE"
BATCH_SIZE = 8


def build_prompt(question, entity):
    return (
        f"You are a Wikipedia notability expert and a film expert. Please score the authority of the given movie entity based on the following criteria, with a rating range of 0 to 9.\n\n"
        f"Entity authority refers to the public recognition, fame, historical significance, or widespread knowledge of the entity (person, place, organization, etc.).\n"
        f"**QUESTION CONTEXT**: {question}\n"
        f"**TARGET ENTITY**: {entity}\n"
        f"Based on the question context and the above criteria, rate the authority of this ENTITY (0-9 points).\n"
        f"Output only: {{\"score\": X}} where X is your evaluated score (0, 1, 2, 3, 4, 5, 6, 7, 8, or 9). Do not add any explanation.\n"
        f"Your answer: "
    )


def extract_score(text):
    text = text.strip()
    if "Your answer:" in text:
        text = text.split("Your answer:")[-1].strip()
    text = re.sub(r'^(user|assistant|Your answer:|Answer:)\s*', '', text, flags=re.IGNORECASE).strip()
    try:
        if match := re.search(r'\{[^}]*"score"\s*:\s*(\d+)[^}]*\}', text, re.IGNORECASE):
            score = int(match.group(1))
            if 0 <= score <= 9:
                return score
    except Exception:
        pass
    if numbers := re.findall(r'\b([0-9])\b', text):
        return int(numbers[0])
    return -1


def main():
    llm = LLM(model=MODEL_DIR, tokenizer=MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=256)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        items = [json.loads(line.strip()) for line in f if line.strip()]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for i in tqdm(range(0, len(items), BATCH_SIZE)):
            batch = items[i:i + BATCH_SIZE]
            prompts = [tokenizer.apply_chat_template([{"role": "user", "content": build_prompt(item["question"], item["question_entity"])}],
                                                     tokenize=False, add_generation_prompt=True, enable_thinking=False) for item in batch]
            outputs = llm.generate(prompts, sampling_params)
            for item, output in zip(batch, outputs):
                text = output.outputs[0].text if output.outputs else ""
                pred = extract_score(text)
                out_f.write(json.dumps({
                    "question": item["question"], "question_entity": item["question_entity"],
                    "popularity": item["popularity"], "true_score": item["score"],
                    "pred_score": pred, "llm_output": text if pred != -1 else ""
                }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
