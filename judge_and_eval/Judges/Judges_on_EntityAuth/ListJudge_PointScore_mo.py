import json

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


MODEL_DIR = "YOUR_MODEL_DIR"
EVAL_FILE = "LISTWISE_FILE"
OUTPUT_FILE = "OUTPUT_FILE"
BATCH_SIZE = 16


def build_prompt(entities):
    return (
        f"You are a Wikipedia notability expert and a film expert. Please score the entity authority of each movie entity "
        f"in the provided list based on the following criteria, with a rating range of 0 to 9.\n\n"
        f"Entity authority refers to the public recognition, fame, historical significance, or widespread knowledge of the entity (person, place, organization, etc.).\n"
        f"Consider all entities together when assigning scores to ensure consistency and relative calibration.\n\n"
        f"**QUESTION CONTEXT**: These entities are the subjects of questions like 'Who is the director of the movie ...'\n\n"
        f"**TARGET ENTITIES**:\n"
        + "\n".join([f'[{i}] "{entity}"' for i, entity in enumerate(entities)])
        + f"\n\nYour output MUST be a single, valid JSON object. The JSON object must map each entity's numerical identifier (as a string) to its integer score (0-9).\n"
        f'Example format: {{"0": 7, "1": 4, ..., "9": 8}}. Do not add any explanation.\n'
        f"Your answer: "
    )


def extract_scores(response, num_entities):
    text = response.strip()
    try:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            scores = {int(k.strip("[]")): float(v) for k, v in json.loads(text[start:end + 1]).items()}
            if len(scores) == num_entities:
                return sorted(range(num_entities), key=lambda x: scores.get(x, 0), reverse=True), scores
    except Exception:
        pass
    return list(range(num_entities)), None


def main():
    llm = LLM(model=MODEL_DIR, tokenizer=MODEL_DIR, max_num_seqs=256, trust_remote_code=True, tensor_parallel_size=4, dtype="bfloat16")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=256)

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
        for i in tqdm(range(0, len(pending), BATCH_SIZE)):
            batch = pending[i:i + BATCH_SIZE]
            prompts = [tokenizer.apply_chat_template([{"role": "user", "content": build_prompt(item["question_entity"])}],
                                                     tokenize=False, add_generation_prompt=True, enable_thinking=False) for _, item in batch]
            outputs = llm.generate(prompts, sampling_params)
            for (idx, item), output in zip(batch, outputs):
                text = output.outputs[0].text if output.outputs else ""
                n = len(item["question_entity"])
                pred, scores = extract_scores(text, n)
                out_f.write(json.dumps({
                    "item_id": idx, "question_entity": item["question_entity"], "score": item["score"],
                    "ground_truth": item["ground_truth"], "pred": pred, "scores": scores, "llm_output": text
                }, ensure_ascii=False) + "\n")
            out_f.flush()


if __name__ == "__main__":
    main()
