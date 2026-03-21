import json
import re

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


MODEL_DIR = "YOUR_MODEL_DIR"
EVAL_FILE = "EVAL_FILE"
OUTPUT_FILE = "OUTPUT_FILE"
BATCH_SIZE = 16


def build_ranking_prompt(question_entity):
    return (
        f"You are a Wikipedia notability expert and a music industry expert that can rank entities based on their authority.\n\n"
        f"I will provide you with 10 song entities, each indicated by a numerical identifier [].\n"
        f"Rank the song entities based on their authority (the public recognition, fame, historical significance, or widespread knowledge of the entity).\n"
        f"Context: These entities are the subjects of questions like 'Who is the performer of the song ...'\n\n"
        + "\n".join([f"[{i}] \"{question_entity[i]}\"" for i in range(10)])
        + f"\n\nRank the 10 song entities above based on their authority (highest to lowest).\n"
        f"All the song entities should be included and listed using identifiers, in descending order of authority.\n"
        f"The output format should be [] > [], e.g., [4] > [0].\n"
        f"Only respond with the ranking results, do not say any word or explain.\n"
        f"Your answer: "
    )


def extract_ranking_answer(response, num_entities=10):
    text = response.strip()
    if match := re.search(r"Your answer:\s*(.*)", text, re.IGNORECASE | re.DOTALL):
        text = match.group(1).strip()
    for extractor in [re.compile(r"\[(\d+)\]"), re.compile(r"\b\d+\b")]:
        if nums := [int(x) for x in extractor.findall(text)]:
            if len(nums) >= num_entities:
                ranking = nums[:num_entities]
                if len(ranking) == num_entities and set(ranking) == set(range(num_entities)):
                    return ranking
    return -1


def main():
    llm = LLM(model=MODEL_DIR, tokenizer=MODEL_DIR, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=256)

    __import__("os").makedirs(__import__("os").path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for i in tqdm(range(0, len(items), BATCH_SIZE)):
            batch = items[i:i + BATCH_SIZE]
            prompts = [tokenizer.apply_chat_template([{"role": "user", "content": build_ranking_prompt(item["question_entity"])}],
                                                     tokenize=False, add_generation_prompt=True, enable_thinking=False) for item in batch]
            for item, output in zip(batch, llm.generate(prompts, sampling_params)):
                text = output.outputs[0].text if output.outputs else ""
                n = len(item["question_entity"])
                pred = extract_ranking_answer(text, n)
                out_f.write(json.dumps({
                    "question_entity": item["question_entity"], "score": item["score"],
                    "ground_truth": item["ground_truth"], "pred": pred, "llm_output": text
                }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
