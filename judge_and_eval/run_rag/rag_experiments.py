import json
import os
import re
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


MODEL_DIR = "YOUR_MODEL_DIR"
MODEL_NAME = "YOUR_MODEL_NAME"
TOP_K = 5


# Prompt Templates
AUTHORITY_PROMPT_TEMPLATE = (
    "You are a web analysis expert. Please score the **SOURCE AUTHORITY** of each source url "
    "in the provided list based on the following criteria, with a rating range of 0 to 9 where 0 is the lowest and 9 is highest.\n\n"
    "Source authority refers to the credibility, professionalism, official certification, or public recognition of the domain itself and its content.\n"
    "Consider all URLs together when assigning scores to ensure consistency and relative calibration.\n\n"
    "**SOURCES**:\n"
    "{urls_list}"
    "\nYour output MUST be a single, valid JSON object. The JSON object must map each domain's numerical identifier (as a string) to its integer score (0-9).\n"
    'Example format: {{"0": 7, "1": 4, ..., "9": 8}}. Do not add any explanation.\n'
    "Your answer: "
)

RELEVANCE_PROMPT_TEMPLATE = (
    "You are a web analysis expert. Please score the **RELEVANCE** of each document "
    "to the question with a rating range of 0 to 9 where 0 is the lowest and 9 is highest.\n"
    "Consider all documents together when assigning scores to ensure consistency and relative calibration.\n\n"
    "**Question**: {question}\n\n"
    "DOCUMENTS:\n"
    "{documents_list}"
    "\nYour output MUST be a single, valid JSON object. The JSON object must map each "
    "document's numerical identifier (as a string) to its integer score (0-9).\n"
    'Example format: {{"0": 7, "1": 4, ..., "9": 8}}. Do not add any explanation.\n'
    "Your answer: "
)

UTILITY_PROMPT_TEMPLATE = (
    "You are a web analysis expert. Please score the **UTILITY** of each document "
    "in answering the question based on the following criteria, with a rating range of 0 to 9 where 0 is the lowest and 9 is highest.\n"
    "The passage has utility in answering the question, meaning that the passage not only is relevant to the question, "
    "but also is useful in generating a correct, reasonable, and perfect answer to the question.\n"
    "Consider all documents together when assigning scores to ensure consistency and relative calibration.\n\n"
    "**Question**: {question}\n\n"
    "DOCUMENTS:\n"
    "{documents_list}\n"
    "Step 1: Generate a pseudo-answer to the question based on all the documents.\n"
    "Step 2: Based on the pseudo-answers, rate each document's utility in answering the question on a scale of 0 to 9.\n"
    "Your output MUST be a single, valid JSON object with two keys:\n"
    "\t- \"answers\": maps each document's numerical identifier (as a string) to its pseudo-answer\n"
    "\t- \"scores\": maps each document's numerical identifier (as a string) to its integer score (0-9)\n"
    'Example format: {{"answers": {{"0": "...", "1": "...", ...}}, "scores": {{"0": 7, "1": 4, ..., "9": 8}}}}. Do not add any explanation.\n'
    "Your answer: "
)

ANSWER_PROMPT_TEMPLATE = (
    "You are given a question and several pieces of documents. "
    "Your task is to analyze the documents and provide a concise answer to the question based solely on the provided reference documents.\n"
    "**Question**: {question}\n\n"
    "**Reference Documents**:\n{documents_context}\n"
    "Please answer with ONLY 'yes' or 'no' based on the documents above. Do not add any explanation.\n"
    "Your answer: "
)


def load_test_data(file_path):
    """Load test data from JSONL file."""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def init_model():
    """Initialize LLM and tokenizer with hardcoded model."""
    llm = LLM(
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
        max_num_seqs=128,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    return llm, tokenizer


def call_llm(llm, tokenizer, sampling_params, prompt, model_name):
    messages = [{"role": "user", "content": prompt}]
    if "qwen3" in model_name.lower():
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    else:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    outputs = llm.generate([formatted], sampling_params)
    return outputs[0].outputs[0].text.strip() if outputs and outputs[0].outputs else ""


def parse_json_response(response_text):
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    try:
        match = re.search(r'```json\s*(\{.*?\})\s*```|(\{.*\})', response_text, re.DOTALL)
        if match:
            json_str = match.group(1) if match.group(1) else match.group(2)
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    return None


def build_authority_listwise_prompt(urls):
    """Build authority scoring prompt with URLs."""
    urls_list = "\n".join([f"[{i}] {url}" for i, url in enumerate(urls)])
    return AUTHORITY_PROMPT_TEMPLATE.format(urls_list=urls_list)


def build_relevance_listwise_prompt(question, documents):
    """Build relevance scoring prompt with documents."""
    documents_list = "\n".join([
        f"[{i}] {doc.get('url', '')}: {doc['doc_text']}"
        for i, doc in enumerate(documents)
    ])
    return RELEVANCE_PROMPT_TEMPLATE.format(question=question, documents_list=documents_list)


def build_utility_listwise_prompt(question, documents):
    """Build utility scoring prompt with documents."""
    documents_list = "\n".join([
        f"[{i}] {doc.get('url', '')}: {doc['doc_text']}"
        for i, doc in enumerate(documents)
    ])
    return UTILITY_PROMPT_TEMPLATE.format(question=question, documents_list=documents_list)


def run_scoring(sample, llm, tokenizer):
    question = sample["question"]
    documents = sample["documents"]
    n_docs = len(documents)
    scoring_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=2048)

    result = {
        "qid": sample["qid"],
        "question": question,
        "ground_truth": sample["ground_truth"],
        "n_docs": n_docs,
    }

    # Authority scoring (URL only)
    print("Authority scoring (URL only)...")
    urls = [doc["url"] for doc in documents]
    authority_prompt = build_authority_listwise_prompt(urls)
    authority_response = call_llm(llm, tokenizer, scoring_params, authority_prompt)
    authority_scores_json = parse_json_response(authority_response)
    authority_scores = [int(authority_scores_json.get(str(i), 0)) for i in range(n_docs)] if authority_scores_json else [0] * n_docs
    result["authority_scores"] = {"response": authority_response, "scores": authority_scores}
    print(f"    Authority scores: {authority_scores}")

    # Relevance scoring (URL + text)
    print("Relevance scoring (URL + text)...")
    relevance_prompt = build_relevance_listwise_prompt(question, documents)
    relevance_response = call_llm(llm, tokenizer, scoring_params, relevance_prompt)
    relevance_scores_json = parse_json_response(relevance_response)
    relevance_scores = [int(relevance_scores_json.get(str(i), 0)) for i in range(n_docs)] if relevance_scores_json else [0] * n_docs
    result["relevance_scores"] = {"response": relevance_response, "scores": relevance_scores}
    print(f"    Relevance scores: {relevance_scores}")

    # Utility scoring (URL + text)
    print("Utility scoring (URL + text)...")
    utility_prompt = build_utility_listwise_prompt(question, documents)
    utility_response = call_llm(llm, tokenizer, scoring_params, utility_prompt)
    utility_result_json = parse_json_response(utility_response)
    if utility_result_json and "scores" in utility_result_json and "answers" in utility_result_json:
        utility_scores = [int(utility_result_json["scores"].get(str(i), 0)) for i in range(n_docs)]
        pseudo_answers = [utility_result_json["answers"].get(str(i), "") for i in range(n_docs)]
    else:
        utility_scores = [0] * n_docs
        pseudo_answers = [""] * n_docs
    result["utility_scores"] = {"response": utility_response, "scores": utility_scores, "pseudo_answers": pseudo_answers}
    print(f"    Utility scores: {utility_scores}")

    # Attach scores to each document
    result["documents"] = []
    for i, doc in enumerate(documents):
        doc_info = doc.copy()
        doc_info["scores"] = {
            "authority": authority_scores[i],
            "relevance": relevance_scores[i],
            "utility": utility_scores[i]
        }
        result["documents"].append(doc_info)

    return result


def generate_final_answer(question, top_k_docs, llm, tokenizer):
    answer_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=10)
    documents_context = ""
    for i, doc in enumerate(top_k_docs):
        documents_context += f"[{i}] Source URL: {doc['url']}\nContent: {doc['doc_text']}\n\n"
    prompt = ANSWER_PROMPT_TEMPLATE.format(question=question, documents_context=documents_context)
    answer = call_llm(llm, tokenizer, answer_params, prompt, model_name)
    # Clean and normalize answer
    cleaned_answer = re.sub(r'[^a-zA-Z]', '', answer.lower())
    return cleaned_answer


def analyze_results(output_file):
    print(f"\n{'='*25} Accuracy Analysis {'='*25}")
    results = []
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))

    if not results:
        print("No results to analyze.")
        return

    accuracies = {}
    for res in results:
        gt = res['ground_truth'].lower()
        for strategy, answer_data in res['final_answers'].items():
            if strategy not in accuracies:
                accuracies[strategy] = {'correct': 0, 'total': 0}

            pred = answer_data['answer']
            if pred == gt:
                accuracies[strategy]['correct'] += 1
            accuracies[strategy]['total'] += 1

    print(f"\n{'Strategy':<15} {'Correct':<10} {'Total':<10} {'Accuracy (%)':<15}")
    print("-" * 55)
    for strategy, counts in accuracies.items():
        acc = (counts['correct'] / counts['total']) * 100 if counts['total'] > 0 else 0
        print(f"{strategy:<15} {counts['correct']:<10} {counts['total']:<10} {acc:>6.2f}%")
    print("-" * 55)


def run_experiment(input_file, output_file):
    test_data = load_test_data(input_file)
    print(f"Loaded {len(test_data)} samples")

    llm, tokenizer = init_model()

    processed_qids = set()
    if os.path.exists(output_file):
        print(f"\nOutput file detected, reading processed samples...")
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        processed_qids.add(json.loads(line)["qid"])
                    except (json.JSONDecodeError, KeyError):
                        continue
        print(f"Already processed: {len(processed_qids)} samples.")

    remaining_samples = [s for s in test_data if s["qid"] not in processed_qids]
    print(f"Remaining samples to process: {len(remaining_samples)}")

    if not remaining_samples:
        print("All samples have been processed!")
        analyze_results(output_file)
        return

    filter_strategies = {
        "authority": lambda docs: sorted(docs, key=lambda x: x["scores"]["authority"], reverse=True),
        "relevance": lambda docs: sorted(docs, key=lambda x: x["scores"]["relevance"], reverse=True),
        "utility":   lambda docs: sorted(docs, key=lambda x: x["scores"]["utility"], reverse=True),
    }

    with open(output_file, "a", encoding="utf-8") as f_out:
        for i, sample in enumerate(remaining_samples, 1):
            print(f"\n{'='*60}")
            print(f"[{i}/{len(remaining_samples)}] qid={sample['qid']}  gt={sample['ground_truth']}")
            print(f"Question: {sample['question'][:80]}...")
            print(f"{'='*60}")

            try:
                scoring_result = run_scoring(sample, llm, tokenizer)
                docs_with_scores = scoring_result["documents"]
                final_answers = {}

                for name, sort_fn in filter_strategies.items():
                    ranked_docs = sort_fn(docs_with_scores)
                    top_k_docs = ranked_docs[:TOP_K]
                    print(f"  -> Generating answer for strategy: '{name}' (top-{TOP_K})")
                    answer = generate_final_answer(sample["question"], top_k_docs, llm, tokenizer)
                    final_answers[name] = {
                        "answer": answer,
                        "ranked_doc_ids": [d["docid"] for d in ranked_docs],
                        "top_k_doc_ids": [d["docid"] for d in top_k_docs]
                    }

                scoring_result["final_answers"] = final_answers
                f_out.write(json.dumps(scoring_result, ensure_ascii=False) + "\n")
                f_out.flush()
                print(f"Sample {sample['qid']} saved.")

            except Exception as e:
                print(f"Sample {sample['qid']} failed: {e}")
                continue

    print(f"\n{'='*60}")
    print(f"Experiment finished. Results saved to: {output_file}")
    print(f"{'='*60}")
    analyze_results(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG Filtering Experiment.")
    parser.add_argument("--input_file", type=str, default="rag/authrag_science_1_60.jsonl",
                        help="Path to the input JSONL data file.")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to the output JSONL results file.")

    args = parser.parse_args()

    try:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        run_experiment(args.input_file, args.output_file)
    except Exception:
        import traceback
        traceback.print_exc()
