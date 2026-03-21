import json
import argparse


def calculate_accuracy(file_path):
    total = 0
    correct = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            try:
                data = json.loads(line.strip())
                label = data.get('label')
                pred = data.get('pred')

                if label is not None and pred is not None:
                    total += 1
                    if label == pred:
                        correct += 1

            except json.JSONDecodeError:
                continue

    accuracy = correct / total if total > 0 else 0.0

    return {
        'total': total,
        'correct': correct,
        'accuracy': accuracy
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate prediction accuracy from JSONL file"
    )
    parser.add_argument("file_path", type=str, help="Path to JSONL file")
    args = parser.parse_args()

    result = calculate_accuracy(args.file_path)

    print(f"  Accuracy:       {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")

    return result


if __name__ == '__main__':
    main()
