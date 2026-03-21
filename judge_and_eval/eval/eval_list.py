import json
import numpy as np
import argparse
from scipy.stats import spearmanr, kendalltau


def ranking_to_ranks(ranking):
    n = len(ranking)
    ranks = np.empty(n, dtype=int)
    for rank, doc_idx in enumerate(ranking):
        ranks[doc_idx] = rank
    return ranks


def calculate_metrics(file_path):
    spearman_scores = []
    kendall_scores = []
    valid_count = 0

    skip_json_error = 0
    skip_wrong_type = 0
    skip_wrong_length = 0
    skip_not_permutation = 0
    skip_other = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        total_lines = 0
        for line in f:
            total_lines += 1
            if not line.strip():
                continue

            try:
                data = json.loads(line)
                pred_ranking = data.get('pred')
                ground_truth_ranking = data.get('ground_truth')

                if not isinstance(pred_ranking, list) or not isinstance(ground_truth_ranking, list):
                    skip_wrong_type += 1
                    continue

                n = len(ground_truth_ranking)

                if len(pred_ranking) != n or n < 2:
                    skip_wrong_length += 1
                    continue

                pred_set = set(pred_ranking)
                gt_set = set(ground_truth_ranking)
                expected_set = set(range(n))

                if pred_set != expected_set:
                    skip_not_permutation += 1
                    continue

                if gt_set != expected_set:
                    skip_not_permutation += 1
                    continue

                gt_ranks = ranking_to_ranks(ground_truth_ranking)
                pred_ranks = ranking_to_ranks(pred_ranking)

                s_corr, _ = spearmanr(gt_ranks, pred_ranks)
                k_corr, _ = kendalltau(gt_ranks, pred_ranks)

                spearman_scores.append(s_corr if not np.isnan(s_corr) else 0.0)
                kendall_scores.append(k_corr if not np.isnan(k_corr) else 0.0)

                valid_count += 1

            except json.JSONDecodeError:
                skip_json_error += 1
                continue
            except (KeyError, TypeError, ValueError):
                skip_other += 1
                continue

    print(f"Evaluation File: {file_path}")

    if valid_count == 0:
        print(f"Error: No valid evaluation data found in {file_path}")

    total_skipped = (skip_json_error + skip_wrong_type +
                     skip_wrong_length + skip_not_permutation + skip_other)
    total_processed = total_lines - skip_json_error

    print(f"\n[Sample Statistics]")
    print(f"  Total lines:    {total_lines}")
    print(f"  Valid samples:  {valid_count}")
    print(f"  Skipped samples: {total_skipped}")
    print(f"\n[Skip Details]")
    print(f"  JSON parse error:       {skip_json_error}")
    print(f"  Wrong field type:       {skip_wrong_type}")
    print(f"  List length mismatch:   {skip_wrong_length}")
    print(f"  Not a permutation:      {skip_not_permutation}")
    print(f"  Other errors:           {skip_other}")

    if valid_count > 0:
        avg_spearman = np.mean(spearman_scores)
        avg_kendall = np.mean(kendall_scores)

        print(f"  Spearman's Rho: {avg_spearman:.4f}")
        print(f"  Kendall's Tau:  {avg_kendall:.4f}")


    return {
        'total_lines': total_lines,
        'valid_count': valid_count,
        'skipped': {
            'json_error': skip_json_error,
            'wrong_type': skip_wrong_type,
            'wrong_length': skip_wrong_length,
            'not_permutation': skip_not_permutation,
            'other': skip_other,
            'total': total_skipped
        },
        'spearman': avg_spearman if valid_count > 0 else None,
        'kendall': avg_kendall if valid_count > 0 else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate ranking metrics (Spearman / Kendall)"
    )
    parser.add_argument("file_path", type=str, help="Path to JSONL file with evaluation results")
    args = parser.parse_args()

    calculate_metrics(args.file_path)
