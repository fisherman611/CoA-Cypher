import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import math

RETURN_PATTERN_MAPPING = {
    "n_name": "n_name",
    "n_prop": "n_prop_combined",
    "n_name_prop": "n_prop_combined",
    "n_prop_distinct": "n_prop_combined",
    "n_prop_array_distinct": "n_prop_combined",
    "n_order_by": "n_order_by",
    "n_argmax": "n_argmax",
    "n_where": "n_where",
    "n_agg": "n_agg",
    "n_group_by": "n_group_by"
}

AVAILABLE_METRICS = ["execution_accuracy", "psjs", "executable"]


def avg_and_round(nums: list[float], n: int = 4):
    return round(sum(nums) / len(nums), n) if nums else math.nan


def aggregate(results: list[tuple[str, float]]):
    res = {}
    for key, value in results:
        if key not in res:
            res[key] = []
        res[key].append(value)
    for key, values in res.items():
        res[key] = avg_and_round(values)
    return res

def calculate_result(result, metrics=['execution_accuracy', 'psjs', 'executable']):
    aggregated = {
        'overall': {
            m: avg_and_round(
                [(item["metrics"][m] if not isinstance(item["metrics"][m], dict) else 0.0)
                 for item in result if "metrics" in item and m in item["metrics"]]
            )
            for m in metrics
        }
    }
    return aggregated


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate metric scores from calculated result JSON."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to calculated result JSON (e.g. test_result.json).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["execution_accuracy", "psjs", "executable"],
        choices=AVAILABLE_METRICS,
        help="Metrics to aggregate.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save aggregated JSON. If omitted, only print.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        result = json.load(f)

    metric_scores = calculate_result(result, metrics=args.metrics)
    print(metric_scores)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(metric_scores, f, ensure_ascii=False, indent=2)
        print(f"Saved aggregated scores to {output_path}")


if __name__ == "__main__":
    main()
