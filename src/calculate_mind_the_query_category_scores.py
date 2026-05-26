import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


CATEGORY_NAMES = {
    "SR": "Simple_Retrieval",
    "SA": "Simple_Aggregation",
    "CR": "Complex_Retrieval",
    "CA": "Complex_Aggregation",
    "EQ": "Evaluation_query",
}

AVAILABLE_METRICS = ["execution_accuracy", "psjs", "executable"]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_value(item: dict, metric: str) -> float:
    value = item.get("metrics", {}).get(metric, 0.0)
    if isinstance(value, dict):
        return 0.0
    if value is None:
        return 0.0
    return float(value)


def avg_and_round(values: list[float], digits: int) -> float:
    return round(sum(values) / len(values), digits) if values else math.nan


def build_category_lookup(benchmark_items: list[dict]) -> dict[tuple[str, str], str]:
    lookup = {}
    for item in benchmark_items:
        if "graph" not in item or "qid" not in item or "categories" not in item:
            continue
        lookup[(str(item["graph"]), str(item["qid"]))] = item["categories"]
    return lookup


def calculate_category_scores(
    results: list[dict],
    category_lookup: dict[tuple[str, str], str],
    metrics: list[str],
    digits: int,
    include_counts: bool,
    expand_names: bool,
):
    buckets = defaultdict(lambda: defaultdict(list))
    missing_category = 0

    for item in results:
        category = item.get("categories")
        if category is None:
            key = (str(item.get("graph", "")), str(item.get("qid", "")))
            category = category_lookup.get(key)

        if category is None:
            missing_category += 1
            continue

        category_key = CATEGORY_NAMES.get(category, category) if expand_names else category
        for metric in metrics:
            buckets[category_key][metric].append(metric_value(item, metric))

    summary = {}
    for category in sorted(buckets):
        metric_scores = {
            metric: avg_and_round(values, digits)
            for metric, values in buckets[category].items()
        }
        if include_counts:
            metric_scores["count"] = len(next(iter(buckets[category].values()), []))
        summary[category] = metric_scores

    return summary, missing_category


def calculate_overall(results: list[dict], metrics: list[str], digits: int):
    return {
        metric: avg_and_round([metric_value(item, metric) for item in results], digits)
        for metric in metrics
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate Mind_the_query scores by category from test_result.json."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to calculated test_result.json.",
    )
    parser.add_argument(
        "--benchmark",
        default="benchmarks/Mind_the_query/test.json",
        help="Path to Mind_the_query benchmark test.json containing categories.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=AVAILABLE_METRICS,
        choices=AVAILABLE_METRICS,
        help="Metrics to aggregate.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save summary JSON. If omitted, only print.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=4,
        help="Number of decimal places for rounded averages.",
    )
    parser.add_argument(
        "--no_counts",
        action="store_true",
        help="Do not include sample counts per category.",
    )
    parser.add_argument(
        "--expand_names",
        action="store_true",
        help="Use full category names instead of SR/SA/CR/CA/EQ.",
    )
    parser.add_argument(
        "--overall",
        action="store_true",
        help="Also include overall metric averages.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    results = load_json(Path(args.input))
    benchmark = load_json(Path(args.benchmark))
    category_lookup = build_category_lookup(benchmark)

    by_category, missing_category = calculate_category_scores(
        results=results,
        category_lookup=category_lookup,
        metrics=args.metrics,
        digits=args.digits,
        include_counts=not args.no_counts,
        expand_names=args.expand_names,
    )

    output = {"by_category": by_category}
    if args.overall:
        output["overall"] = calculate_overall(results, args.metrics, args.digits)
    if missing_category:
        output["missing_category_count"] = missing_category

    print(json.dumps(output, ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"Saved category scores to {output_path}")


if __name__ == "__main__":
    main()
