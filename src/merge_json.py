import argparse
import json
from pathlib import Path

BENCHMARK_GRAPHS = {
    "Cypherbench": [
        "nba",
        "flight_accident",
        "fictional_character",
        "company",
        "geography",
        "movie",
        "politics",
    ],
    "Mind_the_query": ["bloom50", "healthcare", "wwc"],
    "Neo4j_Text2Cypher": [
        "bluesky",
        "buzzoverflow",
        "companies",
        "neoflix",
        "fincen",
        "gameofthrones",
        "grandstack",
        "movies",
        "network",
        "northwind",
        "offshoreleaks",
        "recommendations",
        "stackoverflow2",
        "twitch",
        "twitter",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge benchmark graph result JSON files into one JSON file."
    )
    parser.add_argument(
        "--benchmark",
        default="Cypherbench",
        choices=list(BENCHMARK_GRAPHS.keys()),
        help="Benchmark name used to determine default graph names.",
    )
    parser.add_argument(
        "--result_dir",
        default="results/Cypherbench/gemma/cot_gemma-4-31B-it_temperature0.2_top_p0.6",
        help="Directory containing <graph>_cyphers_result.json files.",
    )
    parser.add_argument(
        "--graphs",
        nargs="+",
        default=None,
        help="Optional graph names override. If omitted, use benchmark defaults.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file path. Default: <result_dir>/test_result.json",
    )
    parser.add_argument(
        "--skip_missing",
        action="store_true",
        help="Skip missing graph files instead of failing.",
    )
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    if args.graphs:
        graph_names = args.graphs
    else:
        default_graphs = BENCHMARK_GRAPHS[args.benchmark]
        existing_files = {p.name for p in result_dir.glob("*_cyphers_result.json")}
        graph_names = [g for g in default_graphs if f"{g}_cyphers_result.json" in existing_files]

        if not graph_names:
            raise FileNotFoundError(
                f"No '*_cyphers_result.json' files found in: {result_dir}"
            )

        missing = [g for g in default_graphs if g not in graph_names]
        if missing:
            print(
                "Missing benchmark graphs in directory (auto-skip): "
                + ", ".join(missing)
            )

    output_path = Path(args.output) if args.output else result_dir / "test_result.json"
    merged = []
    loaded_files = 0

    for graph in graph_names:
        path = result_dir / f"{graph}_cyphers_result.json"
        if not path.exists():
            if args.skip_missing:
                print(f"Skip missing file: {path}")
                continue
            raise FileNotFoundError(f"Missing file: {path}")

        data = load_json(path)
        if isinstance(data, list):
            merged.extend(data)
        else:
            merged.append(data)
        loaded_files += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Merged {loaded_files} files into {output_path}")


if __name__ == "__main__":
    main()
