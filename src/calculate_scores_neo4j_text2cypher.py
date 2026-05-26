import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqdm.auto import tqdm


NEO4J_TEXT2CYPHER_GRAPHS = [
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
]


def safe_compute(metric_fn, pred_cypher, target_cypher, conn, metric_name, **metric_kwargs):
    try:
        return metric_fn(pred_cypher, target_cypher, conn, **metric_kwargs)
    except Exception as e:
        return {"error": f"{metric_name} failed: {str(e)}"}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def extract_question(item):
    return item.get("question", item.get("nl_question", ""))


def result_key(item):
    if item.get("qid") is not None:
        return ("qid", item["qid"])
    return ("qgp", item.get("question", ""), item.get("gold", ""), item.get("pred", ""))


def sample_key(item):
    if item.get("qid") is not None:
        return ("qid", item["qid"])
    question = extract_question(item)
    gold = item.get("gold", item.get("gold_cypher", ""))
    pred = item.get("pred", item.get("pred_cypher", ""))
    return ("qgp", question, gold, pred)


def get_samples_for_graph(data, graph):
    if not data:
        return []
    if any("graph" in item for item in data):
        return [item for item in data if item.get("graph") == graph]
    return data


def build_connector(graph, args):
    from neo4j_connector import Neo4jConnector

    username = args.username if args.username is not None else graph
    password = args.password if args.password is not None else graph
    database = args.database if args.database else graph
    return Neo4jConnector(
        name=args.name,
        host=args.host,
        port=args.port,
        username=username,
        password=password,
        database=database,
    )


def evaluate_graph(graph, samples, args, output_path=None, initial_results=None):
    from metrics.executable import executable
    from metrics.execution_accuracy import execution_accuracy
    from metrics.provenance_subgraph_jaccard_similarity import (
        provenance_subgraph_jaccard_similarity,
    )

    conn = build_connector(graph, args)
    try:
        if args.limit is not None:
            samples = samples[: args.limit]

        output_results = list(initial_results or [])
        done_keys = {result_key(item) for item in output_results}
        pending_samples = [item for item in samples if sample_key(item) not in done_keys]
        processed_count = 0

        for item in tqdm(pending_samples, desc=f"Evaluating {graph}"):
            processed_count += 1
            question = extract_question(item)
            pred_cypher = item.get("pred", item.get("pred_cypher", ""))
            target_cypher = item.get("gold", item.get("gold_cypher", ""))

            exec_score = safe_compute(
                executable,
                pred_cypher,
                target_cypher,
                conn,
                "executable",
                timeout=args.query_timeout,
            )
            ex_score = safe_compute(
                execution_accuracy,
                pred_cypher,
                target_cypher,
                conn,
                "execution_accuracy",
                timeout=args.query_timeout,
                target_timeout=args.target_query_timeout,
            )
            psjs_score = safe_compute(
                provenance_subgraph_jaccard_similarity,
                pred_cypher,
                target_cypher,
                conn,
                "provenance_subgraph_jaccard_similarity",
                timeout=args.query_timeout,
                target_timeout=args.target_query_timeout,
            )

            output_results.append(
                {
                    "qid": item.get("qid", ""),
                    "graph": graph,
                    "question": question,
                    "gold": target_cypher,
                    "pred": pred_cypher,
                    "metrics": {
                        "executable": exec_score,
                        "execution_accuracy": ex_score,
                        "psjs": psjs_score,
                    },
                }
            )

            if output_path and args.flush_every > 0 and processed_count % args.flush_every == 0:
                write_json(output_path, output_results)
                print(
                    f"Checkpoint saved to {output_path} ({processed_count}/{len(pending_samples)} new samples)"
                )

        if output_path:
            write_json(output_path, output_results)

        return output_results
    finally:
        conn.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate Neo4j Text2Cypher metrics from per-graph result files."
    )
    parser.add_argument(
        "--result_dir",
        default=None,
        help="Directory containing <graph>_cyphers_result.json files for one method.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Backward-compatible single merged JSON input file.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Default: <result_dir>/calculated_scores",
    )
    parser.add_argument(
        "--graphs",
        nargs="+",
        choices=NEO4J_TEXT2CYPHER_GRAPHS,
        default=None,
        help="Optional graph list override.",
    )
    parser.add_argument(
        "--subset",
        choices=NEO4J_TEXT2CYPHER_GRAPHS,
        default=None,
        help="Backward-compatible single graph override (same as --graphs <graph>).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit number of samples to evaluate for each graph.",
    )
    parser.add_argument(
        "--flush_every",
        type=int,
        default=10,
        help="Write checkpoint every N newly processed samples.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume scoring from existing output files.",
    )
    parser.add_argument(
        "--host", default="bolt+s://demo.neo4jlabs.com", help="Neo4j host"
    )
    parser.add_argument("--port", type=int, default=7687, help="Neo4j port")
    parser.add_argument(
        "--username",
        default=None,
        help="Neo4j username. If omitted, use graph name.",
    )
    parser.add_argument(
        "--password",
        default=None,
        help="Neo4j password. If omitted, use graph name.",
    )
    parser.add_argument(
        "--database",
        default=None,
        help="Optional database override for all graphs. Default: graph name.",
    )
    parser.add_argument(
        "--name", default="neo4j_text2cypher_db", help="Connector name"
    )
    parser.add_argument(
        "--query_timeout",
        type=int,
        default=120,
        help="Timeout (seconds) for predicted query execution.",
    )
    parser.add_argument(
        "--target_query_timeout",
        type=int,
        default=120,
        help="Timeout (seconds) for gold/target query execution.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.result_dir and not args.input:
        raise ValueError("Provide either --result_dir or --input")

    if args.result_dir:
        result_dir = Path(args.result_dir)
        if not result_dir.exists():
            raise FileNotFoundError(f"Result directory not found: {result_dir}")
        output_dir = (
            Path(args.output_dir) if args.output_dir else result_dir / "calculated_scores"
        )
    else:
        output_dir = Path(args.output_dir) if args.output_dir else Path("calculated_scores")

    os.makedirs(output_dir, exist_ok=True)

    if args.subset:
        target_graphs = [args.subset]
    else:
        target_graphs = args.graphs if args.graphs else NEO4J_TEXT2CYPHER_GRAPHS

    total_saved = 0

    if args.result_dir:
        result_dir = Path(args.result_dir)
        for graph in target_graphs:
            graph_file = result_dir / f"{graph}_cyphers_result.json"
            if not graph_file.exists():
                continue

            data = load_json(graph_file)
            samples = get_samples_for_graph(data, graph)
            if not samples:
                continue

            output_path = output_dir / f"{graph}_cyphers_result.json"
            existing_results = []
            if args.resume and output_path.exists():
                existing_results = load_json(output_path)
                if not isinstance(existing_results, list):
                    existing_results = []

            scored = evaluate_graph(
                graph,
                samples,
                args,
                output_path=output_path,
                initial_results=existing_results,
            )
            print(f"Saved {len(scored)} results to {output_path}")
            total_saved += 1

    if args.input:
        merged_data = load_json(Path(args.input))
        for graph in target_graphs:
            samples = get_samples_for_graph(merged_data, graph)
            if not samples:
                continue

            output_path = output_dir / f"{graph}_cyphers_result.json"
            existing_results = []
            if args.resume and output_path.exists():
                existing_results = load_json(output_path)
                if not isinstance(existing_results, list):
                    existing_results = []

            scored = evaluate_graph(
                graph,
                samples,
                args,
                output_path=output_path,
                initial_results=existing_results,
            )
            print(f"Saved {len(scored)} results to {output_path}")
            total_saved += 1

    if total_saved == 0:
        raise FileNotFoundError(
            "No graph result files found/evaluated. Check --result_dir or --input and --graphs."
        )


if __name__ == "__main__":
    main()
