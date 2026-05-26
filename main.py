import os
import sys
from pathlib import Path
import argparse
import json
from src.utils import read_json_file
from src.neo4j_connector import Neo4jConnector
from dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv()

from src.llm_services import init_llm

def convert_to_serializable(obj):
    if hasattr(obj, '__dict__'):
        return {key: convert_to_serializable(value) for key, value in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj

parser = argparse.ArgumentParser(description="Arguments for running CoA-Cypher")
parser.add_argument("--benchmark", type=str, help="Benchmark name", default="Cypherbench", choices=["Cypherbench"])
parser.add_argument("--cb_graph", type=str, help="Name of graph of the Cypherbench", default="nba", choices=["nba", "flight_accident", "fictional_character", \
                                                                                                             "company", "geography", "movie", "politics"])
parser.add_argument("--method", type=str, help="Name of the method we use", default="coa_cypher", choices=["coa_cypher"])
parser.add_argument("--model", type=str, help="Name of the model we use", default="gpt-oss-120b")
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--top_p", type=float, default=0.6)
parser.add_argument("--max_tokens", type=int, default=8192)
parser.add_argument("--split", type=int, default=-1)
parser.add_argument("--qid", type=str, default=None, help="Run only the sample with this qid after graph filtering.")
parser.add_argument("--output_path", type=str, default=None, help="Write this run to a specific result JSON file.")
parser.add_argument("--max_workers", type=int, default=10)
parser.add_argument("--flush_every", type=int, default=20)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--retry_empty_pred", action="store_true")

parser.add_argument("--use_critic", type=bool, default=True)
parser.add_argument("--use_verifier", type=bool, default=True)
parser.add_argument("--max_rounds", type=int, default=5)
parser.add_argument("--constraint_threshold", type=int, default=2)
parser.add_argument("--hop_threshold", type=int, default=2)
parser.add_argument("--aggregation_trigger", type=bool, default=True)
args = parser.parse_args()

if not args.model.rsplit("/", 1)[-1].lower().startswith("gpt"):
    raise ValueError("Only GPT models are supported. Use a model name whose final slug starts with 'gpt'.")

test_data = read_json_file("benchmarks/Cypherbench/test.json")


def apply_split(data, split):
    return data if split == -1 else data[:split]


def load_existing_results(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def result_key(item):
    if item.get("status") == "failed":
        return None
    if item.get("qid") is not None:
        return ("qid", item["qid"])
    return ("qg", item.get("question", ""), item.get("gold", ""))


def sample_key(item):
    if item.get("qid") is not None:
        return ("qid", item["qid"])
    return ("qg", item.get("nl_question", ""), item.get("gold_cypher", ""))


def is_empty_pred(item):
    return str(item.get("pred", "") or "").strip() == ""


def result_match_key(item):
    if item.get("qid") is not None:
        return ("qid", item["qid"])
    return ("qg", item.get("question", ""), item.get("gold", ""))


def run_for_graph(graph_name, data_test, **run_kwargs):
    if args.qid is not None:
        data_test = [item for item in data_test if str(item.get("qid")) == args.qid]
        if not data_test:
            raise ValueError(f"No sample found for graph={graph_name}, qid={args.qid}")

    output_path = Path(args.output_path) if args.output_path else Path(output_dir) / f"{graph_name}_cyphers_result.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = apply_split(data_test, args.split)

    existing_results = []
    if (args.resume or args.retry_empty_pred) and output_path.exists():
        existing_results = load_existing_results(output_path)

        if args.retry_empty_pred:
            total_existing = len(existing_results)
            selected_keys = {sample_key(item) for item in selected}
            empty_keys = {
                key
                for item in existing_results
                if is_empty_pred(item)
                for key in [result_match_key(item)]
                if key and key in selected_keys
            }
            selected = [item for item in selected if sample_key(item) in empty_keys]
            existing_results = [
                item
                for item in existing_results
                if not (is_empty_pred(item) and result_match_key(item) in empty_keys)
            ]
            print(
                f"[retry_empty_pred] graph={graph_name}, total_existing={total_existing}, "
                f"retry_targets={len(selected)}, kept={len(existing_results)}"
            )
        elif args.resume:
            done_keys = {key for key in (result_key(item) for item in existing_results) if key}
            selected = [item for item in selected if sample_key(item) not in done_keys]
            print(
                f"[resume] graph={graph_name}, existing={len(existing_results)}, remaining={len(selected)}"
            )

    if not selected and existing_results:
        return existing_results

    return run_parallel_pipeline(
        test_data=selected,
        max_workers=args.max_workers,
        output_path=output_path,
        flush_every=args.flush_every,
        initial_results=existing_results,
        **run_kwargs,
    )


def get_model_group(model_name):
    model_slug = model_name.rsplit("/", 1)[-1]
    return model_slug.replace("_", "-").split("-", 1)[0].lower()


model_group = get_model_group(args.model)
output_model_name = args.model

output_dir = (
    f"results/Cypherbench/{model_group}/coa_cypher_{output_model_name}"
    f"_temperature{args.temperature}_top_p{args.top_p}"
    f"_threshold_ct{args.constraint_threshold}_ht{args.hop_threshold}_at{args.aggregation_trigger}"
)

from src.coa_cypher.coa_cypher import run_parallel_pipeline

os.makedirs(output_dir, exist_ok=True)
components = [
    "analyzer/Cypherbench",
    "analyzer_with_feedback/Cypherbench",
    "decomposer",
    "generator",
    "batch_generator",
    "generator_with_feedback",
    "critic",
    "verifier",
]
path = "coa_cypher"


prompt_templates = {}
for c in tqdm(components, desc="Loading prompt templates"):
    prompt_templates[c] = {}
    prompt_templates[c]["system"] = open(
        f"src/{path}/prompts/{c}/system_prompt.txt", "r", encoding="utf-8"
    ).read()
    prompt_templates[c]["user"] = open(
        f"src/{path}/prompts/{c}/user_prompt.txt", "r", encoding="utf-8"
    ).read()

configs = {
    "api_key": os.getenv("MARKET_API_KEY"),
    "base_url": os.getenv("MARKET_API_BASE"),
    "model": args.model,
    "temperature": args.temperature,
    "top_p": args.top_p,
    "max_tokens": args.max_tokens,
}

configs["extra_body"] = {
    "chat_template_kwargs": {"enable_thinking": True},
}

llm = init_llm(configs)

schema = read_json_file(f"benchmarks/Cypherbench/graphs/schemas/{args.cb_graph}_schema.json")
for entity in schema["entities"]:
    entity["properties"]["name"] = "str"

conn = Neo4jConnector(
    name="cypherbench-db",
    host="neo4j://127.0.0.1",
    port=7687,
    username="neo4j",
    password=os.getenv('NEO4J_PASSWORD'),
    database=args.cb_graph.replace("_", ".")
)

data_test = [item for item in test_data if item["graph"] == args.cb_graph]
result = run_for_graph(
    args.cb_graph,
    data_test,
    schema=schema,
    conn=conn,
    llm=llm,
    prompt_templates=prompt_templates,
    use_critic=args.use_critic,
    use_verifier=args.use_verifier,
    max_rounds=args.max_rounds,
    benchmark="Cypherbench",
    constraint_threshold=args.constraint_threshold,
    hop_threshold=args.hop_threshold,
    aggregation_trigger=args.aggregation_trigger,
)

