# Code for CoA-Cypher

## Environment Setup
1. Install the following libraries
```bash
neo4j~=6.0.3
pydantic~=2.12.5
tqdm~=4.67.1
langchain-openai~=1.1.0
langchain-core~=1.1.0
langchain-community
python-dotenv~=1.2.1
numpy>=1.24.0
```
2. Create a `.env` file from `.env.example`
```bash
cp .env.example .env
```
Then fill in the required values in `.env`
```bash
MARKET_API_BASE={your_base_url_here}
MARKET_API_KEY={your_api_key_here}
NEO4J_PASSWORD={your_neo4j_password}
NEO4J_USERNAME={your_neo4j_username}
```
3. Download the benchmark data from [Hugging Face](https://huggingface.co/datasets/fisherman611/full_text_to_cypher_benchmarks) and place it under the `benchmarks/` directory
4. Install and run Neo4j locally if you want to use `Cypherbench` or `Mind_the_query`
5. Create two local Neo4j instances named `cypherbench-db` and `mind-the-query-db`
6. Inside those instances, create empty databases for the graphs you want to use

For `Cypherbench`, the code expects database names such as `nba`, `company`, `geography`, `movie`, `politics`, `fictional.character`, and `flight.accident`.

For `Mind_the_query`, the database name must match the graph argument used in `main.py`, such as `bloom50`, `healthcare`, or `wwc`.

7. To load the `Cypherbench` graphs into your local Neo4j instance, run
```bash
python import_db_to_neo4j.py
```
8. For `Mind_the_query`, manually load the provided Neo4j dump files from `benchmarks/Mind_the_query/graphs/` into your local Neo4j instance. Make sure the restored database name matches the graph argument used in `main.py`
9. For `Neo4j_Text2Cypher`, the code connects to Neo4j's public demo instances at `bolt+s://demo.neo4jlabs.com`, so no local database import is required

## Usage
### Run inference with scripts
You can launch full benchmark runs from the `scripts/` folder.

Run all methods (`direct`, `cot`, `mac_sql`, `coa_cypher`) for one benchmark:
```bash
bash scripts/running_gpt_Cypherbench_all.sh
```

Run only one method via the same `*_all.sh` wrapper:
```bash
bash scripts/running_gpt_Cypherbench_all.sh gpt-oss-120b mac_sql
```

You can also override runtime params through environment variables:
```bash
TEMPERATURE=0.2 TOP_P=0.6 MAX_TOKENS=8192 SPLIT=-1 bash scripts/running_gpt_Cypherbench_all.sh
```

`*_all.sh` scripts available in `scripts/`:
- `running_{gemma|glm|gpt|llama}_{Cypherbench|Mind_the_query|Neo4j_Text2Cypher}_all.sh`

Run one method-specific script directly (single script run):
```bash
bash scripts/mac_sql/running_gpt_Cypherbench.sh
```

Example for another method/benchmark:
```bash
bash scripts/coa_cypher/running_llama_Mind_the_query.sh
```

Method-specific scripts live under:
- `scripts/direct/`
- `scripts/cot/`
- `scripts/mac_sql/`
- `scripts/coa_cypher/`

### Run inference with `main.py`
To run CoA-Cypher on the `nba` graph from `Cypherbench`, use:
```
python main.py --benchmark Cypherbench --cb_graph nba --method coa_cypher --model gpt-oss-120b --temperature 0.2 --top_p 0.6
```
Other common examples:
```
python main.py --benchmark Mind_the_query --mtq_graph bloom50 --method coa_cypher
```

```
python main.py --benchmark Neo4j_Text2Cypher --nt2c_graph bluesky --method coa_cypher
```

```
python main.py --benchmark Cypherbench --cb_graph nba --split 1 --method direct
```

```
python main.py --benchmark Cypherbench --cb_graph nba --split 1 --method cot
```

```
python main.py --benchmark Cypherbench --cb_graph nba --split 1 --method mac_sql
```

Useful arguments:
- `--benchmark`: `Cypherbench`, `Mind_the_query`, or `Neo4j_Text2Cypher`
- `--method`: `coa_cypher`, `direct`, `cot`, or `mac_sql`
- `--split`: limit the number of test samples to run
- `--use_critic`, `--use_verifier`, `--max_rounds`: control the CoA-Cypher feedback loop
- `--constraint_threshold`, `--hop_threshold`, `--aggregation_trigger`: control the heuristic router in CoA-Cypher

Generated predictions are saved to:

For `coa_cypher`:
> `results/{benchmark}/{model_group}/coa_cypher_{model}_temperature{temperature}_top_p{top_p}_threshold_ct{constraint_threshold}_ht{hop_threshold}_at{aggregation_trigger}/{graph}_cyphers_result.json`

For `direct`, `cot`, and `mac_sql`:
> `results/{benchmark}/{model_group}/{method}_{model}_temperature{temperature}_top_p{top_p}/{graph}_cyphers_result.json`

Here, `model_group` is derived from the model name prefix. For example, `gpt-oss-120b` is saved under `results/Cypherbench/gpt/...`.

Method-specific logs are saved under:
> logging_data/

## Evaluation
After generating predictions (`*_cyphers_result.json` per graph), run the metric scripts below.

### 1. Calculate metrics per graph
Cypherbench:
```bash
python src/calculate_scores_cypherbench.py \
  --result_dir results/Cypherbench/gpt/coa_cypher_gpt-oss-120b_temperature0.2_top_p0.6_threshold_ct2_ht2_atTrue
```

Mind_the_query:
```bash
python src/calculate_scores_mind_the_query.py \
  --result_dir results/Mind_the_query/gpt/coa_cypher_gpt-oss-120b_temperature0.2_top_p0.6_threshold_ct2_ht2_atTrue
```

Neo4j_Text2Cypher:
```bash
python src/calculate_scores_neo4j_text2cypher.py \
  --result_dir results/Neo4j_Text2Cypher/gpt/coa_cypher_gpt-oss-120b_temperature0.2_top_p0.6_threshold_ct2_ht2_atTrue
```

Useful options:
- `--graphs ...`: evaluate only selected graphs
- `--limit N`: evaluate only first N samples per graph
- `--flush_every N`: write checkpoint every N new samples
- `--output_dir PATH`: override default output (`<result_dir>/calculated_scores`)
- `--resume`: continue an interrupted run from existing output files; already-scored samples are skipped and only pending samples are evaluated

Example (resume an interrupted Cypherbench evaluation):
```bash
python src/calculate_scores_cypherbench.py \
  --result_dir results/Cypherbench/gpt/coa_cypher_gpt-oss-120b_temperature0.2_top_p0.6_threshold_ct2_ht2_atTrue \
  --resume
```

### 2. Merge all graph files into one JSON
```bash
python src/merge_json.py \
  --benchmark Cypherbench \
  --result_dir results/Cypherbench/gpt/coa_cypher_gpt-oss-120b_temperature0.2_top_p0.6_threshold_ct2_ht2_atTrue/calculated_scores \
  --output results/Cypherbench/gpt/coa_cypher_gpt-oss-120b_temperature0.2_top_p0.6_threshold_ct2_ht2_atTrue/calculated_scores/test_result.json
```

Set `--benchmark` to `Mind_the_query` or `Neo4j_Text2Cypher` for the other datasets.

### 3. Aggregate final scores
```bash
python src/calculate_scores_json.py \
  --input results/Cypherbench/gpt/coa_cypher_gpt-oss-120b_temperature0.2_top_p0.6_threshold_ct2_ht2_atTrue/calculated_scores/test_result.json \
  --output results/Cypherbench/gpt/mac_sql_gpt-oss-120b_temperature0.2_top_p0.6/calculated_scores/summary.json
```

This outputs the overall average of:
- `execution_accuracy`
- `psjs`
- `executable`


## Acknowledgement
The metric implementations are also adapted from the [CypherBench metrics package](https://github.com/megagonlabs/cypherbench/tree/main/cypherbench/metrics). This repository also includes baseline implementations for direct prompting, chain-of-thought prompting, and [MAC-SQL-style](https://arxiv.org/abs/2312.11242) decomposition and refinement.
