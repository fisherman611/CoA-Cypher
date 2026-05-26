#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-${1:-gpt-oss-120b}}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-0.6}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
SPLIT="${SPLIT:--1}"

GRAPHS=(
  "company"
  "fictional_character"
  "flight_accident"
  "geography"
  "movie"
  "nba"
  "politics"
)

echo "Running direct baseline on Cypherbench"
echo "Model: $MODEL"
echo "Temperature: $TEMPERATURE"
echo "Top-p: $TOP_P"
echo "Max tokens: $MAX_TOKENS"
echo "Split: $SPLIT"
echo

for graph in "${GRAPHS[@]}"; do
  echo "=== Running graph: $graph ==="
  python main.py \
    --benchmark Cypherbench \
    --cb_graph "$graph" \
    --method coa_cypher \
    --model "$MODEL" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --max_tokens "$MAX_TOKENS" \
    --split "$SPLIT" \
    --resume
  echo
done

echo "Completed coa_cypher Cypherbench runs for all graphs."
