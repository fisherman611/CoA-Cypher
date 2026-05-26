import json
import os
import re
from pathlib import Path
from statistics import mode
from time import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

from src.schema import Nl2CypherSample
from src.llm_services import parse_json_from_string, call_llm
from src.logger_config import setup_logger

LOG_DIR = "logging_data/coa_cypher"
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, f"log_{int(time())}.txt")
logger = setup_logger(__name__, log_file=log_path)


def _write_results_json(path, results):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    tmp_path.replace(path)


# ==============================================================
# HEURISTIC ROUTER (uses Analyzer output, no LLM call)
# ==============================================================
def heuristic_router(
    analyzer_output: dict,
    constraint_threshold: int = 2,
    hop_threshold: int = 2,
    aggregation_trigger: bool = True,
) -> dict:
    """
    Decide `simple` vs `complex` from the Analyzer's structured output

    Rules:
    - if constraints >= constraint_threshold -> complex
    - if hops >= hop_threshold -> complex
    - if any aggregation is present -> complex
    - otherwise -> simple

    Returns:
        dict: {"mode": "simple"|"complex", "reason": str}
    """
    constraints = analyzer_output.get("constraints", [])
    hops = analyzer_output.get("hops", 0)
    aggregations = analyzer_output.get("aggregations", [])

    num_constraints = len(constraints)
    is_complex = False
    reasons = []

    if num_constraints >= constraint_threshold:
        is_complex = True
        reasons.append(f"{num_constraints} constraints >= {constraint_threshold}")

    if hops >= hop_threshold:
        is_complex = True
        reasons.append(f"{hops} hops >= {hop_threshold}")

    if aggregation_trigger and aggregations:
        is_complex = True
        reasons.append(f"aggregation present: {aggregations}")

    if is_complex:
        mode = "complex"

    else:
        mode = "simple"

    reason = (
        "; ".join(reasons) if reasons else "atomic query with few constraints and hops"
    )

    result = {
        "mode": mode,
        "reason": f"heuristic_router: {reason}",
    }

    logger.info("-" * 60)
    logger.info(f"HEURISTIC ROUTER: {result}")
    logger.info("-" * 60)
    return result


# ==============================================================
# LAYER 1: ANALYZER (Selector + Router combined)
# ==============================================================
def analyzer(question, schema, llm, prompt_templates, benchmark):
    """
    Analyzer = Selector + Router merged into one agent.

    Outputs:
        {
            "selected_schema": {...},
            "constraints": [...],
            "hops": int,
            "aggregations": [...],
            "result_modifiers": [...]
        }

    The heuristic router then uses constraints/hops/aggregations
    to decide "simple" vs "complex".
    """
    logger.info("-" * 60)
    logger.info("ANALYZER: Selecting schema and extracting query metadata")
    logger.info(f"  Question: {question}")

    default_output = {
        "selected_schema": schema,
        "constraints": [],
        "hops": 0,
        "aggregations": [],
        "result_modifiers": [],
    }

    try:
        schema_str = json.dumps(schema, indent=4, ensure_ascii=False)

        res = call_llm(
            llm,
            prompt_templates[f"analyzer/{benchmark}"],
            {"question": question, "schema": schema_str}
        )

        data = parse_json_from_string(res["final_answer"])
        
        if not data:
            logger.warning("ANALYZER: Failed to parse response. Using full schema.")
            return default_output
        
        selected_schema = data.get("selected_schema", schema)
        constraints = data.get("constraints", [])
        hops = data.get("hops", 0)
        aggregations = data.get("aggregations", [])
        result_modifiers = data.get("result_modifiers", [])

        # Log details
        if isinstance(selected_schema, dict):
            if benchmark != "Mind_the_query":
                entities = selected_schema.get("entities", [])
                relations = selected_schema.get("relations", [])
                logger.info(
                    f"  Selected schema — Entities: {len(entities)}, Relations: {len(relations)}"
                )
                if entities:
                    entity_labels = [e.get("label", "Unknown") for e in entities]
                    logger.info(f"  Entity labels: {entity_labels}")
                if relations:
                    relation_labels = [
                        f"{r.get('label', 'Unknown')} ({r.get('subj_label', '?')}->{r.get('obj_label', '?')})"
                        for r in relations
                    ]
                    logger.info(f"  Relations: {relation_labels}")
            else:
                node_props = selected_schema.get("node_props", {})
                rel_props = selected_schema.get("rel_props", {})
                relationships = selected_schema.get("relationships", [])
                
                logger.info(
                    f"  Selected schema — Nodes: {len(node_props)}, Rel props: {len(rel_props)}, Relationships: {len(relationships)}"
                )
                if node_props:
                    logger.info(f"  Node labels: {list(node_props.keys())}")
                if relationships:
                    rel_labels = [
                        f"{r.get('type', 'Unknown')} ({r.get('start', '?')}->{r.get('end', '?')})"
                        for r in relationships if isinstance(r, dict)
                    ]
                    logger.info(f"  Relationships: {rel_labels}")
        else:
            logger.warning("  Selected schema is not a dict, using as-is")

        logger.info(f"  Constraints ({len(constraints)}): {constraints}")
        logger.info(f"  Hops: {hops}")
        logger.info(f"  Aggregations: {aggregations}")
        logger.info(f"  Result modifiers: {result_modifiers}")
        logger.info("-" * 60)

        return {
            "selected_schema": selected_schema,
            "constraints": constraints,
            "hops": hops,
            "aggregations": aggregations,
            "result_modifiers": result_modifiers,
        }

    except Exception as e:
        logger.error(f"ANALYZER error: {e}")
        return default_output
    

def analyzer_with_feedback(
    question, schema, previous_selected_schema, previous_query,
    feedback, llm, prompt_templates, benchmark
):
    """
    Analyzer with feedback: receives structured feedback from the Critic
    indicating schema selection issues, and re-selects the sub-schema
    while correcting the identified problems.

    Args:
        schema: the FULL schema (not the previously selected sub-schema)
        previous_selected_schema: the sub-schema from the prior Analyzer run
        previous_query: the Cypher query that failed
        feedback: the Critic's structured feedback dict

    Returns:
        Same structure as analyzer():
        {
            "selected_schema": {...},
            "constraints": [...],
            "hops": int,
            "aggregations": [...],
            "result_modifiers": [...]
        }
    """
    logger.info("-" * 60)
    logger.info("ANALYZER (with feedback): Re-selecting schema based on Critic feedback")
    logger.info(f"  Question: {question}")
    logger.info(f"  Critic diagnosis: {feedback.get('diagnosis', 'N/A')}")

    default_output = {
        "selected_schema": schema,
        "constraints": [],
        "hops": 0,
        "aggregations": [],
        "result_modifiers": [],
    }

    try:
        schema_str = json.dumps(schema, indent=4, ensure_ascii=False)
        prev_schema_str = json.dumps(previous_selected_schema, indent=4, ensure_ascii=False)

        res = call_llm(
            llm,
            prompt_templates[f"analyzer_with_feedback/{benchmark}"],
            {
                "question": question,
                "schema": schema_str,
                "previous_selected_schema": prev_schema_str,
                "previous_query": previous_query,
                "feedback": json.dumps(feedback, indent=4, ensure_ascii=False),
            },
        )

        data = parse_json_from_string(res["final_answer"])

        if not data:
            logger.warning("ANALYZER (with feedback): Failed to parse. Using full schema.")
            return default_output

        selected_schema = data.get("selected_schema", schema)
        constraints = data.get("constraints", [])
        hops = data.get("hops", 0)
        aggregations = data.get("aggregations", [])
        result_modifiers = data.get("result_modifiers", [])
        corrections = data.get("corrections_applied", [])

        # Log details
        if isinstance(selected_schema, dict):
            if benchmark != "Mind_the_query":
                entities = selected_schema.get("entities", [])
                relations = selected_schema.get("relations", [])
                logger.info(
                    f"  Re-selected schema — Entities: {len(entities)}, Relations: {len(relations)}"
                )
                if entities:
                    entity_labels = [e.get("label", "Unknown") for e in entities]
                    logger.info(f"  Entity labels: {entity_labels}")
                if relations:
                    relation_labels = [
                        f"{r.get('label', 'Unknown')} ({r.get('subj_label', '?')}->{r.get('obj_label', '?')})"
                        for r in relations
                    ]
                    logger.info(f"  Relations: {relation_labels}")
            else:
                node_props = selected_schema.get("node_props", {})
                rel_props = selected_schema.get("rel_props", {})
                relationships = selected_schema.get("relationships", [])
                
                logger.info(
                    f"  Re-selected schema — Nodes: {len(node_props)}, Rel props: {len(rel_props)}, Relationships: {len(relationships)}"
                )
                if node_props:
                    logger.info(f"  Node labels: {list(node_props.keys())}")
                if relationships:
                    rel_labels = [
                        f"{r.get('type', 'Unknown')} ({r.get('start', '?')}->{r.get('end', '?')})"
                        for r in relationships if isinstance(r, dict)
                    ]
                    logger.info(f"  Relationships: {rel_labels}")
        else:
            logger.warning("  Re-selected schema is not a dict, using as-is")

        logger.info(f"  Constraints ({len(constraints)}): {constraints}")
        logger.info(f"  Hops: {hops}")
        logger.info(f"  Aggregations: {aggregations}")
        logger.info(f"  Result modifiers: {result_modifiers}")
        if corrections:
            logger.info(f"  Corrections applied: {corrections}")
        logger.info("-" * 60)

        return {
            "selected_schema": selected_schema,
            "constraints": constraints,
            "hops": hops,
            "aggregations": aggregations,
            "result_modifiers": result_modifiers,
        }

    except Exception as e:
        logger.error(f"ANALYZER (with feedback) error: {e}")
        return default_output

# ==============================================================
# LAYER 2: DECOMPOSER (only for complex queries)
# ==============================================================
def decomposer(question, schema, constraints, hops, llm, prompt_templates):
    """
    Decomposer: decomposes the question into sub-questions AND produces
    a partial Cypher fragment for each sub-question.

    Only invoked for "complex" queries. Always uses multi-strategy (K=3).

    Args:
        constraints: list of constraint dicts from the Analyzer
        hops: int, number of hops from the Analyzer
    """
    logger.info("-" * 60)
    logger.info("DECOMPOSER: Decomposing complex question with partial Cypher")
    logger.info(f"  Question: {question}")
    logger.info(f"  Constraints: {len(constraints)}, Hops: {hops}")

    try:
        schema_str = json.dumps(schema, indent=4, ensure_ascii=False)
        constraints_str = json.dumps(constraints, indent=4, ensure_ascii=False)

        res = call_llm(
            llm,
            prompt_templates["decomposer"],
            {
                "question": question,
                "schema": schema_str,
                "constraints": constraints_str,
                "hops": str(hops),
            },
        )
        data = parse_json_from_string(res["final_answer"])

        if data:
            strategies = data.get("strategies", [])
            result = {"strategies": strategies}

            logger.info(f"DECOMPOSER: Generated {len(strategies)} strategies")
            for idx, strategy in enumerate(strategies):
                logger.info(
                    f"  [Strategy {idx + 1}] {strategy.get('strategy_name', 'unnamed')}"
                )
                sqs = strategy.get("sub_questions", [])
                for sq_idx, sq in enumerate(sqs):
                    logger.info(f"    [{sq_idx + 1}] {sq.get('description', 'N/A')}")
                    if sq.get("partial_cypher"):
                        logger.info(f"        Partial Cypher: {sq['partial_cypher']}")

            logger.info("-" * 60)
            return result

        logger.error("DECOMPOSER: Parse failed.")
        logger.info("-" * 60)
        return {"strategies": []}

    except Exception as e:
        logger.error(f"DECOMPOSER: Error — {e}")
        logger.info("-" * 60)
        return {"strategies": []}
    

# ==============================================================
# LAYER 3: GENERATE CYPHER QUERY (GENERATOR)
# ==============================================================
def generator(question, schema, sub_questions, analysis, llm, prompt_templates):
    """
    Generator produces a Cypher query from the question, decomposed sub-questions
    (with partial Cypher), and the Analyzer's structured metadata.
    """
    logger.info("-" * 60)
    logger.info("GENERATOR: Generating initial Cypher")
    logger.info(f"  Question: {question}")
    logger.info(f"  Sub-questions: {len(sub_questions)}")

    try:
        schema_str = json.dumps(schema, indent=4, ensure_ascii=False)
        sub_questions_str = json.dumps(sub_questions, indent=4, ensure_ascii=False)
        analysis_str = json.dumps(analysis, indent=4, ensure_ascii=False)

        res = call_llm(
            llm,
            prompt_templates["generator"],
            {
                "question": question,
                "schema": schema_str,
                "sub_questions": sub_questions_str,
                "analysis": analysis_str,
            },
        )

        gen_data = parse_json_from_string(res["final_answer"])

        if gen_data:
            cypher = gen_data.get("cypher", "")
            logger.info(f"GENERATOR: Generated Cypher:\n{cypher}")
            logger.info("-" * 60)
            return cypher
        else:
            logger.warning("GENERATOR: Failed to parse response")
            logger.info("-" * 60)
            return ""

    except Exception as e:
        logger.error(f"GENERATOR error: {e}")
        logger.info("-" * 60)
        return ""


def generator_with_feedback(
    question, schema, sub_questions, analysis, previous_query,
    execution_result, feedback, llm, prompt_templates
):
    """
    Generator with feedback: receives structured feedback from Critic (syntax/schema error)
    or Verifier (semantic mismatch) and produces a corrected Cypher query.
    """
    logger.info("-" * 60)
    logger.info("GENERATOR (with feedback): Generating corrected Cypher based on feedback")
    logger.info(f"  Previous query:\n{previous_query}")

    try:
        schema_str = json.dumps(schema, indent=4, ensure_ascii=False)
        sub_questions_str = json.dumps(sub_questions, indent=4, ensure_ascii=False)
        analysis_str = json.dumps(analysis, indent=4, ensure_ascii=False)

        res = call_llm(
            llm,
            prompt_templates["generator_with_feedback"],
            {
                "question": question,
                "schema": schema_str,
                "sub_questions": sub_questions_str,
                "analysis": analysis_str,
                "previous_query": previous_query,
                "execution_result": json.dumps(execution_result, indent=4, ensure_ascii=False),
                "feedback": json.dumps(feedback, indent=4, ensure_ascii=False),
            },
        )

        gen_data = parse_json_from_string(res["final_answer"])

        if gen_data:
            new_query = gen_data.get("cypher", previous_query)
            fix_reason = gen_data.get("fix_reason", "No reason provided")

            logger.info(f"Generation with feedback successful!")
            logger.info(f"Reason: {fix_reason}")

            if new_query != previous_query:
                logger.info(f"New query:\n{new_query}")
            else:
                logger.info("Query unchanged after feedback")

            logger.info("-" * 60)
            return new_query
        else:
            logger.warning("Failed to parse generator_with_feedback response")
            logger.info("-" * 60)
            return previous_query

    except Exception as e:
        logger.error(f"Generator with feedback error: {e}")
        logger.info("-" * 60)
        return previous_query


def batch_generator(
    question, schema, analysis, previous_queries,
    execution_results, feedbacks, llm, prompt_templates
):
    """
    Batch Generator receives multiple candidate queries (from diverse strategies),
    their execution results, and structured feedback, then synthesizes ONE final
    improved Cypher query.

    Returns:
        str: final synthesized Cypher query
    """
    logger.info("-" * 60)
    logger.info("BATCH_GENERATOR: Synthesizing one final Cypher from multiple candidates")

    try:
        if not previous_queries:
            logger.warning("BATCH_GENERATOR: No previous queries provided")
            logger.info("-" * 60)
            return ""

        logger.info(f"BATCH_GENERATOR: Number of candidate queries = {len(previous_queries)}")

        for i, query in enumerate(previous_queries):
            logger.info(f"[Candidate {i}] Previous query:\n{query}")

        schema_str = json.dumps(schema, indent=4, ensure_ascii=False)
        analysis_str = json.dumps(analysis, indent=4, ensure_ascii=False)

        res = call_llm(
            llm,
            prompt_templates["batch_generator"],
            {
                "question": question,
                "schema": schema_str,
                "analysis": analysis_str,
                "previous_queries": json.dumps(previous_queries, indent=4, ensure_ascii=False),
                "execution_results": json.dumps(execution_results, indent=4, ensure_ascii=False),
                "feedbacks": json.dumps(feedbacks, indent=4, ensure_ascii=False),
            },
        )

        gen_data = parse_json_from_string(res["final_answer"])

        if not gen_data:
            logger.warning("BATCH_GENERATOR: Failed to parse response. Falling back to first query.")
            logger.info("-" * 60)
            return previous_queries[0]

        if "cypher" in gen_data:
            new_query = gen_data.get("cypher", previous_queries[0])
            fix_reason = gen_data.get("fix_reason", "No reason provided")

            logger.info("BATCH_GENERATOR: Synthesis successful")
            logger.info(f"Reason: {fix_reason}")

            if new_query != previous_queries[0]:
                logger.info(f"Final synthesized query:\n{new_query}")
            else:
                logger.info("Final query is same as first candidate")

            logger.info("-" * 60)
            return new_query

        logger.warning("BATCH_GENERATOR: Response missing 'cypher'. Falling back to first query.")
        logger.info("-" * 60)
        return previous_queries[0]

    except Exception as e:
        logger.error(f"BATCH_GENERATOR error: {e}")
        logger.info("-" * 60)
        return previous_queries[0] if previous_queries else ""
    

# =======================================================================
# LAYER 4: ASSESS THE GENERATED CYPHER AND PROVIDE FEEDBACK (CRITIC + VERIFIER)
# =======================================================================
def critic(question, schema, previous_query, execution_result, llm, prompt_templates):
    """
    Critic diagnoses SYNTAX and SCHEMA errors only.
    Routes feedback to either "generator" or "analyzer" via feedback_target.
    """
    logger.info("-" * 60)
    logger.info("CRITIC: Reviewing execution error and producing feedback")

    try:
        schema_str = json.dumps(schema, indent=4, ensure_ascii=False)

        res = call_llm(
            llm,
            prompt_templates["critic"],
            {
                "question": question,
                "schema": schema_str,
                "previous_query": previous_query,
                "execution_result": json.dumps(execution_result, indent=4, ensure_ascii=False),
            },
        )
        feedback = parse_json_from_string(res["final_answer"])
        if not feedback:
            logger.warning("Critic parse failed. Using fallback feedback.")
            feedback = {
                "status": "needs_fix",
                "error_type": "unknown",
                "feedback_target": "generator",
                "diagnosis": "Could not parse critic output.",
                "schema_issues": [],
                "patch_instructions": [
                    "Regenerate query conservatively using only schema labels and properties."
                ],
                "regenerate": True,
            }
        logger.info("CRITIC feedback:")
        logger.info(json.dumps(feedback, indent=2, ensure_ascii=False))
        logger.info("-" * 60)
        return feedback

    except Exception as e:
        logger.error(f"Critic error: {e}")
        return {
            "status": "needs_fix",
            "error_type": "unknown",
            "feedback_target": "generator",
            "diagnosis": str(e),
            "schema_issues": [],
            "patch_instructions": ["Regenerate query conservatively."],
            "regenerate": True,
        }


def verifier(question, schema, cypher, llm, prompt_templates):
    """
    Verifier compares (QUESTION intent) vs (CYPHER intent) and returns
    structured feedback if there is a semantic mismatch.
    Only redirects to Generator.
    """
    logger.info("-" * 60)
    logger.info("VERIFIER: Round-trip semantic validation (NL ⇄ Cypher)")

    try:
        schema_str = json.dumps(schema, indent=4, ensure_ascii=False)
        res = call_llm(
            llm,
            prompt_templates["verifier"],
            {
                "question": question,
                "schema": schema_str,
                "cypher": cypher,
            },
        )
        sim_fb = parse_json_from_string(res["final_answer"])
        if not sim_fb:
            logger.warning("Verifier parse failed; defaulting to looks_ok")
            sim_fb = {
                "status": "looks_ok",
                "error_type": "unknown",
                "diagnosis": "Could not parse verifier output.",
                "intent_question": {
                    "targets": [], "filters": [], "aggregations": [],
                    "relationships": [], "grouping": [], "result_modifiers": [],
                    "notes": ""
                },
                "intent_cypher": {
                    "targets": [], "filters": [], "aggregations": [],
                    "relationships": [], "grouping": [], "result_modifiers": [],
                    "notes": ""
                },
                "mismatches": [],
                "patch_instructions": [],
                "regenerate": False,
            }

        logger.info("VERIFIER feedback:")
        logger.info(json.dumps(sim_fb, indent=2, ensure_ascii=False))
        logger.info("-" * 60)
        return sim_fb

    except Exception as e:
        logger.error(f"Verifier error: {e}")
        return {
            "status": "looks_ok",
            "error_type": "unknown",
            "diagnosis": str(e),
            "intent_question": {
                "targets": [], "filters": [], "aggregations": [],
                "relationships": [], "grouping": [], "result_modifiers": [],
                "notes": ""
            },
            "intent_cypher": {
                "targets": [], "filters": [], "aggregations": [],
                "relationships": [], "grouping": [], "result_modifiers": [],
                "notes": ""
            },
            "mismatches": [],
            "patch_instructions": [],
            "regenerate": False,
        }


# ======================================================
# MAIN PIPELINE
# ======================================================
def pipeline(
    sample,
    schema,
    conn,
    llm,
    prompt_templates,
    use_critic=True,
    use_verifier=True,
    max_rounds=5,
    benchmark="Cypherbench",
    constraint_threshold=2,
    hop_threshold=2,
    aggregation_trigger=True,
    use_analyzer=True,
    use_decomposer=True,
):
    question = sample.nl_question
    if benchmark == "Neo4j_Text2Cypher":
        schema = sample.schema if hasattr(sample, "schema") and sample.schema else None

    logger.info("=" * 80)
    logger.info(f"Processing QID {sample.qid}: {question}")
    logger.info("=" * 80)

    try:
        # ── LAYER 1: ANALYZER (schema selection + query metadata) ──
        if use_analyzer:
            analyzer_output = analyzer(question, schema, llm, prompt_templates, benchmark)
            selected_schema = analyzer_output["selected_schema"]
            constraints = analyzer_output["constraints"]
            hops = analyzer_output["hops"]
        else:
            logger.info("ABLATION: Analyzer disabled. Using full schema and empty analysis.")
            analyzer_output = {
                "selected_schema": schema,
                "constraints": [],
                "hops": 0,
                "aggregations": [],
                "result_modifiers": [],
                "ablation": "analyzer_disabled",
            }
            selected_schema = schema
            constraints = []
            hops = 0

        # Build analysis dict to pass downstream
        analysis = {
            "constraints": constraints,
            "hops": hops,
            "aggregations": analyzer_output.get("aggregations", []),
            "result_modifiers": analyzer_output.get("result_modifiers", []),
        }

        # ── HEURISTIC ROUTER (no LLM call) ──
        route = heuristic_router(analyzer_output, constraint_threshold, hop_threshold, aggregation_trigger)
        mode = route["mode"]
        if not use_analyzer:
            route = {
                **route,
                "mode": "simple",
                "original_mode": mode,
                "ablation": "analyzer_disabled",
            }
            mode = "simple"
            logger.info("ABLATION: Analyzer disabled. Forcing generator-only path.")
        elif not use_decomposer:
            route = {
                **route,
                "mode": "simple",
                "original_mode": mode,
                "ablation": "decomposer_disabled",
            }
            mode = "simple"
            logger.info("ABLATION: Decomposer disabled. Forcing generator-only path.")

        # ────────────────────────────────────────────────────────────
        # SIMPLE PATH
        # ────────────────────────────────────────────────────────────
        if mode == "simple":
            logger.info("SIMPLE mode: Skipping decomposer, sending directly to generator.")
            sub_questions = []
            decomposition_result = {"sub_questions": []}

            # GENERATOR (initial, no decomposition)
            cypher = generator(
                question=question,
                schema=selected_schema,
                sub_questions=sub_questions,
                analysis=analysis,
                llm=llm,
                prompt_templates=prompt_templates,
            )

            for round_idx in range(max_rounds):
                logger.info(f"\n[ROUND {round_idx + 1}/{max_rounds}] Executing Cypher:")
                logger.info(f"{cypher}")
                execution_result = conn.run_query_advance(cypher)

                # ── PATH A: Execution error → Critic ──
                if not execution_result.get("success", False):
                    if not use_critic:
                        logger.warning("Execution failed but critic disabled. Stopping.")
                        break

                    logger.warning("Execution failed. Sending to CRITIC.")
                    fb = critic(
                        question=question,
                        schema=selected_schema,
                        previous_query=cypher,
                        execution_result=execution_result,
                        llm=llm,
                        prompt_templates=prompt_templates,
                    )

                    if not fb.get("regenerate", True):
                        logger.warning("Critic advised no regeneration. Stopping.")
                        break

                    # Check feedback_target: redirect to analyzer with feedback if schema is wrong
                    if fb.get("feedback_target") == "analyzer":
                        if not use_analyzer:
                            logger.warning("CRITIC requested analyzer rerun, but analyzer is disabled.")
                            break
                        logger.warning("CRITIC: Schema issue detected. Re-running ANALYZER with feedback.")
                        analyzer_output = analyzer_with_feedback(
                            question=question,
                            schema=schema,
                            previous_selected_schema=selected_schema,
                            previous_query=cypher,
                            feedback=fb,
                            llm=llm,
                            prompt_templates=prompt_templates,
                            benchmark=benchmark,
                        )
                        selected_schema = analyzer_output["selected_schema"]
                        analysis = {
                            "constraints": analyzer_output.get("constraints", []),
                            "hops": analyzer_output.get("hops", 0),
                            "aggregations": analyzer_output.get("aggregations", []),
                            "result_modifiers": analyzer_output.get("result_modifiers", []),
                        }
                        # Re-generate with corrected schema
                        cypher = generator(
                            question=question,
                            schema=selected_schema,
                            sub_questions=sub_questions,
                            analysis=analysis,
                            llm=llm,
                            prompt_templates=prompt_templates,
                        )
                        continue

                    # feedback_target == "generator": fix the query
                    new_cypher = generator_with_feedback(
                        question=question,
                        schema=selected_schema,
                        sub_questions=sub_questions,
                        analysis=analysis,
                        previous_query=cypher,
                        execution_result=execution_result,
                        feedback=fb,
                        llm=llm,
                        prompt_templates=prompt_templates,
                    )

                    if new_cypher == cypher:
                        logger.warning("Generated query unchanged after Critic feedback. Stopping loop.")
                        break

                    cypher = new_cypher
                    continue

                logger.info("Query executed successfully.")
        
                # ── PATH B: Executable → Intent Verifier ──
                if use_verifier:
                    sim_fb = verifier(
                        question=question,
                        schema=selected_schema,
                        cypher=cypher,
                        llm=llm,
                        prompt_templates=prompt_templates,
                    )

                    if sim_fb.get("status") != "needs_fix" or not sim_fb.get("regenerate", True):
                        logger.info("VERIFIER: Query looks good. Done.")
                        break

                    # Verifier always redirects to generator
                    logger.warning("VERIFIER detected semantic mismatch. Sending to GENERATOR...")
                    new_cypher = generator_with_feedback(
                        question=question,
                        schema=selected_schema,
                        sub_questions=sub_questions,
                        analysis=analysis,
                        previous_query=cypher,
                        execution_result=execution_result,
                        feedback=sim_fb,
                        llm=llm,
                        prompt_templates=prompt_templates,
                    )

                    if new_cypher == cypher:
                        logger.warning("Generated query unchanged after Verifier feedback. Stopping loop.")
                        break

                    cypher = new_cypher
                    continue

                # Verifier disabled and execution succeeded → DONE
                break

            sample.pred_cypher = cypher

        # ────────────────────────────────────────────────────────────
        # COMPLEX PATH
        # ────────────────────────────────────────────────────────────
        else:
            # DECOMPOSER (question → sub-questions + partial Cypher)
            decomposition_result = decomposer(
                question=question,
                schema=selected_schema,
                constraints=constraints,
                hops=hops,
                llm=llm,
                prompt_templates=prompt_templates,
            )

            # 1. Get all decomposition strategies
            strategies = decomposition_result.get("strategies", [])

            if not strategies:
                logger.warning("No strategies returned by decomposer. Using empty sub-questions.")
                strategies = [{"strategy_name": "fallback", "sub_questions": []}]

            # 2. Generate Cypher for each strategy using the Generator
            previous_queries = []
            all_sub_questions = []
            for idx, strategy in enumerate(strategies):
                strategy_sub_questions = strategy.get("sub_questions", [])
                all_sub_questions.append(strategy_sub_questions)

                logger.info(f"\n[Strategy {idx + 1}] Generating Cypher...")
                cypher_candidate = generator(
                    question=question,
                    schema=selected_schema,
                    sub_questions=strategy_sub_questions,
                    analysis=analysis,
                    llm=llm,
                    prompt_templates=prompt_templates,
                )
                previous_queries.append(cypher_candidate)

            if not previous_queries:
                logger.warning("No queries generated. Creating an empty query.")
                previous_queries = [""]

            # 3. Gather structural feedback for each candidate cypher.
            # Ablation without feedback layer sends candidates directly to batch_generator.
            execution_results = []
            feedbacks = []

            if use_critic or use_verifier:
                for cand_cypher in previous_queries:
                    logger.info(f"\nExecuting Candidate Cypher:\n{cand_cypher}")
                    cand_exec_result = conn.run_query_advance(cand_cypher)
                    execution_results.append(cand_exec_result)

                    # If execution fails, get critic feedback
                    if not cand_exec_result.get("success", False):
                        if use_critic:
                            fb = critic(
                                question=question,
                                schema=selected_schema,
                                previous_query=cand_cypher,
                                execution_result=cand_exec_result,
                                llm=llm,
                                prompt_templates=prompt_templates,
                            )
                        else:
                            fb = {"status": "needs_fix", "feedback_target": "generator", "regenerate": False}
                        feedbacks.append(fb)

                    # If execution succeeds, optionally get verifier feedback
                    else:
                        if use_verifier:
                            fb = verifier(
                                question=question,
                                schema=selected_schema,
                                cypher=cand_cypher,
                                llm=llm,
                                prompt_templates=prompt_templates,
                            )
                        else:
                            fb = {"status": "looks_ok", "regenerate": False}
                        feedbacks.append(fb)
            else:
                logger.info("ABLATION: Feedback layer disabled. Skipping candidate execution before batch_generator.")

            # 4. Check if any critic feedback targets the analyzer
            analyzer_rerun_needed = any(
                fb.get("feedback_target") == "analyzer"
                for fb in feedbacks
                if fb.get("status") == "needs_fix"
            )

            if analyzer_rerun_needed and not use_analyzer:
                logger.warning("Feedback requested analyzer rerun, but analyzer is disabled.")

            if analyzer_rerun_needed and use_analyzer:
                # Collect all analyzer-targeted feedback for context
                analyzer_feedbacks = [
                    fb for fb in feedbacks
                    if fb.get("status") == "needs_fix" and fb.get("feedback_target") == "analyzer"
                ]
                # Use the first analyzer-targeted feedback (most representative)
                # and the corresponding query for context
                representative_fb = analyzer_feedbacks[0]
                fb_idx = feedbacks.index(representative_fb)
                representative_query = previous_queries[fb_idx] if fb_idx < len(previous_queries) else ""

                logger.warning("CRITIC(s) flagged schema issues. Re-running ANALYZER with feedback.")
                analyzer_output = analyzer_with_feedback(
                    question=question,
                    schema=schema,
                    previous_selected_schema=selected_schema,
                    previous_query=representative_query,
                    feedback=representative_fb,
                    llm=llm,
                    prompt_templates=prompt_templates,
                    benchmark=benchmark
                )
                selected_schema = analyzer_output["selected_schema"]
                analysis = {
                    "constraints": analyzer_output.get("constraints", []),
                    "hops": analyzer_output.get("hops", 0),
                    "aggregations": analyzer_output.get("aggregations", []),
                    "result_modifiers": analyzer_output.get("result_modifiers", []),
                }

            # 5. Synthesize ONE master cypher query using batch_generator
            cypher = batch_generator(
                question=question,
                schema=selected_schema,
                analysis=analysis,
                previous_queries=previous_queries,
                execution_results=execution_results,
                feedbacks=feedbacks,
                llm=llm,
                prompt_templates=prompt_templates,
            )

            # 6. Iteratively refine the synthesized cypher via critic/verifier → generator_with_feedback
            combined_sub_questions = []
            for sqs in all_sub_questions:
                combined_sub_questions.extend(sqs)

            for round_idx in range(1, max_rounds):
                logger.info(f"\n[ROUND {round_idx + 1}/{max_rounds}] Executing Synthesized Cypher:\n{cypher}")
                exec_result = conn.run_query_advance(cypher)

                if not exec_result.get("success", False):
                    if not use_critic:
                        break
                    fb = critic(question, selected_schema, cypher, exec_result, llm, prompt_templates)
                    if not fb.get("regenerate", True):
                        break

                    # Check feedback_target
                    if fb.get("feedback_target") == "analyzer":
                        if not use_analyzer:
                            logger.warning("CRITIC requested analyzer rerun, but analyzer is disabled.")
                            break
                        logger.warning("CRITIC: Schema issue in refinement loop. Re-running ANALYZER with feedback.")
                        analyzer_output = analyzer_with_feedback(
                            question=question,
                            schema=schema,
                            previous_selected_schema=selected_schema,
                            previous_query=cypher,
                            feedback=fb,
                            llm=llm,
                            prompt_templates=prompt_templates,
                            benchmark=benchmark,
                        )
                        selected_schema = analyzer_output["selected_schema"]
                        analysis = {
                            "constraints": analyzer_output.get("constraints", []),
                            "hops": analyzer_output.get("hops", 0),
                            "aggregations": analyzer_output.get("aggregations", []),
                            "result_modifiers": analyzer_output.get("result_modifiers", []),
                        }
                        cypher = generator(
                            question, selected_schema, combined_sub_questions,
                            analysis, llm, prompt_templates
                        )
                        continue

                    cypher = generator_with_feedback(
                        question, selected_schema, combined_sub_questions,
                        analysis, cypher, exec_result, fb, llm, prompt_templates
                    )
                    continue

                if use_verifier:
                    sim_fb = verifier(question, selected_schema, cypher, llm, prompt_templates)
                    if sim_fb.get("status") != "needs_fix" or not sim_fb.get("regenerate", True):
                        break
                    cypher = generator_with_feedback(
                        question, selected_schema, combined_sub_questions,
                        analysis, cypher, exec_result, sim_fb, llm, prompt_templates
                    )
                    continue
                break

            sample.pred_cypher = cypher

        return {
            "qid": sample.qid,
            "graph": sample.graph,
            "question": question,
            "gold": sample.gold_cypher,
            "pred": cypher,
            "status": "ok",
            "analyzer_output": analyzer_output,
            "route": route,
        }

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise

# ======================================================
# PARALLEL RUNNER
# ======================================================
def run_parallel_pipeline(
    test_data,
    schema,
    conn,
    llm,
    prompt_templates,
    max_workers=4,
    use_critic=True,
    use_verifier=True,
    max_rounds=5,
    benchmark="Cypherbench",
    constraint_threshold=2,
    hop_threshold=2,
    aggregation_trigger=True,
    output_path=None,
    flush_every=10,
    initial_results=None,
    skip_qids=None,
    use_analyzer=True,
    use_decomposer=True,
):
    results = list(initial_results or [])
    errors = []
    processed_count = 0
    skip_qids = set(skip_qids or [])
    existing_count = len(results)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_meta = {}

        for item in test_data:
            sample = Nl2CypherSample(**item)
            if sample.qid in skip_qids:
                continue
            future = executor.submit(
                pipeline,
                sample,
                schema,
                conn,
                llm,
                prompt_templates,
                use_critic,
                use_verifier,
                max_rounds,
                benchmark,
                constraint_threshold,
                hop_threshold,
                aggregation_trigger,
                use_analyzer,
                use_decomposer,
            )
            future_to_meta[future] = {
                "qid": sample.qid,
                "graph": sample.graph,
                "question": sample.nl_question,
                "gold": sample.gold_cypher,
            }

        for future in tqdm(
            as_completed(future_to_meta),
            total=len(future_to_meta),
            desc="Processing samples",
        ):
            processed_count += 1
            meta = future_to_meta[future]
            qid = meta["qid"]
            try:
                results.append(future.result())
            except Exception as e:
                errors.append({"qid": qid, "error": str(e)})
                logger.error(f"Error processing QID {qid}: {e}")
                results.append(
                    {
                        "qid": qid,
                        "graph": meta["graph"],
                        "question": meta["question"],
                        "gold": meta["gold"],
                        "pred": "",
                        "status": "failed",
                        "error": str(e),
                    }
                )

            if output_path and flush_every > 0 and processed_count % flush_every == 0:
                _write_results_json(output_path, results)
                logger.info(
                    f"Checkpoint saved to {output_path} ({processed_count}/{len(future_to_meta)} processed)"
                )

    if output_path:
        _write_results_json(output_path, results)
        logger.info(f"Final results saved to {output_path}")

    logger.info(
        f"Finished. Existing: {existing_count}, New processed: {len(future_to_meta)}, Success total: {len(results)}, Failed: {len(errors)}"
    )
    return results
