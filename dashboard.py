"""
dashboard.py

Compare AI-generated test-plan CSVs against a stock (ground-truth) test plan.
The stock plan is a coverage checklist and may omit STR and expected results.
Only generated plans are scored.
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

# Streamlit keeps old project modules in memory across reruns. Drop them so
# every run imports the files currently on disk (avoids stale ImportError).
_PROJECT_ROOT = Path(__file__).resolve().parent
_LOCAL_MODULES = {path.stem for path in _PROJECT_ROOT.glob("*.py")} - {"dashboard"}
for _mod_name in list(sys.modules):
    if _mod_name.split(".")[0] in _LOCAL_MODULES:
        del sys.modules[_mod_name]

from csv_parser import filter_plan_by_areas, ingest_and_parse_test_plan_csv, unique_areas
from evaluator import compare_and_evaluate_plan
from schemas import ModelEvaluation
from source_parser import merge_source_intents

LLM_SLOTS = ("Gemini", "ChatGPT", "Rovo")

METRIC_COLUMNS = [
    ("Coverage", "requirement_coverage"),
    ("Match conf.", "match_confidence"),
    ("Area cov.", "area_coverage"),
    ("Source cov.", "source_coverage"),
    ("Source fit", "source_fidelity"),
    ("Edge cases", "edge_case_quality"),
    ("Actionable", "actionability"),
    ("STR", "str_completeness"),
    ("Expected result", "expected_result_quality"),
    ("Redundancy", "redundancy"),
    ("Unique STR", "procedure_uniqueness"),
    ("Scope drift", "scope_drift"),
    ("Traceability", "traceability"),
]
METRIC_DETAILS = [
    ("Coverage", "requirement_coverage"),
    ("Match confidence", "match_confidence"),
    ("Area coverage", "area_coverage"),
    ("Source coverage", "source_coverage"),
    ("Source fidelity", "source_fidelity"),
    ("Edge cases", "edge_case_quality"),
    ("Actionability", "actionability"),
    ("STR completeness", "str_completeness"),
    ("Expected result quality", "expected_result_quality"),
    ("Redundancy", "redundancy"),
    ("Procedure uniqueness", "procedure_uniqueness"),
    ("Scope drift", "scope_drift"),
    ("Traceability", "traceability"),
]


def _metric(evaluation, field_name: str) -> Optional[object]:
    return getattr(evaluation, field_name, None)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

st.set_page_config(page_title="AI Test Plan Comparator", layout="wide")

st.title("AI Test Plan Comparator")
st.markdown(
    """
Compare **Gemini**, **ChatGPT**, and/or **Rovo** generated test plans against a **stock** QA checklist.

- **Stock** (`AREA / FEATURE`, `TEST CASE`) — what must be tested. No STR or expected result.
- **Generated** (`TestID`, `Area/Feature`, `Test Case`, `Steps to Reproduce`, `Expected Result`) — scored.
- **GDD** (optional `.txt`) — the design doc used to generate cases. Adds source coverage/fidelity.

Matching uses **area + title**. Only generated plans are scored. Offline, no API keys.
"""
)

stock_col, gdd_col = st.columns(2)
with stock_col:
    stock_tp_file = st.file_uploader(
        "Stock test plan CSV (AREA / FEATURE + TEST CASE)",
        type=["csv"],
        key="stock_csv",
    )
with gdd_col:
    source_doc_file = st.file_uploader(
        "GDD / source spec (txt, md, or csv)",
        type=["csv", "txt", "md"],
        key="gdd_file",
        help="e.g. GDD_Mission1.txt — numbered mission beats used to generate the TPs.",
    )

st.subheader("Generated test plans (one CSV per LLM — upload any 1, 2, or all 3)")
llm_cols = st.columns(3)
llm_files = {}
for column, name in zip(llm_cols, LLM_SLOTS):
    with column:
        llm_files[name] = st.file_uploader(
            f"{name} CSV",
            type=["csv"],
            key=f"llm_{name.lower()}",
        )

selected_areas = []
if stock_tp_file:
    try:
        preview_stock = ingest_and_parse_test_plan_csv(stock_tp_file, "Stock")
        areas = unique_areas(preview_stock)
        st.caption(f"Stock parsed: {len(preview_stock.test_cases)} cases across {len(areas)} areas.")
        if areas:
            selected_areas = st.multiselect(
                "Stock areas to score against",
                options=areas,
                default=areas,
                help="Stock_mission1.1.csv includes Mission 1.1, 1.2, 1.3, and Misc. "
                "Deselect later missions if a generated file is 1.1-only.",
            )
    except Exception as parse_error:
        st.error(f"Could not parse stock CSV: {parse_error}")


def run_comparison(stock_file, llm_uploads, source_intents, areas) -> List[ModelEvaluation]:
    evaluations: List[ModelEvaluation] = []

    st.write("Parsing stock test plan...")
    stock_plan = ingest_and_parse_test_plan_csv(stock_file, "Stock")
    stock_plan = filter_plan_by_areas(stock_plan, areas)
    st.caption(f"Stock intents used: {len(stock_plan.test_cases)}")
    if source_intents:
        st.caption(f"GDD/source statements: {len(source_intents)}")

    for model_name, file in llm_uploads:
        st.write(f"Evaluating {model_name}...")
        try:
            candidate_plan = ingest_and_parse_test_plan_csv(file, model_name)
            evaluation = compare_and_evaluate_plan(
                stock_plan, candidate_plan, source_intents=source_intents or None
            )
            evaluations.append(evaluation)
        except Exception as e:
            logging.error("Failed to process %s: %s", model_name, e)
            st.error(f"An error occurred while processing {model_name}: {e}")
            evaluations.append(ModelEvaluation(model_name=model_name, error=str(e)))

    return evaluations


uploaded_llms = [(name, file) for name, file in llm_files.items() if file is not None]
run_clicked = st.button("Run comparison", use_container_width=True)

if run_clicked and (not stock_tp_file or not uploaded_llms):
    if not stock_tp_file:
        st.warning("Upload the stock test plan CSV.")
    if not uploaded_llms:
        st.warning("Upload at least one generated CSV (Gemini, ChatGPT, and/or Rovo).")
elif run_clicked:
    with st.spinner("Evaluating generated test plans..."):
        try:
            source_intents = merge_source_intents("", source_doc_file)
            evaluations = run_comparison(
                stock_tp_file, uploaded_llms, source_intents, selected_areas
            )
            valid_evaluations = [e for e in evaluations if not e.error]
            failed = [e for e in evaluations if e.error]

            if failed:
                for e in failed:
                    st.error(f"{e.model_name}: {e.error}")

            if not valid_evaluations:
                st.warning("No valid evaluations could be completed.")
            else:
                st.success("Comparison complete. Leaderboard ranks generated plans only.")
                st.header("Leaderboard")

                sorted_evals = sorted(
                    valid_evaluations,
                    key=lambda x: x.overall_score or 0.0,
                    reverse=True,
                )

                leaderboard_data = []
                for e in sorted_evals:
                    row = {"Model": e.model_name, "Overall": e.overall_score}
                    for label, field in METRIC_COLUMNS:
                        metric = _metric(e, field)
                        row[label] = metric.score if metric else None
                    leaderboard_data.append(row)
                st.dataframe(pd.DataFrame(leaderboard_data), use_container_width=True)
                st.caption(
                    "Overall is a weighted 1–5 blend of metrics that have values. "
                    "Source coverage/fidelity appear when you provide an overall intent or source document. "
                    "Redundancy, unique STR, scope drift, and source fidelity are inverse (5 = none of the problem)."
                )

                st.subheader("Diagnostics")
                diag_rows = []
                for e in sorted_evals:
                    row = {"Model": e.model_name}
                    row.update(getattr(e, "diagnostics", None) or {})
                    diag_rows.append(row)
                if any(getattr(e, "diagnostics", None) for e in sorted_evals):
                    st.dataframe(pd.DataFrame(diag_rows), use_container_width=True)

                st.header("Detailed analysis")
                for e in sorted_evals:
                    with st.expander(f"{e.model_name} (score: {e.overall_score})"):
                        st.subheader("Metric reasoning")
                        for label, field in METRIC_DETAILS:
                            metric = _metric(e, field)
                            if metric:
                                st.markdown(f"**{label} ({metric.score}/5)** — {metric.reasoning}")

                        if e.alignments:
                            st.subheader("Stock intent mapping")
                            align_df = pd.DataFrame(
                                [
                                    {
                                        "Stock ID": a.stock_id,
                                        "Stock title": a.stock_title,
                                        "Match": a.match_type,
                                        "Similarity %": getattr(a, "best_similarity", None),
                                        "Generated IDs": ", ".join(a.matched_candidate_ids) or "—",
                                        "Rationale": a.rationale,
                                    }
                                    for a in e.alignments
                                ]
                            )
                            st.dataframe(align_df, use_container_width=True)

                        if getattr(e, "source_alignments", None):
                            st.subheader("Source / overall intent mapping")
                            src_df = pd.DataFrame(
                                [
                                    {
                                        "Source ID": a.stock_id,
                                        "Source statement": a.stock_title,
                                        "Match": a.match_type,
                                        "Similarity %": getattr(a, "best_similarity", None),
                                        "Generated IDs": ", ".join(a.matched_candidate_ids) or "—",
                                        "Rationale": a.rationale,
                                    }
                                    for a in e.source_alignments
                                ]
                            )
                            st.dataframe(src_df, use_container_width=True)

                        if e.extra_in_scope_case_ids:
                            st.markdown(
                                "**In-scope extras (edge cases):** "
                                + ", ".join(e.extra_in_scope_case_ids)
                            )
                        if e.extra_out_of_scope_case_ids:
                            st.markdown(
                                "**Out of scope:** " + ", ".join(e.extra_out_of_scope_case_ids)
                            )
        except Exception as e:
            st.error(f"A critical error occurred: {e}")
            logging.error("Dashboard critical error: %s", e)
