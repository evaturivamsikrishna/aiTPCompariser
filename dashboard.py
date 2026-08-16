"""
dashboard.py

Compare AI-generated test-plan CSVs.
Stock TP is optional. GDD alone is enough as the spec. Two or more generated
plans are compared head-to-head even with no stock file.
"""

import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import altair as alt
import pandas as pd
import streamlit as st

# Streamlit keeps old project modules in memory across reruns. Drop them so
# every run imports the files currently on disk (avoids stale ImportError).
_PROJECT_ROOT = Path(__file__).resolve().parent
_LOCAL_MODULES = {path.stem for path in _PROJECT_ROOT.glob("*.py")} - {"dashboard"}
for _mod_name in list(sys.modules):
    if _mod_name.split(".")[0] in _LOCAL_MODULES:
        del sys.modules[_mod_name]

from auth import render_logout, require_login
from csv_parser import filter_plan_by_areas, ingest_and_parse_test_plan_csv, unique_areas
from evaluator import compare_and_evaluate_plan
from llm_judge import (
    DEFAULT_GROQ_MODEL,
    GROQ_MODELS,
    JudgeConfig,
    apply_llm_judgment,
    groq_key,
    groq_model_label,
    groq_tpm_limit,
    judge_plan,
    probe_ollama,
)
from peer_compare import compare_generated_plans
import run_store
from schemas import (
    ComparisonRun,
    ComparisonRunMeta,
    CrossPlanComparison,
    ModelEvaluation,
)
from source_parser import merge_source_intents

LLM_SLOTS = ("Gemini", "ChatGPT", "Rovo")

METRIC_COLUMNS = [
    ("Coverage", "requirement_coverage"),
    ("Match conf.", "match_confidence"),
    ("Area cov.", "area_coverage"),
    ("Source cov.", "source_coverage"),
    ("Source fit", "source_fidelity"),
    ("Peer cov.", "peer_coverage"),
    ("LLM GDD cov.", "llm_gdd_coverage"),
    ("LLM fidelity", "llm_gdd_fidelity"),
    ("LLM runnable", "llm_tester_readiness"),
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
    ("Peer coverage", "peer_coverage"),
    ("LLM GDD coverage", "llm_gdd_coverage"),
    ("LLM GDD fidelity", "llm_gdd_fidelity"),
    ("LLM tester readiness", "llm_tester_readiness"),
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


def _render_charts(sorted_evals: List) -> None:
    st.header("Charts")

    summary_cols = st.columns(min(3, len(sorted_evals)))
    for i, evaluation in enumerate(sorted_evals[:3]):
        delta = None
        if i > 0:
            leader = sorted_evals[0].overall_score or 0
            delta = round((evaluation.overall_score or 0) - leader, 2)
        summary_cols[i].metric(
            evaluation.model_name,
            f"{evaluation.overall_score}",
            delta=None if delta is None else f"{delta} vs leader",
        )

    overall_df = pd.DataFrame(
        [
            {"Model": e.model_name, "Overall": e.overall_score or 0}
            for e in sorted_evals
        ]
    )
    overall_chart = (
        alt.Chart(overall_df)
        .mark_bar()
        .encode(
            x=alt.X("Model:N", sort="-y", title="Model"),
            y=alt.Y("Overall:Q", title="Overall score", scale=alt.Scale(domain=[0, 5])),
            color=alt.Color("Model:N", legend=None),
            tooltip=["Model", "Overall"],
        )
        .properties(height=280, title="Overall score (1–5)")
    )
    st.altair_chart(overall_chart, use_container_width=True)

    metric_rows = []
    for evaluation in sorted_evals:
        for label, field in METRIC_COLUMNS:
            metric = _metric(evaluation, field)
            if metric:
                metric_rows.append(
                    {"Model": evaluation.model_name, "Metric": label, "Score": metric.score}
                )
    if metric_rows:
        metric_df = pd.DataFrame(metric_rows)
        grouped = (
            alt.Chart(metric_df)
            .mark_bar()
            .encode(
                x=alt.X("Metric:N", sort=None, title="Metric"),
                y=alt.Y("Score:Q", title="Score", scale=alt.Scale(domain=[0, 5])),
                color=alt.Color("Model:N", title="Model"),
                xOffset="Model:N",
                tooltip=["Model", "Metric", "Score"],
            )
            .properties(height=340, title="Side-by-side metric scores (1–5)")
        )
        st.altair_chart(grouped, use_container_width=True)

        heatmap = (
            alt.Chart(metric_df)
            .mark_rect()
            .encode(
                x=alt.X("Metric:N", title="Metric"),
                y=alt.Y("Model:N", title="Model"),
                color=alt.Color(
                    "Score:Q",
                    title="Score",
                    scale=alt.Scale(domain=[1, 5], scheme="blues"),
                ),
                tooltip=["Model", "Metric", "Score"],
            )
            .properties(height=max(90, 70 * len(sorted_evals)), title="Score heatmap")
        )
        labels = (
            alt.Chart(metric_df)
            .mark_text()
            .encode(x="Metric:N", y="Model:N", text="Score:Q")
        )
        st.altair_chart(heatmap + labels, use_container_width=True)

    match_rows = []
    for evaluation in sorted_evals:
        mapping = evaluation.alignments or evaluation.source_alignments or []
        for alignment in mapping:
            match_rows.append(
                {"Model": evaluation.model_name, "Match": alignment.match_type}
            )
    if match_rows:
        match_counts = (
            pd.DataFrame(match_rows)
            .groupby(["Model", "Match"])
            .size()
            .reset_index(name="Count")
        )
        stacked = (
            alt.Chart(match_counts)
            .mark_bar()
            .encode(
                x=alt.X("Model:N", title="Model"),
                y=alt.Y("Count:Q", title="Stock intents"),
                color=alt.Color(
                    "Match:N",
                    title="Match",
                    scale=alt.Scale(
                        domain=["semantic", "partial", "unmatched"],
                        range=["#2A9D8F", "#E9C46A", "#E76F51"],
                    ),
                ),
                tooltip=["Model", "Match", "Count"],
            )
            .properties(height=280, title="Stock intent match quality")
        )
        st.altair_chart(stacked, use_container_width=True)
        st.caption(
            "Semantic = strong title match. Partial = related. Unmatched = missing from the generated plan."
        )


def _cluster_table(clusters) -> pd.DataFrame:
    rows = []
    for cluster in clusters:
        models = sorted({m.model_name for m in cluster.members})
        ids = "; ".join(
            f"{m.model_name}:{m.case_id}" for m in cluster.members if m.case_id
        )
        rows.append(
            {
                "Intent": cluster.label,
                "Models": ", ".join(models),
                "Cases": ids,
            }
        )
    return pd.DataFrame(rows)


def _render_head_to_head(cross: CrossPlanComparison) -> None:
    st.header("Head-to-head (generated plans)")
    st.caption(
        "Distinct intents are clustered by title similarity across the uploaded CSVs. "
        "No stock file is required for this view."
    )
    if cross.cluster_count == 0:
        st.info("Not enough generated cases to cluster.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Distinct intents (union)", cross.cluster_count)
    c2.metric("Consensus (all models)", len(cross.consensus))
    unique_total = sum(len(v) for v in cross.unique_by_model.values())
    c3.metric("Unique to one model", unique_total)

    if cross.pairwise:
        st.subheader("Overlap between models")
        overlap_df = pd.DataFrame(
            [
                {
                    "Pair": f"{p.model_a} × {p.model_b}",
                    "Shared intents": p.shared,
                    "Union": p.union,
                    "Jaccard": p.jaccard,
                }
                for p in cross.pairwise
            ]
        )
        st.dataframe(overlap_df, use_container_width=True)
        chart = (
            alt.Chart(overlap_df)
            .mark_bar()
            .encode(
                x=alt.X("Pair:N", title="Pair"),
                y=alt.Y("Jaccard:Q", title="Jaccard overlap", scale=alt.Scale(domain=[0, 1])),
                tooltip=["Pair", "Shared intents", "Union", "Jaccard"],
            )
            .properties(height=240, title="How similar the generated plans are")
        )
        st.altair_chart(chart, use_container_width=True)

    if cross.consensus:
        with st.expander(f"Consensus cases ({len(cross.consensus)}) — every model has this intent"):
            st.dataframe(_cluster_table(cross.consensus), use_container_width=True)

    if cross.partial_overlap:
        with st.expander(f"Partial overlap ({len(cross.partial_overlap)}) — some models, not all"):
            st.dataframe(_cluster_table(cross.partial_overlap), use_container_width=True)

    for model_name, clusters in cross.unique_by_model.items():
        if clusters:
            with st.expander(f"Only in {model_name} ({len(clusters)})"):
                st.dataframe(_cluster_table(clusters), use_container_width=True)

    if cross.gdd_beats:
        st.subheader("GDD beats vs generated plans")
        gdd_df = pd.DataFrame(
            [
                {
                    "Beat": b.beat_text,
                    "Origin": b.origin,
                    "Match": b.match_type,
                    "Covered by": ", ".join(b.covered_by) or "—",
                }
                for b in cross.gdd_beats
            ]
        )
        st.dataframe(gdd_df, use_container_width=True)
        gaps = [b for b in cross.gdd_beats if not b.covered_by]
        if gaps:
            st.warning(
                f"{len(gaps)} GDD beats are not covered by any uploaded generated plan."
            )
        covered_counts = []
        model_names = sorted({m for b in cross.gdd_beats for m in b.covered_by})
        for name in model_names:
            hits = sum(1 for b in cross.gdd_beats if name in b.covered_by)
            covered_counts.append(
                {"Model": name, "GDD beats covered": hits, "Total beats": len(cross.gdd_beats)}
            )
        if covered_counts:
            cov_df = pd.DataFrame(covered_counts)
            gdd_chart = (
                alt.Chart(cov_df)
                .mark_bar()
                .encode(
                    x=alt.X("Model:N"),
                    y=alt.Y("GDD beats covered:Q"),
                    tooltip=["Model", "GDD beats covered", "Total beats"],
                )
                .properties(height=240, title="GDD beats covered by each generated plan")
            )
            st.altair_chart(gdd_chart, use_container_width=True)


def _upload_name(file_obj) -> Optional[str]:
    if file_obj is None:
        return None
    return getattr(file_obj, "name", None)


def _current_run() -> Optional[ComparisonRun]:
    raw = st.session_state.get("current_run")
    if not raw:
        return None
    try:
        return ComparisonRun.model_validate(raw)
    except Exception:
        return None


def _store_current_run(run: ComparisonRun) -> None:
    st.session_state["current_run"] = run.model_dump(mode="json")


def _build_run(
    evaluations: List[ModelEvaluation],
    cross: Optional[CrossPlanComparison],
    model_names: List[str],
    stock_file,
    source_file,
    selected_areas: List[str],
    selected_gdd_origins: List[str],
    judge_config: JudgeConfig,
) -> ComparisonRun:
    valid = [item for item in evaluations if not item.error]
    scores = {item.model_name: float(item.overall_score or 0) for item in valid}
    stamp = datetime.now(timezone.utc)
    models_label = ", ".join(model_names) or "Comparison"
    title = f"{models_label} · {stamp.strftime('%Y-%m-%d %H:%M UTC')}"
    return ComparisonRun(
        id=str(uuid.uuid4()),
        created_at=stamp.isoformat(),
        title=title,
        meta=ComparisonRunMeta(
            model_names=model_names,
            gdd_filename=_upload_name(source_file),
            stock_filename=_upload_name(stock_file),
            selected_areas=list(selected_areas or []),
            selected_gdd_origins=list(selected_gdd_origins or []),
            judge_backend=judge_config.backend,
            judge_model=judge_config.model or "",
            has_stock=bool(stock_file),
            has_gdd=bool(source_file),
            overall_scores=scores,
        ),
        evaluations=evaluations,
        cross=cross,
    )


def render_comparison_view(run: ComparisonRun) -> None:
    evaluations = run.evaluations
    cross = run.cross
    failed = [item for item in evaluations if item.error]
    valid_evaluations = [item for item in evaluations if not item.error]
    for item in failed:
        st.error(f"{item.model_name}: {item.error}")
    if not valid_evaluations:
        st.warning("No valid evaluations could be completed.")
        return

    st.success(f"{run.title}")
    if not run.meta.has_stock and run.meta.has_gdd:
        st.caption("Coverage is against the GDD/source document (no stock TP uploaded).")
    elif not run.meta.has_stock:
        st.caption(
            "No GDD or stock TP — ranking uses plan quality plus overlap with the other generated plans."
        )
    bits = []
    if run.meta.stock_filename:
        bits.append(f"Stock: {run.meta.stock_filename}")
    if run.meta.gdd_filename:
        bits.append(f"GDD: {run.meta.gdd_filename}")
    if run.meta.judge_backend != "off":
        bits.append(f"Judge: {run.meta.judge_backend} / {run.meta.judge_model}")
    if bits:
        st.caption(" · ".join(bits))

    st.subheader("Save this comparison")
    title = st.text_input("Entry title", value=run.title, key=f"save_title_{run.id}")
    if st.button("Save comparison", key=f"save_run_{run.id}"):
        run.title = (title or "").strip() or run.title
        try:
            run_store.save_run(run)
            _store_current_run(run)
            st.success(f'Saved “{run.title}”.')
        except Exception as exc:
            st.error(str(exc))

    st.header("Leaderboard")
    sorted_evals = sorted(
        valid_evaluations,
        key=lambda x: x.overall_score or 0.0,
        reverse=True,
    )
    leaderboard_data = []
    for evaluation in sorted_evals:
        row = {"Model": evaluation.model_name, "Overall": evaluation.overall_score}
        for label, field in METRIC_COLUMNS:
            metric = _metric(evaluation, field)
            row[label] = metric.score if metric else None
        leaderboard_data.append(row)
    st.dataframe(pd.DataFrame(leaderboard_data), use_container_width=True)
    st.caption(
        "Overall is a weighted 1–5 blend of metrics that have values. "
        "Stock coverage appears when you upload a stock CSV. "
        "GDD coverage appears when you upload a source document (and stock is also present). "
        "Without stock, GDD beats drive Coverage / Match / Area / Scope. "
        "Peer coverage appears when two or more generated plans are uploaded. "
        "LLM GDD coverage/fidelity/readiness appear when the sidebar judge is Ollama or Groq. "
        "Redundancy, unique STR, scope drift, and source fidelity are inverse (5 = none of the problem)."
    )

    _render_charts(sorted_evals)
    if cross and (cross.cluster_count or cross.gdd_beats):
        _render_head_to_head(cross)

    st.subheader("Diagnostics")
    diag_rows = []
    for evaluation in sorted_evals:
        row = {"Model": evaluation.model_name}
        row.update(getattr(evaluation, "diagnostics", None) or {})
        diag_rows.append(row)
    if any(getattr(evaluation, "diagnostics", None) for evaluation in sorted_evals):
        st.dataframe(pd.DataFrame(diag_rows), use_container_width=True)

    st.header("Detailed analysis")
    for evaluation in sorted_evals:
        with st.expander(f"{evaluation.model_name} (score: {evaluation.overall_score})"):
            st.subheader("Metric reasoning")
            for label, field in METRIC_DETAILS:
                metric = _metric(evaluation, field)
                if metric:
                    st.markdown(f"**{label} ({metric.score}/5)** — {metric.reasoning}")

            if evaluation.alignments:
                st.subheader("Stock intent mapping")
                align_df = pd.DataFrame(
                    [
                        {
                            "Stock ID": alignment.stock_id,
                            "Stock title": alignment.stock_title,
                            "Match": alignment.match_type,
                            "Similarity %": getattr(alignment, "best_similarity", None),
                            "Generated IDs": ", ".join(alignment.matched_candidate_ids) or "—",
                            "Rationale": alignment.rationale,
                        }
                        for alignment in evaluation.alignments
                    ]
                )
                st.dataframe(align_df, use_container_width=True)

            if getattr(evaluation, "llm_summary", None):
                st.subheader("LLM judge summary")
                st.write(evaluation.llm_summary)
            if getattr(evaluation, "llm_alignments", None):
                st.subheader("LLM design-beat mapping")
                llm_df = pd.DataFrame(
                    [
                        {
                            "Beat ID": alignment.stock_id,
                            "Beat": alignment.stock_title,
                            "Match": alignment.match_type,
                            "Generated IDs": ", ".join(alignment.matched_candidate_ids) or "—",
                            "Rationale": alignment.rationale,
                        }
                        for alignment in evaluation.llm_alignments
                    ]
                )
                st.dataframe(llm_df, use_container_width=True)

            if getattr(evaluation, "source_alignments", None):
                st.subheader("Source / overall intent mapping")
                src_df = pd.DataFrame(
                    [
                        {
                            "Source ID": alignment.stock_id,
                            "Source statement": alignment.stock_title,
                            "Match": alignment.match_type,
                            "Similarity %": getattr(alignment, "best_similarity", None),
                            "Generated IDs": ", ".join(alignment.matched_candidate_ids) or "—",
                            "Rationale": alignment.rationale,
                        }
                        for alignment in evaluation.source_alignments
                    ]
                )
                st.dataframe(src_df, use_container_width=True)

            if evaluation.extra_in_scope_case_ids:
                st.markdown(
                    "**In-scope extras (edge cases):** "
                    + ", ".join(evaluation.extra_in_scope_case_ids)
                )
            if evaluation.extra_out_of_scope_case_ids:
                st.markdown(
                    "**Out of scope:** " + ", ".join(evaluation.extra_out_of_scope_case_ids)
                )


def _render_history_sidebar() -> None:
    st.sidebar.header("Saved comparisons")
    if run_store.github_enabled():
        st.sidebar.caption(f"GitHub `{run_store.github_branch()}` · newest first.")
    else:
        st.sidebar.caption(
            "Stored on this machine. Add GITHUB_TOKEN + GITHUB_REPO in secrets "
            "to keep history on Streamlit Cloud."
        )
    try:
        summaries = run_store.list_runs()
    except Exception as exc:
        st.sidebar.error(f"Could not load history: {exc}")
        return
    if not summaries:
        st.sidebar.caption("No saved entries yet.")
        return
    for summary in summaries:
        score_bits = [
            f"{name} {score}"
            for name, score in (summary.overall_scores or {}).items()
        ]
        st.sidebar.markdown(f"**{summary.title}**")
        st.sidebar.caption(
            f"{summary.created_at[:16]} · {', '.join(score_bits) or 'no scores'}"
        )
        open_col, delete_col = st.sidebar.columns(2)
        if open_col.button("Open", key=f"open_{summary.id}", use_container_width=True):
            try:
                loaded = run_store.load_run(summary.id)
                _store_current_run(loaded)
                st.rerun()
            except Exception as exc:
                st.sidebar.error(str(exc))
        if delete_col.button("Delete", key=f"del_{summary.id}", use_container_width=True):
            st.session_state["pending_delete"] = summary.id
        if st.session_state.get("pending_delete") == summary.id:
            if st.sidebar.button(
                "Confirm delete",
                key=f"confirm_del_{summary.id}",
                use_container_width=True,
            ):
                try:
                    run_store.delete_run(summary.id)
                    current = _current_run()
                    if current and current.id == summary.id:
                        st.session_state.pop("current_run", None)
                    st.session_state.pop("pending_delete", None)
                    st.rerun()
                except Exception as exc:
                    st.sidebar.error(str(exc))


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

st.set_page_config(page_title="AI Test Plan Comparator", layout="wide")
require_login()

st.title("AI Test Plan Comparator")
st.markdown(
    """
Compare **Gemini**, **ChatGPT**, and/or **Rovo** generated test plans.

- **Generated CSVs** (required) — scored against each other and against any reference you provide.
- **GDD** (optional but recommended) — design doc used as the spec when you have no stock TP.
- **Stock** (optional) — QA checklist (`AREA / FEATURE`, `TEST CASE`). Used when you have one.
- **Judge** — fuzzy matching always runs. Optionally add a local Ollama model or Groq API for meaning-level scoring.

Matching uses **area + title**. Fuzzy mode needs no API keys.
"""
)

render_logout()
_render_history_sidebar()

with st.sidebar:
    st.header("Semantic judge (optional)")
    st.caption(
        "Fuzzy scoring always runs. An LLM judge checks whether cases actually "
        "cover GDD meaning. MCP is not required."
    )
    ollama_models = probe_ollama()
    ollama_names = [item.get("name") for item in ollama_models if item.get("name")]
    groq_default = groq_key()
    engine_options = ["Offline fuzzy only", "Local Ollama", "Groq API"]
    judge_choice = st.radio(
        "Engine",
        engine_options,
        index=2 if groq_default else 0,
    )
    judge_config = JudgeConfig(backend="off")
    if judge_choice == "Local Ollama":
        if not ollama_names:
            st.warning(
                "Ollama is not reachable at 127.0.0.1:11434. "
                "Start it, then pull a small model: `ollama pull llama3.2:3b`"
            )
        else:
            picked = st.selectbox(
                "Ollama model",
                ollama_names,
                index=0,
                help="Smaller models are faster. Your machine already has larger coder models too.",
            )
            judge_config = JudgeConfig(backend="ollama", model=picked, base_url="http://127.0.0.1:11434")
            st.caption(f"Using local {picked}. First run may be slow while the model loads.")
    elif judge_choice == "Groq API":
        if groq_default:
            st.success("GROQ_API_KEY loaded from `.env` / environment.")
            groq_input = groq_default
        else:
            groq_input = st.text_input(
                "GROQ_API_KEY",
                value="",
                type="password",
                help="Create a key at console.groq.com/keys, then put GROQ_API_KEY=... in `.env`.",
            )
        groq_model = st.selectbox(
            "Groq model",
            list(GROQ_MODELS),
            index=0,
            format_func=groq_model_label,
            help="70B has 12k TPM on your Developer plan. 8B is faster but only 6k TPM.",
        )
        st.caption(
            f"Packing to {groq_tpm_limit(groq_model) // 1000}k tokens/min for this model. "
            "The judge may pause between batches if the minute window is full."
        )
        if groq_input:
            judge_config = JudgeConfig(
                backend="groq",
                model=groq_model.strip() or DEFAULT_GROQ_MODEL,
                api_key=groq_input.strip(),
            )
        else:
            st.warning("Set GROQ_API_KEY in `.env` or paste a key to enable the LLM judge.")

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

selected_gdd_origins = []
if source_doc_file:
    try:
        preview_intents = merge_source_intents("", source_doc_file)
        origins = []
        for intent in preview_intents:
            if intent.origin not in origins:
                origins.append(intent.origin)
        st.caption(f"GDD/source parsed: {len(preview_intents)} beats across {len(origins)} sections.")
        if origins:
            selected_gdd_origins = st.multiselect(
                "GDD sections to score against",
                options=origins,
                default=origins,
                help="Deselect later missions if generated plans are 1.1-only.",
            )
    except Exception as parse_error:
        st.error(f"Could not parse GDD/source: {parse_error}")


def run_comparison(stock_file, llm_uploads, source_intents, areas, judge_config: JudgeConfig):
    evaluations: List[ModelEvaluation] = []
    candidate_plans = []

    stock_plan = None
    if stock_file:
        st.write("Parsing stock test plan...")
        stock_plan = ingest_and_parse_test_plan_csv(stock_file, "Stock")
        stock_plan = filter_plan_by_areas(stock_plan, areas)
        st.caption(f"Stock intents used: {len(stock_plan.test_cases)}")
    else:
        st.caption("No stock TP — scoring against GDD and/or the other generated plans.")
    if source_intents:
        st.caption(f"GDD/source statements: {len(source_intents)}")

    for model_name, file in llm_uploads:
        st.write(f"Parsing {model_name}...")
        try:
            candidate_plans.append(ingest_and_parse_test_plan_csv(file, model_name))
        except Exception as e:
            logging.error("Failed to parse %s: %s", model_name, e)
            st.error(f"An error occurred while processing {model_name}: {e}")
            evaluations.append(ModelEvaluation(model_name=model_name, error=str(e)))

    cross = compare_generated_plans(candidate_plans, source_intents=source_intents or None)
    stock_cases = stock_plan.test_cases if stock_plan else None

    for plan in candidate_plans:
        st.write(f"Evaluating {plan.model_name}...")
        try:
            evaluation = compare_and_evaluate_plan(
                stock_plan,
                plan,
                source_intents=source_intents or None,
                peer_coverage_ratio=cross.model_peer_ratios.get(plan.model_name),
                peer_coverage_note=cross.model_peer_notes.get(plan.model_name),
            )
            if judge_config.backend != "off":
                st.write(
                    f"LLM judge ({judge_config.backend} / {judge_config.model}) "
                    f"for {plan.model_name}..."
                )
                try:
                    llm_result = judge_plan(
                        plan,
                        source_intents or None,
                        stock_cases,
                        judge_config,
                    )
                    evaluation = apply_llm_judgment(evaluation, llm_result)
                except Exception as llm_error:
                    logging.error("LLM judge failed for %s: %s", plan.model_name, llm_error)
                    st.warning(
                        f"LLM judge failed for {plan.model_name}: {llm_error}. "
                        "Fuzzy scores are still shown."
                    )
            evaluations.append(evaluation)
        except Exception as e:
            logging.error("Failed to evaluate %s: %s", plan.model_name, e)
            st.error(f"An error occurred while processing {plan.model_name}: {e}")
            evaluations.append(ModelEvaluation(model_name=plan.model_name, error=str(e)))

    return evaluations, cross


uploaded_llms = [(name, file) for name, file in llm_files.items() if file is not None]
run_clicked = st.button("Run comparison", use_container_width=True)
has_reference = bool(stock_tp_file or source_doc_file or len(uploaded_llms) >= 2)

if run_clicked and not uploaded_llms:
    st.warning("Upload at least one generated CSV (Gemini, ChatGPT, and/or Rovo).")
elif run_clicked and not has_reference:
    st.warning(
        "Upload a GDD, a stock test plan, or at least two generated CSVs so there is something to compare against."
    )
elif run_clicked:
    with st.spinner("Evaluating generated test plans..."):
        try:
            source_intents = merge_source_intents("", source_doc_file)
            if source_doc_file and selected_gdd_origins:
                source_intents = [
                    intent for intent in source_intents if intent.origin in selected_gdd_origins
                ]
            elif source_doc_file and not selected_gdd_origins:
                source_intents = []
            evaluations, cross = run_comparison(
                stock_tp_file, uploaded_llms, source_intents, selected_areas, judge_config
            )
            run = _build_run(
                evaluations,
                cross,
                [name for name, _ in uploaded_llms],
                stock_tp_file,
                source_doc_file,
                selected_areas,
                selected_gdd_origins,
                judge_config,
            )
            _store_current_run(run)
        except Exception as e:
            st.error(f"A critical error occurred: {e}")
            logging.error("Dashboard critical error: %s", e)

displayed = _current_run()
if displayed:
    render_comparison_view(displayed)
