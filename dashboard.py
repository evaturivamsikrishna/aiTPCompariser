"""
dashboard.py

Compare AI-generated test-plan CSVs.
Stock plan is optional. GDD alone is enough as the spec. Two or more generated
plans are compared head-to-head even with no stock file.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional

import altair as alt
import pandas as pd
import streamlit as st

from auth import render_logout, require_login
from csv_parser import filter_plan_by_areas, ingest_and_parse_test_plan_csv, unique_areas
from evaluator import compare_and_evaluate_plan
from llm_judge import (
    JudgeConfig,
    apply_llm_judgment,
    groq_key,
    judge_plan,
    probe_ollama,
)
import llm_judge as _llm_judge
from peer_compare import compare_generated_plans
import run_store
from schemas import (
    ComparisonRun,
    ComparisonRunMeta,
    CrossPlanComparison,
    ModelEvaluation,
)
from source_parser import merge_source_intents

PLAN_SLOTS = (
    ("plan_1", "Gemini"),
    ("plan_2", "ChatGPT"),
    ("plan_3", "Rovo"),
)

DEFAULT_GROQ_MODEL = getattr(_llm_judge, "DEFAULT_GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_MODELS = getattr(
    _llm_judge,
    "GROQ_MODELS",
    ("llama-3.3-70b-versatile", "llama-3.1-8b-instant"),
)
_GROQ_TPM = getattr(
    _llm_judge,
    "GROQ_MODEL_TPM",
    {"llama-3.3-70b-versatile": 12_000, "llama-3.1-8b-instant": 6_000},
)

INVERSE_METRIC_FIELDS = {
    "source_fidelity",
    "redundancy",
    "procedure_uniqueness",
    "scope_drift",
}

_METRIC_DEFS = [
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


def _groq_tpm_limit(model: str) -> int:
    limiter = getattr(_llm_judge, "groq_tpm_limit", None)
    if callable(limiter):
        return int(limiter(model))
    return int(_GROQ_TPM.get(model or DEFAULT_GROQ_MODEL, 6_000))


def _groq_model_label(model: str) -> str:
    tpm = _groq_tpm_limit(model)
    suffix = f"{tpm // 1000}k TPM"
    if model == DEFAULT_GROQ_MODEL:
        suffix = f"{suffix}, recommended"
    return f"{model}  ·  {suffix}"


def _metric_label(name: str, field: str) -> str:
    return f"{name} (inv)" if field in INVERSE_METRIC_FIELDS else name


METRIC_COLUMNS = [(_metric_label(name, field), field) for name, field in _METRIC_DEFS]

METRIC_FAMILIES = {
    "Coverage": {
        "requirement_coverage",
        "match_confidence",
        "area_coverage",
        "source_coverage",
        "peer_coverage",
        "llm_gdd_coverage",
    },
    "Quality": {
        "edge_case_quality",
        "actionability",
        "str_completeness",
        "expected_result_quality",
        "llm_tester_readiness",
        "traceability",
    },
    "Inverse": {
        "source_fidelity",
        "llm_gdd_fidelity",
        "redundancy",
        "procedure_uniqueness",
        "scope_drift",
    },
}

MODEL_PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
MATCH_DOMAIN = ["semantic", "partial", "unmatched"]
MATCH_RANGE = ["#2A9D8F", "#E9C46A", "#E76F51"]
SCORE_DOMAIN = [0, 5]


def _metric(evaluation, field_name: str) -> Optional[object]:
    return getattr(evaluation, field_name, None)


def _model_color_scale(model_names: List[str]) -> alt.Scale:
    names = list(model_names)
    colors = [MODEL_PALETTE[i % len(MODEL_PALETTE)] for i in range(len(names))]
    return alt.Scale(domain=names, range=colors)


def _format_ts(iso_str: str) -> str:
    if not iso_str:
        return ""
    try:
        normalized = iso_str.replace("Z", "+00:00")
        stamp = datetime.fromisoformat(normalized)
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return iso_str


def _format_score(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


def _relative_time(iso_str: str) -> str:
    if not iso_str:
        return ""
    try:
        normalized = iso_str.replace("Z", "+00:00")
        stamp = datetime.fromisoformat(normalized)
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        seconds = int((datetime.now(timezone.utc) - stamp.astimezone(timezone.utc)).total_seconds())
        if seconds < 60:
            return "just now"
        if seconds < 3600:
            return f"{seconds // 60}m ago"
        if seconds < 86400:
            return f"{seconds // 3600}h ago"
        return f"{seconds // 86400}d ago"
    except Exception:
        return _format_ts(iso_str)


def _rewind(file_obj) -> None:
    if file_obj is not None and hasattr(file_obj, "seek"):
        try:
            file_obj.seek(0)
        except Exception:
            pass


def _preview_cases(plan, heading: str) -> None:
    rows = [
        {
            "ID": case.id or "",
            "Area": case.area or "",
            "Case": case.title,
        }
        for case in plan.test_cases[:5]
    ]
    if not rows:
        return
    st.caption(f"{heading} (first {len(rows)} of {len(plan.test_cases)})")
    _show_dataframe(pd.DataFrame(rows))


def _prefer_mission_11(options: List[str]) -> List[str]:
    matched = [item for item in options if "1.1" in item]
    return matched or list(options)


def _fail(step: str, exc: Exception, name: Optional[str] = None) -> None:
    if name:
        st.error(f"{step} failed for {name}: {exc}")
    else:
        st.error(f"{step} failed: {exc}")


def _show_dataframe(df: pd.DataFrame, column_config=None) -> None:
    kwargs = {}
    if column_config:
        kwargs["column_config"] = column_config
    try:
        st.dataframe(df, hide_index=True, width="stretch", **kwargs)
    except TypeError:
        try:
            st.dataframe(df, hide_index=True, use_container_width=True, **kwargs)
        except TypeError:
            st.dataframe(df, use_container_width=True, **kwargs)


def _score_column_config(columns) -> dict:
    config = {}
    for col in columns:
        if col in {"Model", "Generated plan"}:
            continue
        config[col] = st.column_config.NumberColumn(col, format="%.2f")
    return config


def _render_charts(sorted_evals: List) -> None:
    st.header("Charts")
    model_names = [e.model_name for e in sorted_evals]
    model_scale = _model_color_scale(model_names)

    n = max(1, len(sorted_evals))
    summary_cols = st.columns(n)
    leader = sorted_evals[0].overall_score or 0
    for i, evaluation in enumerate(sorted_evals):
        delta = None
        if i > 0:
            delta = round((evaluation.overall_score or 0) - leader, 2)
        summary_cols[i].metric(
            evaluation.model_name,
            f"{_format_score(evaluation.overall_score)}",
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
            x=alt.X("Model:N", sort="-y", title="Generated plan"),
            y=alt.Y(
                "Overall:Q",
                title="Overall score",
                scale=alt.Scale(domain=SCORE_DOMAIN),
            ),
            color=alt.Color("Model:N", scale=model_scale, legend=None),
            tooltip=["Model", "Overall"],
        )
        .properties(height=280, title="Overall score (1–5)")
    )
    st.altair_chart(overall_chart, width="stretch")

    metric_rows = []
    for evaluation in sorted_evals:
        for label, field in METRIC_COLUMNS:
            metric = _metric(evaluation, field)
            if metric:
                metric_rows.append(
                    {
                        "Model": evaluation.model_name,
                        "Metric": label,
                        "Score": metric.score,
                        "Field": field,
                    }
                )
    if metric_rows:
        family = st.radio(
            "Metric family",
            ["Coverage", "Quality", "Inverse", "All"],
            horizontal=True,
            key="chart_metric_family",
        )
        if family != "All":
            allowed = METRIC_FAMILIES[family]
            metric_rows = [row for row in metric_rows if row["Field"] in allowed]
        if not metric_rows:
            st.caption(f"No {family.lower()} metric scores to plot.")
        else:
            metric_df = pd.DataFrame(metric_rows)
            metric_axis = alt.Axis(labelAngle=-40, labelLimit=180)
            grouped = (
                alt.Chart(metric_df)
                .mark_bar()
                .encode(
                    x=alt.X("Metric:N", sort=None, title="Metric", axis=metric_axis),
                    y=alt.Y("Score:Q", title="Score (1–5)", scale=alt.Scale(domain=SCORE_DOMAIN)),
                    color=alt.Color("Model:N", title="Generated plan", scale=model_scale),
                    xOffset="Model:N",
                    tooltip=["Model", "Metric", "Score"],
                )
                .properties(height=340, title=f"{family} metric scores (1–5)")
            )
            st.altair_chart(grouped, width="stretch")

            heatmap = (
                alt.Chart(metric_df)
                .mark_rect()
                .encode(
                    x=alt.X("Metric:N", title="Metric", axis=metric_axis),
                    y=alt.Y("Model:N", title="Generated plan"),
                    color=alt.Color(
                        "Score:Q",
                        title="Score",
                        scale=alt.Scale(domain=SCORE_DOMAIN, scheme="blues"),
                    ),
                    tooltip=["Model", "Metric", "Score"],
                )
                .properties(height=max(90, 70 * len(sorted_evals)), title=f"{family} score heatmap")
            )
            labels = (
                alt.Chart(metric_df)
                .mark_text()
                .encode(x="Metric:N", y="Model:N", text="Score:Q")
            )
            st.altair_chart(heatmap + labels, width="stretch")
    else:
        st.caption("No metric scores to plot.")

    match_rows = []
    mapping_kinds = set()
    for evaluation in sorted_evals:
        if evaluation.alignments:
            mapping = evaluation.alignments
            mapping_kinds.add("stock")
        elif evaluation.source_alignments:
            mapping = evaluation.source_alignments
            mapping_kinds.add("gdd")
        else:
            mapping = []
        for alignment in mapping:
            match_rows.append(
                {"Model": evaluation.model_name, "Match": alignment.match_type}
            )
    if match_rows:
        if mapping_kinds == {"stock"}:
            y_title = "Stock cases"
            chart_title = "Stock case match quality"
        elif mapping_kinds == {"gdd"}:
            y_title = "GDD beats"
            chart_title = "GDD beat match quality"
        else:
            y_title = "Items"
            chart_title = "Match quality"
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
                x=alt.X("Model:N", title="Generated plan"),
                y=alt.Y("Count:Q", title=y_title),
                color=alt.Color(
                    "Match:N",
                    title="Match",
                    scale=alt.Scale(domain=MATCH_DOMAIN, range=MATCH_RANGE),
                ),
                tooltip=["Model", "Match", "Count"],
            )
            .properties(height=280, title=chart_title)
        )
        st.altair_chart(stacked, width="stretch")
        st.caption(
            "Semantic = strong title match. Partial = related. Unmatched = missing from the generated plan."
        )
    else:
        st.caption("No stock or GDD alignments to plot.")


def _cluster_table(clusters) -> pd.DataFrame:
    rows = []
    for cluster in clusters:
        models = sorted({m.model_name for m in cluster.members})
        ids = "; ".join(
            f"{m.model_name}:{m.case_id}" for m in cluster.members if m.case_id
        )
        rows.append(
            {
                "Case": cluster.label,
                "Models": ", ".join(models),
                "Cases": ids,
            }
        )
    return pd.DataFrame(rows)


def _render_head_to_head(cross: CrossPlanComparison) -> None:
    st.header("Head-to-head (generated plans)")
    st.caption(
        "Distinct cases are clustered by title similarity across the uploaded generated plans. "
        "No stock plan is required for this view."
    )
    if cross.cluster_count == 0:
        st.info("Not enough generated cases to cluster.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Distinct cases", cross.cluster_count)
        c2.metric("Shared by every plan", len(cross.consensus))
        unique_total = sum(len(v) for v in cross.unique_by_model.values())
        c3.metric("Only in one plan", unique_total)
        st.caption(
            "Distinct cases = union across generated plans. "
            "Shared by every plan = in all uploads. "
            "Only in one plan = unique to a single upload."
        )

        if cross.pairwise:
            st.subheader("Overlap between generated plans")
            overlap_df = pd.DataFrame(
                [
                    {
                        "Pair": f"{p.model_a} × {p.model_b}",
                        "Shared cases": p.shared,
                        "Union": p.union,
                        "Jaccard %": round((p.jaccard or 0) * 100, 1),
                    }
                    for p in cross.pairwise
                ]
            )
            _show_dataframe(
                overlap_df,
                column_config={
                    "Shared cases": st.column_config.NumberColumn(format="%d"),
                    "Union": st.column_config.NumberColumn(format="%d"),
                    "Jaccard %": st.column_config.NumberColumn(
                        "Jaccard %",
                        help="Shared cases ÷ union × 100. 100 means the two generated plans cover the same cases.",
                        format="%.1f",
                    ),
                },
            )
            chart = (
                alt.Chart(overlap_df)
                .mark_bar()
                .encode(
                    x=alt.X("Pair:N", title="Pair"),
                    y=alt.Y("Jaccard %:Q", title="Jaccard overlap (%)", scale=alt.Scale(domain=[0, 100])),
                    tooltip=["Pair", "Shared cases", "Union", "Jaccard %"],
                )
                .properties(height=240, title="Jaccard overlap between generated plans")
            )
            st.altair_chart(chart, width="stretch")
            st.caption("Jaccard % = shared cases ÷ union × 100. 100 means the two generated plans cover the same cases.")

        if cross.consensus:
            with st.expander(f"Shared by every plan · {len(cross.consensus)}"):
                _show_dataframe(_cluster_table(cross.consensus))

        if cross.partial_overlap:
            with st.expander(f"In some plans · {len(cross.partial_overlap)}"):
                _show_dataframe(_cluster_table(cross.partial_overlap))

        for model_name, clusters in cross.unique_by_model.items():
            if clusters:
                with st.expander(f"Only in {model_name} · {len(clusters)}"):
                    _show_dataframe(_cluster_table(clusters))

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
        _show_dataframe(gdd_df)
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
                    x=alt.X("Model:N", title="Generated plan"),
                    y=alt.Y("GDD beats covered:Q", title="GDD beats covered"),
                    color=alt.Color(
                        "Model:N",
                        legend=None,
                        scale=_model_color_scale(model_names),
                    ),
                    tooltip=["Model", "GDD beats covered", "Total beats"],
                )
                .properties(height=240, title="GDD beats covered by each generated plan")
            )
            st.altair_chart(gdd_chart, width="stretch")


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


def _render_setup_inputs():
    """File slots and filters. Call every rerun so keyed uploaders keep their files."""
    st.subheader("Generated plans (required)")
    st.caption("Upload one CSV per generated plan. Name the slot whatever produced the file.")
    llm_cols = st.columns(3)
    llm_files = {}
    used_names = set()
    for column, (slot_id, default_name) in zip(llm_cols, PLAN_SLOTS):
        with column:
            plan_name = st.text_input(
                "Plan name",
                value=default_name,
                key=f"llm_name_{slot_id}",
            )
            uploaded = st.file_uploader(
                "Generated plan CSV",
                type=["csv"],
                key=f"llm_file_{slot_id}",
                help="Example: AiTestPlanGenerator_Mission 1.1.csv",
            )
            display = (plan_name or "").strip() or default_name
            if display in used_names:
                display = f"{display} ({slot_id})"
            if uploaded is not None:
                used_names.add(display)
                llm_files[display] = uploaded

    for display, uploaded in llm_files.items():
        try:
            preview_plan = ingest_and_parse_test_plan_csv(uploaded, display)
            _preview_cases(preview_plan, f"{display} parsed")
        except Exception as parse_error:
            _fail("Parse", parse_error, display)
        finally:
            _rewind(uploaded)

    st.subheader("Reference (optional)")
    st.caption("GDD is the spec when you have no stock plan. Stock plan is an existing QA checklist.")
    gdd_col, stock_col = st.columns(2)
    with gdd_col:
        source_doc_file = st.file_uploader(
            "GDD (txt, md, or csv)",
            type=["csv", "txt", "md"],
            key="gdd_file",
            help="Example: GDD_Mission1.txt — numbered beats used to generate the plans.",
        )
    with stock_col:
        stock_tp_file = st.file_uploader(
            "Stock plan CSV",
            type=["csv"],
            key="stock_csv",
            help="Example: Stock_mission1.1.csv — AREA / FEATURE and TEST CASE columns.",
        )

    selected_areas = []
    if stock_tp_file:
        try:
            preview_stock = ingest_and_parse_test_plan_csv(stock_tp_file, "Stock")
            areas = unique_areas(preview_stock)
            st.caption(
                f"Stock plan parsed: {len(preview_stock.test_cases)} cases across {len(areas)} areas."
            )
            _preview_cases(preview_stock, "Stock plan preview")
            _rewind(stock_tp_file)
            if areas:
                selected_areas = st.multiselect(
                    "Stock plan areas to score against",
                    options=areas,
                    default=_prefer_mission_11(areas),
                    key=f"stock_areas_{_upload_name(stock_tp_file)}",
                    help="Defaults to Mission 1.1 rows when those labels exist.",
                )
                st.caption(
                    f"Scoring against {len(selected_areas)} of {len(areas)} stock plan areas."
                )
        except Exception as parse_error:
            _fail("Parse", parse_error, "stock plan")
            _rewind(stock_tp_file)

    selected_gdd_origins = []
    gdd_origin_options: List[str] = []
    if source_doc_file:
        try:
            preview_intents = merge_source_intents("", source_doc_file)
            origins = []
            for intent in preview_intents:
                if intent.origin not in origins:
                    origins.append(intent.origin)
            gdd_origin_options = origins
            st.caption(
                f"GDD parsed: {len(preview_intents)} beats across {len(origins)} sections."
            )
            beat_preview = [
                {
                    "Section": intent.origin,
                    "Beat": intent.text if len(intent.text) <= 160 else intent.text[:157] + "…",
                }
                for intent in preview_intents[:5]
            ]
            if beat_preview:
                st.caption(f"GDD preview (first {len(beat_preview)} of {len(preview_intents)})")
                _show_dataframe(pd.DataFrame(beat_preview))
            _rewind(source_doc_file)
            if origins:
                selected_gdd_origins = st.multiselect(
                    "GDD sections to score against",
                    options=origins,
                    default=_prefer_mission_11(origins),
                    key=f"gdd_sections_{_upload_name(source_doc_file)}",
                    help="Defaults to Mission 1.1 sections when those labels exist.",
                )
                st.caption(
                    f"Scoring against {len(selected_gdd_origins)} of {len(origins)} GDD sections."
                )
        except Exception as parse_error:
            _fail("Parse", parse_error, "GDD")
            _rewind(source_doc_file)

    return (
        stock_tp_file,
        source_doc_file,
        llm_files,
        selected_areas,
        selected_gdd_origins,
        gdd_origin_options,
    )


def _queue_app_view(view: str) -> None:
    """Switch Setup/Results on the next rerun (cannot write the radio key after it exists)."""
    st.session_state["pending_app_view"] = view


def _render_results_banner() -> None:
    origin = st.session_state.get("run_origin")
    if origin == "history":
        st.info(
            "Opened from **Saved comparisons**. Scores on this page are from that "
            "saved entry, not from the files currently in Setup."
        )
    elif origin == "live":
        st.caption(
            "Results from the last Run. Changing files in Setup does not update this view until you Run again."
        )


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
        st.error(f"Evaluate failed for {item.model_name}: {item.error}")
    if not valid_evaluations:
        st.warning("No valid evaluations could be completed.")
        return

    title_col, name_col, save_col = st.columns([2.2, 2.2, 1])
    title_col.header(run.title)
    if not run.meta.has_stock and run.meta.has_gdd:
        st.caption("Coverage is against the GDD (no stock plan uploaded).")
    elif not run.meta.has_stock:
        st.caption(
            "No GDD or stock plan — ranking uses plan quality plus overlap with the other generated plans."
        )
    bits = []
    if run.meta.stock_filename:
        bits.append(f"Stock plan: {run.meta.stock_filename}")
    if run.meta.gdd_filename:
        bits.append(f"GDD: {run.meta.gdd_filename}")
    if run.meta.judge_backend != "off":
        bits.append(f"Judge: {run.meta.judge_backend} / {run.meta.judge_model}")
    if bits:
        st.caption(" · ".join(bits))

    save_title = name_col.text_input(
        "Comparison name",
        value=run.title,
        key=f"save_title_{run.id}",
        label_visibility="collapsed",
        placeholder="Comparison name",
    )
    if save_col.button("Save", type="secondary", key=f"save_run_{run.id}"):
        run.title = (save_title or "").strip() or run.title
        try:
            run_store.save_run(run)
            _store_current_run(run)
            st.success(f'Saved “{run.title}”.')
        except Exception as exc:
            _fail("Save", exc)

    st.header("Leaderboard")
    sorted_evals = sorted(
        valid_evaluations,
        key=lambda x: x.overall_score or 0.0,
        reverse=True,
    )
    leaderboard_data = []
    for evaluation in sorted_evals:
        row = {
            "Generated plan": evaluation.model_name,
            "Overall": _format_score(evaluation.overall_score),
        }
        for label, field in METRIC_COLUMNS:
            metric = _metric(evaluation, field)
            row[label] = _format_score(metric.score) if metric else None
        leaderboard_data.append(row)
    leaderboard_df = pd.DataFrame(leaderboard_data)
    _show_dataframe(leaderboard_df, _score_column_config(leaderboard_df.columns))
    with st.expander("How scoring works"):
        st.markdown(
            """
- **Overall** is a weighted 1–5 blend of metrics that have values.
- **Stock plan** coverage appears when you upload a stock plan CSV.
- **GDD** coverage appears from GDD beats (and from GDD+stock when both are present).
- **Peer coverage** appears when two or more generated plans are uploaded.
- **LLM GDD** metrics appear when the sidebar judge is Ollama or Groq.
- Columns marked **(inv)** are inverse: 5 means none of the problem.
- Matching uses **area + title**. Fuzzy mode needs no API keys.
"""
        )

    _render_charts(sorted_evals)
    if cross and (cross.cluster_count or cross.gdd_beats):
        _render_head_to_head(cross)

    if any(getattr(evaluation, "diagnostics", None) for evaluation in sorted_evals):
        st.subheader("Diagnostics")
        diag_rows = []
        for evaluation in sorted_evals:
            row = {"Generated plan": evaluation.model_name}
            row.update(getattr(evaluation, "diagnostics", None) or {})
            diag_rows.append(row)
        _show_dataframe(pd.DataFrame(diag_rows))

    st.header("Detailed analysis")
    for evaluation in sorted_evals:
        score = _format_score(evaluation.overall_score)
        with st.expander(f"{evaluation.model_name} · score {score}"):
            st.markdown("**Metric reasoning**")
            for label, field in METRIC_COLUMNS:
                metric = _metric(evaluation, field)
                if metric:
                    st.markdown(f"**{label} ({metric.score}/5)** — {metric.reasoning}")

            if evaluation.alignments:
                st.markdown("**Stock plan mapping**")
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
                _show_dataframe(align_df)

            if getattr(evaluation, "llm_summary", None):
                st.markdown("**Judge summary**")
                st.write(evaluation.llm_summary)
            if getattr(evaluation, "llm_alignments", None):
                st.markdown("**Judge GDD beat mapping**")
                llm_df = pd.DataFrame(
                    [
                        {
                            "Beat ID": alignment.stock_id,
                            "Beat": alignment.stock_title,
                            "Match": alignment.match_type,
                            "Similarity %": getattr(alignment, "best_similarity", None),
                            "Generated IDs": ", ".join(alignment.matched_candidate_ids) or "—",
                            "Rationale": alignment.rationale,
                        }
                        for alignment in evaluation.llm_alignments
                    ]
                )
                _show_dataframe(llm_df)

            if getattr(evaluation, "source_alignments", None):
                st.markdown("**GDD beat mapping**")
                src_df = pd.DataFrame(
                    [
                        {
                            "Beat ID": alignment.stock_id,
                            "Beat": alignment.stock_title,
                            "Match": alignment.match_type,
                            "Similarity %": getattr(alignment, "best_similarity", None),
                            "Generated IDs": ", ".join(alignment.matched_candidate_ids) or "—",
                            "Rationale": alignment.rationale,
                        }
                        for alignment in evaluation.source_alignments
                    ]
                )
                _show_dataframe(src_df)

            extra_rows = [
                {"Kind": "In-scope extra", "Case ID": case_id}
                for case_id in (evaluation.extra_in_scope_case_ids or [])
            ] + [
                {"Kind": "Out of scope", "Case ID": case_id}
                for case_id in (evaluation.extra_out_of_scope_case_ids or [])
            ]
            if extra_rows:
                st.markdown("**Extra cases**")
                _show_dataframe(pd.DataFrame(extra_rows))


def _render_history_sidebar() -> None:
    st.sidebar.header("Saved comparisons")
    if run_store.github_enabled():
        st.sidebar.caption(f"GitHub `{run_store.github_branch()}` · newest first")
    else:
        st.sidebar.caption("This machine · newest first")
    try:
        summaries = run_store.list_runs()
    except Exception as exc:
        st.sidebar.error(f"Could not load history: {exc}")
        return
    if not summaries:
        st.sidebar.caption("No saved entries yet.")
        return

    query = st.sidebar.text_input(
        "Search saved comparisons",
        key="history_search",
        placeholder="Search",
        label_visibility="collapsed",
    )
    needle = (query or "").strip().lower()
    filtered = [
        summary
        for summary in summaries
        if not needle or needle in (summary.title or "").lower()
    ]
    if not filtered:
        st.sidebar.caption("No matches.")
        return

    st.session_state.setdefault("history_shown", 5)
    shown = st.session_state["history_shown"]
    for summary in filtered[:shown]:
        scores = summary.overall_scores or {}
        leader = max(scores.items(), key=lambda item: item[1]) if scores else None
        leader_txt = f"{leader[0]} {leader[1]:.2f}" if leader else "no scores"
        title = summary.title or summary.id
        short = title if len(title) <= 42 else title[:39] + "…"
        with st.sidebar.expander(short):
            st.caption(f"{_relative_time(summary.created_at)} · {leader_txt}")
            open_col, delete_col = st.columns(2)
            if open_col.button("Open", key=f"open_{summary.id}", type="secondary"):
                try:
                    loaded = run_store.load_run(summary.id)
                    _store_current_run(loaded)
                    st.session_state["run_origin"] = "history"
                    _queue_app_view("Results")
                    st.rerun()
                except Exception as exc:
                    st.sidebar.error(str(exc))
            if delete_col.button("Delete", key=f"del_{summary.id}", type="secondary"):
                st.session_state["pending_delete"] = summary.id
            if st.session_state.get("pending_delete") == summary.id:
                confirm_col, cancel_col = st.columns(2)
                if confirm_col.button(
                    "Confirm",
                    key=f"confirm_del_{summary.id}",
                    type="primary",
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
                if cancel_col.button("Cancel", key=f"cancel_del_{summary.id}", type="secondary"):
                    st.session_state.pop("pending_delete", None)
                    st.rerun()
    remaining = len(filtered) - shown
    if remaining > 0:
        if st.sidebar.button(f"Show more ({remaining} left)", type="secondary"):
            st.session_state["history_shown"] = shown + 5
            st.rerun()


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

st.set_page_config(page_title="AI Test Plan Comparator", page_icon="📋", layout="wide")
require_login()

st.title("AI Test Plan Comparator")
st.session_state.setdefault("app_view", "Setup")
st.session_state.setdefault("run_origin", "live")
if "pending_app_view" in st.session_state:
    st.session_state["app_view"] = st.session_state.pop("pending_app_view")
if _current_run() is None and st.session_state.get("app_view") == "Results":
    st.session_state["app_view"] = "Setup"

st.radio(
    "Workspace",
    ["Setup", "Results"],
    horizontal=True,
    key="app_view",
    help="Setup is files and Run. Results is the last comparison only.",
)

with st.sidebar:
    st.header("Judge (optional)")
    st.caption(
        "Fuzzy scoring always runs. An optional judge checks whether cases cover GDD meaning."
    )
    ollama_models = probe_ollama()
    ollama_names = [item.get("name") for item in ollama_models if item.get("name")]
    groq_default = groq_key()
    engine_options = ["Offline fuzzy only", "Local Ollama", "Groq API"]
    judge_choice = st.radio(
        "Scoring mode",
        engine_options,
        key="judge_engine",
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
                help="Smaller models usually respond faster.",
            )
            judge_config = JudgeConfig(backend="ollama", model=picked, base_url="http://127.0.0.1:11434")
            st.caption(f"Using local {picked}. First run may be slow while the model loads.")
    elif judge_choice == "Groq API":
        if groq_default:
            st.caption("GROQ_API_KEY is set in the environment.")
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
            format_func=_groq_model_label,
            help="Larger models are slower and typically have a lower tokens-per-minute cap.",
        )
        st.caption(
            f"Packing to {_groq_tpm_limit(groq_model) // 1000}k tokens/min for this model. "
            "The judge may pause between batches if the minute window is full."
        )
        if groq_input:
            judge_config = JudgeConfig(
                backend="groq",
                model=groq_model.strip() or DEFAULT_GROQ_MODEL,
                api_key=groq_input.strip(),
            )
        else:
            st.warning("Set GROQ_API_KEY in `.env` or paste a key to enable the judge.")

_render_history_sidebar()
render_logout()

app_view = st.session_state.get("app_view", "Setup")

if app_view == "Setup":
    st.caption("1. Upload generated plans  ·  2. Optional GDD / stock plan  ·  3. Run")

(
    stock_tp_file,
    source_doc_file,
    llm_files,
    selected_areas,
    selected_gdd_origins,
    gdd_origin_options,
) = _render_setup_inputs() if app_view == "Setup" else (None, None, {}, [], [], [])

if app_view == "Setup" and not llm_files and not stock_tp_file and not source_doc_file:
    st.info(
        "Start with a generated plan CSV. Add a GDD if you have no stock plan, "
        "then Run. A 5-row parse preview appears after each file loads."
    )

# Keep keyed uploaders mounted on Results so files survive view switches.
if app_view == "Results":
    displayed = _current_run()
    _render_results_banner()
    if displayed:
        render_comparison_view(displayed)
    else:
        st.info(
            "No comparison yet. Go to Setup, upload at least one generated plan "
            "plus a GDD, a stock plan, or a second generated plan, then Run."
        )
    st.divider()
    st.caption(
        "Files below are for the **next** Run. They are not what produced the scores above."
    )
    with st.expander("Files for the next Run", expanded=False):
        (
            stock_tp_file,
            source_doc_file,
            llm_files,
            selected_areas,
            selected_gdd_origins,
            gdd_origin_options,
        ) = _render_setup_inputs()


def run_comparison(stock_file, llm_uploads, source_intents, areas, judge_config: JudgeConfig):
    evaluations: List[ModelEvaluation] = []
    candidate_plans = []

    stock_plan = None
    if stock_file:
        st.caption("Parsing stock plan…")
        _rewind(stock_file)
        stock_plan = ingest_and_parse_test_plan_csv(stock_file, "Stock")
        stock_plan = filter_plan_by_areas(stock_plan, areas)
        st.caption(f"Stock plan cases used: {len(stock_plan.test_cases)}")
    else:
        st.caption("No stock plan — scoring against GDD and/or the other generated plans.")
    if source_intents:
        st.caption(f"GDD beats: {len(source_intents)}")

    for model_name, file in llm_uploads:
        st.caption(f"Parsing {model_name}…")
        try:
            _rewind(file)
            candidate_plans.append(ingest_and_parse_test_plan_csv(file, model_name))
        except Exception as e:
            logging.error("Failed to parse %s: %s", model_name, e)
            _fail("Parse", e, model_name)
            evaluations.append(ModelEvaluation(model_name=model_name, error=str(e)))

    cross = compare_generated_plans(candidate_plans, source_intents=source_intents or None)
    stock_cases = stock_plan.test_cases if stock_plan else None

    for plan in candidate_plans:
        st.caption(f"Evaluating {plan.model_name}…")
        try:
            evaluation = compare_and_evaluate_plan(
                stock_plan,
                plan,
                source_intents=source_intents or None,
                peer_coverage_ratio=cross.model_peer_ratios.get(plan.model_name),
                peer_coverage_note=cross.model_peer_notes.get(plan.model_name),
            )
            if judge_config.backend != "off":
                st.caption(
                    f"Judge ({judge_config.backend} / {judge_config.model}) "
                    f"for {plan.model_name}…"
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
                        f"Judge failed for {plan.model_name}: {llm_error}. "
                        "Fuzzy scores are still shown."
                    )
            evaluations.append(evaluation)
        except Exception as e:
            logging.error("Failed to evaluate %s: %s", plan.model_name, e)
            _fail("Evaluate", e, plan.model_name)
            evaluations.append(ModelEvaluation(model_name=plan.model_name, error=str(e)))

    return evaluations, cross


uploaded_llms = [(name, file) for name, file in llm_files.items() if file is not None]
gdd_filter_empty = bool(source_doc_file and gdd_origin_options and not selected_gdd_origins)

if app_view == "Setup":
    if _current_run() is not None:
        st.caption("A comparison is ready on the Results view.")
    has_reference = bool(stock_tp_file or source_doc_file or len(uploaded_llms) >= 2)
    run_blockers = []
    if not uploaded_llms:
        run_blockers.append("Upload at least one generated plan.")
    elif not has_reference:
        run_blockers.append(
            "Upload a GDD, a stock plan, or a second generated plan so there is something to compare against."
        )
    if gdd_filter_empty:
        run_blockers.append(
            "Select at least one GDD section to score against, or remove the GDD file."
        )
    run_clicked = st.button(
        "Run comparison",
        type="primary",
        disabled=bool(run_blockers),
    )
    if run_blockers:
        st.caption(run_blockers[0])

    if run_clicked and not run_blockers:
        with st.spinner("Evaluating generated plans…"):
            try:
                _rewind(source_doc_file)
                source_intents = merge_source_intents("", source_doc_file)
                if source_doc_file and selected_gdd_origins:
                    source_intents = [
                        intent
                        for intent in source_intents
                        if intent.origin in selected_gdd_origins
                    ]
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
                st.session_state["run_origin"] = "live"
                _queue_app_view("Results")
                st.rerun()
            except Exception as e:
                _fail("Run", e)
                logging.error("Dashboard critical error: %s", e)
