"""
Offline judge: score generated test plans.

Reference can be a stock checklist, a GDD/source document, other generated
plans (peer coverage), or any combination. Fuzzy title matching. No API keys.
"""

import logging
import re
import statistics
from collections import Counter
from typing import List, Optional, Tuple

from rapidfuzz import fuzz

from schemas import (
    CoverageAlignment,
    GeneratedTestPlan,
    MetricScore,
    ModelEvaluation,
    TestCase,
)
from source_parser import SourceIntent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SEMANTIC_MATCH = 72
PARTIAL_MATCH = 52
IN_SCOPE_EXTRA = 40
NEAR_DUPLICATE = 88
NEAR_DUPLICATE_STEPS = 90

_EDGE_HINTS = (
    "edge",
    "boundary",
    "invalid",
    "empty",
    "null",
    "none",
    "max",
    "min",
    "overflow",
    "offline",
    "timeout",
    "interrupt",
    "cancel",
    "retry",
    "duplicate",
    "special",
    "unicode",
    "long string",
    "concurrent",
    "permission",
    "unauthorized",
    "negative",
    "zero",
    "network",
    "disconnect",
)

_ACTION_VERBS = (
    "tap",
    "click",
    "press",
    "enter",
    "type",
    "select",
    "swipe",
    "wait",
    "verify",
    "check",
    "open",
    "close",
    "navigate",
    "launch",
    "observe",
    "confirm",
    "scroll",
    "set",
    "enable",
    "disable",
    "login",
    "log in",
    "sign in",
    "purchase",
    "buy",
    "claim",
    "watch",
    "reward",
)

_SPECIFIC_ER = (
    "display",
    "shown",
    "show",
    "error",
    "message",
    "screen",
    "popup",
    "dialog",
    "toast",
    "remain",
    "disabled",
    "enabled",
    "redirect",
    "value",
    "count",
    "score",
    "balance",
    "coins",
    "reward",
    "ad",
    "video",
    "button",
)

_VAGUE_ER = (
    "as expected",
    "should work",
    "works correctly",
    "works fine",
    "success",
    "successful",
    "ok",
    "properly",
    "correctly",
    "no issues",
    "as per requirement",
)


def _ratio(a: str, b: str) -> int:
    return int(fuzz.token_set_ratio(a or "", b or ""))


_TITLE_BOILERPLATE = re.compile(
    r"^\s*(verify that|verify the|verify|validate|confirm the option to|confirm the|confirm)\s+",
    re.I,
)


def _core_title(text: str) -> str:
    cleaned = _TITLE_BOILERPLATE.sub("", text or "").strip()
    return cleaned or (text or "")


def _case_intent(tc: TestCase) -> str:
    return _core_title(tc.title)


def _match_score(stock: TestCase, gen: TestCase) -> int:
    title_score = _ratio(_core_title(stock.title), _core_title(gen.title))
    if stock.area and gen.area:
        area_score = _ratio(stock.area, gen.area)
        if area_score >= 70:
            title_score = min(100, title_score + 8)
    return title_score


def _best_against_stock(title: str, stock_cases: List[TestCase]) -> Tuple[int, Optional[TestCase]]:
    probe = TestCase(id="probe", title=title)
    best_score = -1
    best_case: Optional[TestCase] = None
    for stock in stock_cases:
        score = _match_score(stock, probe)
        if score > best_score:
            best_score = score
            best_case = stock
    return best_score, best_case


def _ratio_to_score(ratio: float) -> int:
    if ratio >= 0.90:
        return 5
    if ratio >= 0.75:
        return 4
    if ratio >= 0.55:
        return 3
    if ratio >= 0.35:
        return 2
    return 1


def _actionability_unit(tc: TestCase) -> float:
    score = 0.0
    if tc.steps:
        score += 0.45
        joined = " ".join(tc.steps)
        if len(tc.steps) >= 2 or len(joined) >= 40:
            score += 0.15
        if len(joined) >= 80:
            score += 0.10
    if tc.expected_result and len(tc.expected_result.strip()) >= 8:
        score += 0.30
    return min(score, 1.0)


def _str_completeness_unit(tc: TestCase) -> float:
    if not tc.steps:
        return 0.0
    joined = " ".join(tc.steps).lower()
    n = len(tc.steps)
    verb_hits = sum(1 for verb in _ACTION_VERBS if verb in joined)
    score = 0.25
    if n >= 2:
        score += 0.25
    if n >= 4:
        score += 0.10
    if verb_hits >= 2:
        score += 0.25
    elif verb_hits >= 1:
        score += 0.15
    avg_len = sum(len(step) for step in tc.steps) / n
    if avg_len >= 20:
        score += 0.15
    return min(score, 1.0)


def _expected_result_unit(tc: TestCase) -> float:
    er = (tc.expected_result or "").strip()
    if not er:
        return 0.0
    low = er.lower()
    score = 0.35
    if len(er) >= 20:
        score += 0.20
    if len(er) >= 50:
        score += 0.10
    if any(token in low for token in _SPECIFIC_ER):
        score += 0.20
    if re.search(r"\d", er):
        score += 0.10
    if any(phrase in low for phrase in _VAGUE_ER) and len(er) < 40:
        score -= 0.20
    return min(max(score, 0.0), 1.0)


def _joined_steps(tc: TestCase) -> str:
    return " ".join(tc.steps).strip()


def _looks_like_edge(title: str) -> bool:
    lowered = title.lower()
    return any(hint in lowered for hint in _EDGE_HINTS)


def _case_search_text(tc: TestCase) -> str:
    parts = [tc.title]
    if tc.steps:
        parts.extend(tc.steps)
    if tc.expected_result:
        parts.append(tc.expected_result)
    return " ".join(parts)


def _match_source_intents(
    gen_cases: List[TestCase], source_intents: List[SourceIntent]
) -> Tuple[List[CoverageAlignment], float, float, List[str]]:
    alignments: List[CoverageAlignment] = []
    covered_gen_ids = set()

    for intent in source_intents:
        scored = []
        for gen in gen_cases:
            title_score = _ratio(intent.text, gen.title)
            blob_score = _ratio(intent.text, _case_search_text(gen))
            scored.append((max(title_score, blob_score), gen))
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_gen = scored[0]
        matches = [gen for score, gen in scored if score >= PARTIAL_MATCH]

        if best_score >= SEMANTIC_MATCH:
            match_type = "semantic"
        elif best_score >= PARTIAL_MATCH:
            match_type = "partial"
        else:
            match_type = "unmatched"

        matched_ids = [g.id or "" for g in matches] if match_type != "unmatched" else []
        if match_type != "unmatched":
            covered_gen_ids.update(matched_ids)

        preview = intent.text if len(intent.text) <= 180 else intent.text[:177] + "..."
        alignments.append(
            CoverageAlignment(
                stock_id=intent.id,
                stock_title=preview,
                matched_candidate_ids=matched_ids,
                match_type=match_type,
                best_similarity=best_score,
                rationale=(
                    f"[{intent.origin}] Best generated match [{best_gen.id}] '{best_gen.title}' "
                    f"at {best_score}% similarity to the source statement."
                    if match_type != "unmatched"
                    else f"[{intent.origin}] No generated case reached {PARTIAL_MATCH}% similarity. "
                    f"Closest was [{best_gen.id}] '{best_gen.title}' ({best_score}%)."
                ),
            )
        )

    covered = sum(1 for a in alignments if a.match_type in {"semantic", "partial"})
    coverage_ratio = covered / max(len(source_intents), 1)

    off_source = []
    for gen in gen_cases:
        gen_id = gen.id or ""
        blob = _case_search_text(gen)
        best = max(_ratio(intent.text, blob) for intent in source_intents)
        if best < IN_SCOPE_EXTRA:
            off_source.append(gen_id)
    fidelity_ratio = 1.0 - (len(off_source) / max(len(gen_cases), 1))
    return alignments, coverage_ratio, fidelity_ratio, off_source


def intents_as_cases(source_intents: List[SourceIntent]) -> List[TestCase]:
    cases: List[TestCase] = []
    for intent in source_intents:
        origin = intent.origin or ""
        area = origin if origin.lower().startswith("mission") else None
        cases.append(TestCase(id=intent.id, title=intent.text, area=area))
    return cases


def _keyword_edge_ratio(gen_cases: List[TestCase]) -> Tuple[float, str]:
    hits = sum(1 for gen in gen_cases if _looks_like_edge(gen.title))
    frac = hits / max(len(gen_cases), 1)
    if frac == 0:
        ratio = 0.25
    elif frac <= 0.35:
        ratio = 0.45 + 0.55 * (frac / 0.35)
    else:
        ratio = max(0.55, 1.0 - 0.4 * min((frac - 0.35) / 0.40, 1.0))
    return ratio, (
        f"{hits}/{len(gen_cases)} titles look like edge/negative/boundary cases."
    )


def compare_and_evaluate_plan(
    stock_plan: Optional[GeneratedTestPlan],
    candidate_plan: GeneratedTestPlan,
    source_intents: Optional[List[SourceIntent]] = None,
    peer_coverage_ratio: Optional[float] = None,
    peer_coverage_note: Optional[str] = None,
) -> ModelEvaluation:
    """
    Score a generated plan. Stock is optional. GDD/source can be the checklist.
    Peer coverage is the share of distinct intents across all uploaded generated plans.
    """
    gen_cases = candidate_plan.test_cases
    if not gen_cases:
        return ModelEvaluation(
            model_name=candidate_plan.model_name,
            error="Generated plan has no test cases.",
        )

    stock_cases = list(stock_plan.test_cases) if stock_plan and stock_plan.test_cases else []
    gdd_as_reference = False
    if not stock_cases and source_intents:
        stock_cases = intents_as_cases(source_intents)
        gdd_as_reference = True
        ref_label = "GDD/source beats"
    else:
        ref_label = "stock intents"

    logging.info(
        "Evaluating generated plan '%s' against %s %s (%s generated cases).",
        candidate_plan.model_name,
        len(stock_cases),
        ref_label,
        len(gen_cases),
    )

    alignments: List[CoverageAlignment] = []
    extra_in_scope: List[str] = []
    extra_out_of_scope: List[str] = []
    coverage_metric = None
    match_confidence_metric = None
    area_metric = None
    scope_metric = None
    trace_metric = None
    covered = 0
    stock_areas: List[str] = []
    area_hits = 0
    avg_similarity = 0.0

    if stock_cases:
        covered_gen_ids = set()
        best_similarities: List[int] = []
        for stock in stock_cases:
            scored = sorted(
                ((_match_score(stock, gen), gen) for gen in gen_cases),
                key=lambda item: item[0],
                reverse=True,
            )
            best_score, best_gen = scored[0]
            best_similarities.append(best_score)
            matches = [gen for score, gen in scored if score >= PARTIAL_MATCH]

            if best_score >= SEMANTIC_MATCH:
                match_type = "semantic"
            elif best_score >= PARTIAL_MATCH:
                match_type = "partial"
            else:
                match_type = "unmatched"

            matched_ids = [g.id or "" for g in matches] if match_type != "unmatched" else []
            if match_type != "unmatched":
                covered_gen_ids.update(matched_ids)

            alignments.append(
                CoverageAlignment(
                    stock_id=stock.id or "",
                    stock_title=f"[{stock.area}] {stock.title}" if stock.area else stock.title,
                    matched_candidate_ids=matched_ids,
                    match_type=match_type,
                    best_similarity=best_score,
                    rationale=(
                        f"Best generated match [{best_gen.id}] '{best_gen.title}' "
                        f"at {best_score}% title similarity."
                        if match_type != "unmatched"
                        else f"No generated case reached {PARTIAL_MATCH}% similarity. "
                        f"Closest was [{best_gen.id}] '{best_gen.title}' ({best_score}%)."
                    ),
                )
            )

        for gen in gen_cases:
            gen_id = gen.id or ""
            if gen_id in covered_gen_ids:
                continue
            best_score, _ = _best_against_stock(gen.title, stock_cases)
            if best_score >= IN_SCOPE_EXTRA:
                extra_in_scope.append(gen_id)
            else:
                extra_out_of_scope.append(gen_id)

        covered = sum(1 for a in alignments if a.match_type in {"semantic", "partial"})
        semantic = sum(1 for a in alignments if a.match_type == "semantic")
        coverage_ratio = covered / len(stock_cases)
        coverage_score = _ratio_to_score(coverage_ratio)
        if semantic == len(stock_cases):
            coverage_score = 5
        coverage_metric = MetricScore(
            score=coverage_score,
            reasoning=(
                f"{covered}/{len(stock_cases)} {ref_label} matched "
                f"({semantic} strong, {covered - semantic} partial). "
                "Matched by title similarity."
            ),
        )

        stock_areas = sorted(
            {(tc.area or "Unspecified").strip() or "Unspecified" for tc in stock_cases}
        )
        gen_areas = [(tc.area or "Unspecified").strip() or "Unspecified" for tc in gen_cases]
        area_hits = 0
        for area in stock_areas:
            best = max((_ratio(area, gen_area) for gen_area in gen_areas), default=0)
            if best >= PARTIAL_MATCH:
                area_hits += 1
        area_metric = MetricScore(
            score=_ratio_to_score(area_hits / max(len(stock_areas), 1)),
            reasoning=(
                f"{area_hits}/{len(stock_areas)} reference areas have generated cases "
                f"({', '.join(stock_areas[:8])}{'…' if len(stock_areas) > 8 else ''})."
            ),
        )

        avg_similarity = sum(best_similarities) / len(best_similarities)
        match_confidence_metric = MetricScore(
            score=_ratio_to_score(avg_similarity / 100.0),
            reasoning=(
                f"Average best title similarity across {ref_label} is {avg_similarity:.0f}%."
            ),
        )

        drift_ratio = len(extra_out_of_scope) / len(gen_cases)
        scope_metric = MetricScore(
            score=_ratio_to_score(1.0 - drift_ratio),
            reasoning=(
                f"{len(extra_out_of_scope)}/{len(gen_cases)} generated cases are weakly related "
                f"to any {ref_label[:-1] if ref_label.endswith('s') else ref_label} "
                f"(below {IN_SCOPE_EXTRA}% similarity)."
            ),
        )

        gen_to_stock = Counter()
        for alignment in alignments:
            for gid in alignment.matched_candidate_ids:
                gen_to_stock[gid] += 1
        healthy_maps = sum(1 for a in alignments if 1 <= len(a.matched_candidate_ids) <= 2)
        fan_in = sum(1 for count in gen_to_stock.values() if count >= 3)
        oversplit = sum(1 for a in alignments if len(a.matched_candidate_ids) >= 5)
        mapped = max(len(gen_to_stock), 1)
        trace_ratio = (
            0.60 * (healthy_maps / len(alignments))
            + 0.25 * (1.0 - fan_in / mapped)
            + 0.15 * (1.0 - oversplit / len(alignments))
        )
        trace_metric = MetricScore(
            score=_ratio_to_score(trace_ratio),
            reasoning=(
                f"{healthy_maps}/{len(alignments)} {ref_label} map to 1–2 generated cases. "
                f"{fan_in} generated cases cover 3+ reference intents; "
                f"{oversplit} intents split across 5+ cases."
            ),
        )

    action_ratio = sum(_actionability_unit(tc) for tc in gen_cases) / len(gen_cases)
    str_ratio = sum(_str_completeness_unit(tc) for tc in gen_cases) / len(gen_cases)
    er_ratio = sum(_expected_result_unit(tc) for tc in gen_cases) / len(gen_cases)
    missing_str = sum(1 for tc in gen_cases if not tc.steps)
    missing_er = sum(1 for tc in gen_cases if not tc.expected_result)
    verbish = sum(1 for tc in gen_cases if _str_completeness_unit(tc) >= 0.5)

    dup_pairs = 0
    pair_count = 0
    duplicate_ids = set()
    for i, left in enumerate(gen_cases):
        for right in gen_cases[i + 1 :]:
            pair_count += 1
            if _ratio(left.title, right.title) >= NEAR_DUPLICATE:
                dup_pairs += 1
                duplicate_ids.add(left.id or "")
                duplicate_ids.add(right.id or "")
    unique_ratio = 1.0 if pair_count == 0 else 1.0 - (dup_pairs / pair_count)
    involved_ratio = len(duplicate_ids) / len(gen_cases)
    redundancy_ratio = max(0.0, unique_ratio * (1.0 - 0.5 * involved_ratio))

    step_dup_pairs = 0
    step_pair_count = 0
    step_dup_ids = set()
    for i, left in enumerate(gen_cases):
        left_steps = _joined_steps(left)
        if len(left_steps) < 20:
            continue
        for right in gen_cases[i + 1 :]:
            right_steps = _joined_steps(right)
            if len(right_steps) < 20:
                continue
            step_pair_count += 1
            if _ratio(left_steps, right_steps) >= NEAR_DUPLICATE_STEPS:
                step_dup_pairs += 1
                step_dup_ids.add(left.id or "")
                step_dup_ids.add(right.id or "")
    step_unique = 1.0 if step_pair_count == 0 else 1.0 - (step_dup_pairs / step_pair_count)

    extra_count = len(extra_in_scope)
    keyword_hits = sum(
        1
        for gen in gen_cases
        if (gen.id or "") in extra_in_scope and _looks_like_edge(gen.title)
    )
    if stock_cases:
        if extra_count == 0:
            edge_ratio = 0.25
            edge_note = (
                f"{extra_count} extra in-scope cases beyond {ref_label} "
                f"({keyword_hits} look like explicit edge/negative cases)."
            )
        else:
            density = min(extra_count / max(len(stock_cases), 1), 1.0)
            keyword_boost = min(keyword_hits / max(extra_count, 1), 1.0)
            edge_ratio = 0.45 + 0.35 * density + 0.20 * keyword_boost
            edge_note = (
                f"{extra_count} extra in-scope cases beyond {ref_label} "
                f"({keyword_hits} look like explicit edge/negative cases)."
            )
    else:
        edge_ratio, edge_note = _keyword_edge_ratio(gen_cases)

    avg_steps = sum(len(tc.steps) for tc in gen_cases) / len(gen_cases)
    action_units = [_actionability_unit(tc) for tc in gen_cases]
    consistency = 1.0 - min(
        (statistics.pstdev(action_units) * 2) if len(action_units) > 1 else 0.0,
        1.0,
    )

    source_alignments: List[CoverageAlignment] = []
    source_coverage_metric = None
    source_fidelity_metric = None
    diagnostics = {
        "generated_cases": len(gen_cases),
        "stock_intents": 0 if gdd_as_reference else len(stock_cases),
        "unmatched_stock": (len(stock_cases) - covered) if stock_cases and not gdd_as_reference else 0,
        "reference": ref_label if stock_cases else "none (peer/quality only)",
        "stock_areas": len(stock_areas),
        "areas_covered": area_hits,
        "avg_title_similarity_pct": round(avg_similarity, 1) if stock_cases else None,
        "avg_steps_per_case": round(avg_steps, 2),
        "pct_with_str": round(100 * (1 - missing_str / len(gen_cases)), 1),
        "pct_with_expected_result": round(100 * (1 - missing_er / len(gen_cases)), 1),
        "format_consistency_0_to_1": round(consistency, 2),
        "source_statements": 0,
    }

    apply_source = bool(source_intents) and not gdd_as_reference
    if gdd_as_reference and source_intents:
        source_alignments = alignments
        diagnostics["source_statements"] = len(source_intents)
        diagnostics["unmatched_source"] = sum(
            1 for a in alignments if a.match_type == "unmatched"
        )
        diagnostics["off_source_generated"] = len(extra_out_of_scope)
    elif apply_source:
        source_alignments, src_cov, src_fid, off_source = _match_source_intents(
            gen_cases, source_intents
        )
        src_covered = sum(
            1 for a in source_alignments if a.match_type in {"semantic", "partial"}
        )
        diagnostics["source_statements"] = len(source_intents)
        diagnostics["unmatched_source"] = len(source_intents) - src_covered
        diagnostics["off_source_generated"] = len(off_source)
        source_coverage_metric = MetricScore(
            score=_ratio_to_score(src_cov),
            reasoning=(
                f"{src_covered}/{len(source_intents)} statements from the overall intent/"
                f"source document are reflected in generated cases."
            ),
        )
        source_fidelity_metric = MetricScore(
            score=_ratio_to_score(src_fid),
            reasoning=(
                f"{len(gen_cases) - len(off_source)}/{len(gen_cases)} generated cases stay "
                "within the source/intent. Others look unrelated to the generating document."
            ),
        )

    if gdd_as_reference:
        alignments = []

    peer_metric = None
    if peer_coverage_ratio is not None:
        peer_metric = MetricScore(
            score=_ratio_to_score(peer_coverage_ratio),
            reasoning=peer_coverage_note
            or f"Covers {peer_coverage_ratio:.0%} of distinct intents across uploaded generated plans.",
        )
        diagnostics["peer_coverage_pct"] = round(100 * peer_coverage_ratio, 1)

    evaluation = ModelEvaluation(
        model_name=candidate_plan.model_name,
        alignments=alignments,
        extra_in_scope_case_ids=extra_in_scope,
        extra_out_of_scope_case_ids=extra_out_of_scope,
        source_alignments=source_alignments,
        diagnostics=diagnostics,
        requirement_coverage=coverage_metric,
        match_confidence=match_confidence_metric,
        area_coverage=area_metric,
        source_coverage=source_coverage_metric,
        source_fidelity=source_fidelity_metric,
        peer_coverage=peer_metric,
        edge_case_quality=MetricScore(
            score=_ratio_to_score(edge_ratio),
            reasoning=edge_note,
        ),
        actionability=MetricScore(
            score=_ratio_to_score(action_ratio),
            reasoning=(
                f"{len(gen_cases) - missing_str}/{len(gen_cases)} cases have STR; "
                f"{len(gen_cases) - missing_er}/{len(gen_cases)} have expected results. "
                "Scored on generated plans only."
            ),
        ),
        str_completeness=MetricScore(
            score=_ratio_to_score(str_ratio),
            reasoning=(
                f"{verbish}/{len(gen_cases)} cases have usable multi-step STR with action verbs "
                f"(tap/click/verify/...). Average {avg_steps:.1f} steps per case."
            ),
        ),
        expected_result_quality=MetricScore(
            score=_ratio_to_score(er_ratio),
            reasoning=(
                f"{len(gen_cases) - missing_er}/{len(gen_cases)} cases include an expected result. "
                "Rewards specific UI/state outcomes; penalizes vague phrases like 'works correctly'."
            ),
        ),
        redundancy=MetricScore(
            score=_ratio_to_score(redundancy_ratio),
            reasoning=(
                f"{dup_pairs} near-duplicate title pairs in the generated plan "
                f"({len(duplicate_ids)} cases involved)."
            ),
        ),
        procedure_uniqueness=MetricScore(
            score=_ratio_to_score(step_unique),
            reasoning=(
                f"{step_dup_pairs} generated cases share near-identical STR text "
                f"({len(step_dup_ids)} cases involved). Copy-pasted procedures score lower."
            ),
        ),
        scope_drift=scope_metric,
        traceability=trace_metric,
    )
    logging.info(
        "Evaluated '%s' overall_score=%s",
        evaluation.model_name,
        evaluation.overall_score,
    )
    return evaluation
