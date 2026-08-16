"""
Head-to-head comparison of generated test plans.

Builds a union of distinct test intents across uploaded plans (no stock TP required)
and optionally maps GDD beats to which models cover them.
"""

import re
from typing import Dict, List, Optional, Tuple

from rapidfuzz import fuzz

from schemas import (
    ClusterMember,
    CrossPlanComparison,
    GddBeatCoverage,
    GeneratedTestPlan,
    IntentCluster,
    PairwiseOverlap,
    TestCase,
)
from source_parser import SourceIntent

SEMANTIC_MATCH = 72
PARTIAL_MATCH = 52

_TITLE_BOILERPLATE = re.compile(
    r"^\s*(verify that|verify the|verify|validate|confirm the option to|confirm the|confirm)\s+",
    re.I,
)


def _core_title(text: str) -> str:
    cleaned = _TITLE_BOILERPLATE.sub("", text or "").strip()
    return cleaned or (text or "")


def _ratio(a: str, b: str) -> int:
    return int(fuzz.token_set_ratio(a or "", b or ""))


def _cluster_score(left: TestCase, right: TestCase) -> int:
    score = _ratio(_core_title(left.title), _core_title(right.title))
    if left.area and right.area:
        area_score = _ratio(left.area, right.area)
        if area_score >= 70:
            score = min(100, score + 8)
    return score


def cluster_intents(plans: List[GeneratedTestPlan]) -> List[IntentCluster]:
    """Greedy clusters of similar titles across (and within) generated plans."""
    items: List[Tuple[str, TestCase]] = []
    for plan in plans:
        for case in plan.test_cases:
            items.append((plan.model_name, case))

    used = [False] * len(items)
    clusters: List[IntentCluster] = []
    for i, (model, case) in enumerate(items):
        if used[i]:
            continue
        members = [
            ClusterMember(
                model_name=model,
                case_id=case.id or "",
                title=case.title,
                area=case.area,
            )
        ]
        used[i] = True
        for j in range(i + 1, len(items)):
            if used[j]:
                continue
            other_model, other = items[j]
            best = max(_cluster_score(case, other), *(
                _ratio(_core_title(m.title), _core_title(other.title)) for m in members
            ))
            if best >= SEMANTIC_MATCH:
                members.append(
                    ClusterMember(
                        model_name=other_model,
                        case_id=other.id or "",
                        title=other.title,
                        area=other.area,
                    )
                )
                used[j] = True
        clusters.append(IntentCluster(label=case.title, members=members))
    return clusters


def _models_in(cluster: IntentCluster) -> List[str]:
    seen = []
    for member in cluster.members:
        if member.model_name not in seen:
            seen.append(member.model_name)
    return seen


def peer_coverage_for_model(
    model_name: str, clusters: List[IntentCluster]
) -> Tuple[float, str]:
    if not clusters:
        return 0.0, "No shared intent clusters to compare."
    hits = sum(1 for cluster in clusters if model_name in _models_in(cluster))
    ratio = hits / len(clusters)
    return ratio, (
        f"{hits}/{len(clusters)} distinct intents found across all uploaded generated "
        f"plans also appear in this plan (title similarity ≥ {SEMANTIC_MATCH}%)."
    )


def _gdd_beat_coverage(
    plans: List[GeneratedTestPlan], source_intents: List[SourceIntent]
) -> List[GddBeatCoverage]:
    beats: List[GddBeatCoverage] = []
    for intent in source_intents:
        covered_by = []
        best_type = "unmatched"
        preview = intent.text if len(intent.text) <= 180 else intent.text[:177] + "..."
        for plan in plans:
            best = 0
            for case in plan.test_cases:
                title_score = _ratio(intent.text, case.title)
                blob = " ".join(
                    [case.title] + (case.steps or []) + ([case.expected_result] if case.expected_result else [])
                )
                best = max(best, title_score, _ratio(intent.text, blob))
            if best >= SEMANTIC_MATCH:
                covered_by.append(plan.model_name)
                best_type = "semantic"
            elif best >= PARTIAL_MATCH:
                covered_by.append(plan.model_name)
                if best_type != "semantic":
                    best_type = "partial"
        beats.append(
            GddBeatCoverage(
                beat_id=intent.id,
                beat_text=preview,
                origin=intent.origin,
                covered_by=covered_by,
                match_type=best_type if covered_by else "unmatched",
            )
        )
    return beats


def compare_generated_plans(
    plans: List[GeneratedTestPlan],
    source_intents: Optional[List[SourceIntent]] = None,
) -> CrossPlanComparison:
    names = [plan.model_name for plan in plans]
    clusters = cluster_intents(plans) if plans else []

    consensus: List[IntentCluster] = []
    partial: List[IntentCluster] = []
    unique_by_model: Dict[str, List[IntentCluster]] = {name: [] for name in names}
    for cluster in clusters:
        present = _models_in(cluster)
        if len(names) >= 2 and len(present) == len(names):
            consensus.append(cluster)
        elif len(present) == 1:
            unique_by_model[present[0]].append(cluster)
        else:
            partial.append(cluster)

    pairwise: List[PairwiseOverlap] = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            left_set = {idx for idx, cluster in enumerate(clusters) if left in _models_in(cluster)}
            right_set = {idx for idx, cluster in enumerate(clusters) if right in _models_in(cluster)}
            shared = len(left_set & right_set)
            union = len(left_set | right_set)
            pairwise.append(
                PairwiseOverlap(
                    model_a=left,
                    model_b=right,
                    shared=shared,
                    union=union,
                    jaccard=round(shared / union, 3) if union else 0.0,
                )
            )

    ratios: Dict[str, float] = {}
    notes: Dict[str, str] = {}
    if len(plans) >= 2:
        for name in names:
            ratio, note = peer_coverage_for_model(name, clusters)
            ratios[name] = ratio
            notes[name] = note

    gdd_beats: List[GddBeatCoverage] = []
    if source_intents:
        gdd_beats = _gdd_beat_coverage(plans, source_intents)

    return CrossPlanComparison(
        cluster_count=len(clusters),
        consensus=consensus,
        partial_overlap=partial,
        unique_by_model=unique_by_model,
        pairwise=pairwise,
        gdd_beats=gdd_beats,
        model_peer_ratios=ratios,
        model_peer_notes=notes,
    )
