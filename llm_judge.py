"""
Optional semantic judge: local Ollama or an OpenAI-compatible API (Groq, etc.).

The fuzzy evaluator still runs. This layer answers whether a generated plan
actually covers GDD/stock *meaning*, which string similarity cannot do.
MCP is not used here — the dashboard calls HTTP APIs directly.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

from evaluator import _ratio_to_score
from schemas import CoverageAlignment, GeneratedTestPlan, MetricScore, ModelEvaluation, TestCase
from source_parser import SourceIntent

load_dotenv(Path(__file__).resolve().parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
GROQ_URL = "https://api.groq.com/openai/v1"
# Developer-plan chat TPM from Groq Organization Limits (tokens per minute).
GROQ_MODEL_TPM = {
    "llama-3.3-70b-versatile": 12_000,
    "openai/gpt-oss-20b": 8_000,
    "openai/gpt-oss-120b": 8_000,
    "qwen/qwen2.5-27b": 8_000,
    "llama-3.1-8b-instant": 6_000,
}
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_MODELS = tuple(GROQ_MODEL_TPM)
BEAT_BATCH = 18
CASE_LINE_LIMIT = 220
GROQ_BEAT_MAX_TOKENS = 900
GROQ_QUAL_MAX_TOKENS = 400
_GROQ_TPM_WINDOW: List[Tuple[float, int]] = []


@dataclass
class JudgeConfig:
    backend: str  # off | ollama | groq | openai
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    timeout_sec: int = 180


def _on_streamlit_cloud() -> bool:
    return Path("/mount/src").is_dir() or bool(os.environ.get("STREAMLIT_RUNTIME_ENV"))


def probe_ollama(base_url: str = OLLAMA_URL) -> List[Dict[str, Any]]:
    """Return installed Ollama models, smallest first. Empty if daemon is down."""
    if _on_streamlit_cloud():
        return []
    try:
        payload = _http_json("GET", f"{base_url}/api/tags", timeout=2)
    except Exception:
        return []
    models = payload.get("models") or []
    models.sort(key=lambda item: item.get("size") or 0)
    return models


def groq_key() -> str:
    try:
        import streamlit as st

        value = st.secrets["GROQ_API_KEY"]
        if value:
            return str(value).strip()
    except Exception:
        pass
    return (os.environ.get("GROQ_API_KEY") or os.environ.get("GROQ_KEY") or "").strip()


def _groq_client(api_key: str):
    from groq import Groq

    return Groq(api_key=api_key)


def _http_json(
    method: str,
    url: str,
    body: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: int = 60,
) -> dict:
    data = None
    req_headers = {"Accept": "application/json", **(headers or {})}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
    if not raw:
        return {}
    return json.loads(raw)


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}


def groq_tpm_limit(model: str) -> int:
    return GROQ_MODEL_TPM.get(model or DEFAULT_GROQ_MODEL, 6_000)


def groq_model_label(model: str) -> str:
    tpm = groq_tpm_limit(model)
    suffix = "recommended" if model == DEFAULT_GROQ_MODEL else f"{tpm // 1000}k TPM"
    if model == DEFAULT_GROQ_MODEL:
        suffix = f"{tpm // 1000}k TPM, recommended"
    return f"{model}  ·  {suffix}"


def _est_tokens(*parts: str) -> int:
    """Over-estimate prompt size so Groq TPM packing stays under the model cap."""
    chars = sum(len(part or "") for part in parts)
    return max(1, (chars + 2) // 3) + 24


def _prompt_budget(config: JudgeConfig, completion_tokens: int) -> Optional[int]:
    if config.backend != "groq":
        return None
    tpm = groq_tpm_limit(config.model)
    # One request must stay under TPM (Groq 413s if prompt + max_tokens > limit).
    billed_cap = min(int(tpm * 0.85), tpm - 200)
    return max(700, billed_cap - completion_tokens - 80)


def _pace_groq(request_tokens: int, tpm_limit: int) -> None:
    """Wait if this call plus recent Groq usage would exceed the 1-minute TPM cap."""
    now = time.monotonic()
    cutoff = now - 60.0
    _GROQ_TPM_WINDOW[:] = [(stamp, used) for stamp, used in _GROQ_TPM_WINDOW if stamp > cutoff]
    used = sum(tokens for _, tokens in _GROQ_TPM_WINDOW)
    if used + request_tokens <= tpm_limit:
        return
    oldest = min(stamp for stamp, _ in _GROQ_TPM_WINDOW)
    sleep_for = 60.0 - (now - oldest) + 0.5
    logging.info(
        "Groq TPM pacing: sleeping %.1fs (window used %s, next %s, cap %s)",
        sleep_for,
        used,
        request_tokens,
        tpm_limit,
    )
    time.sleep(max(0.0, sleep_for))
    _GROQ_TPM_WINDOW.clear()


def _chat(config: JudgeConfig, system: str, user: str, max_tokens: int = 800) -> dict:
    if config.backend == "groq":
        billed = _est_tokens(system, user) + max_tokens
        _pace_groq(billed, groq_tpm_limit(config.model))

    if config.backend == "ollama":
        payload = {
            "model": config.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": max_tokens},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        result = _http_json(
            "POST",
            f"{(config.base_url or OLLAMA_URL)}/api/chat",
            body=payload,
            timeout=config.timeout_sec,
        )
        content = ((result.get("message") or {}).get("content")) or ""
        return _extract_json(content)

    if config.backend in {"groq", "openai"}:
        if config.backend == "openai":
            base = (config.base_url or GROQ_URL).rstrip("/")
            headers = {"Authorization": f"Bearer {config.api_key}"}
            payload = {
                "model": config.model,
                "temperature": 0,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
            result = _http_json(
                "POST",
                f"{base}/chat/completions",
                body=payload,
                headers=headers,
                timeout=config.timeout_sec,
            )
            content = (((result.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
            return _extract_json(content)

        client = _groq_client(config.api_key)
        billed = _est_tokens(system, user) + max_tokens
        _GROQ_TPM_WINDOW.append((time.monotonic(), billed))
        completion = client.chat.completions.create(
            model=config.model or DEFAULT_GROQ_MODEL,
            temperature=0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = (completion.choices[0].message.content or "") if completion.choices else ""
        return _extract_json(content)

    raise ValueError(f"Unknown judge backend: {config.backend}")


def _clip(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _case_catalog(
    cases: Sequence[TestCase],
    line_limit: int = CASE_LINE_LIMIT,
    detail: str = "full",
) -> str:
    lines = []
    for case in cases:
        if detail == "title":
            line = f"{case.id or '?'} | {case.area or '-'} | {case.title}"
        elif detail == "short":
            er = _clip(case.expected_result or "", 90)
            line = f"{case.id or '?'} | {case.area or '-'} | {case.title} | ER: {er}"
        else:
            steps = "; ".join(case.steps[:2]) if case.steps else ""
            er = case.expected_result or ""
            line = (
                f"{case.id or '?'} | {case.area or '-'} | {case.title} "
                f"| STR: {steps} | ER: {er}"
            )
        lines.append(_clip(line, line_limit))
    return "\n".join(lines)


def _beat_lines(intents: Sequence[SourceIntent], clip: int = 280) -> str:
    lines = []
    for intent in intents:
        lines.append(_clip(f"{intent.id} | {intent.origin} | {intent.text}", clip))
    return "\n".join(lines)


def _diverse_cases(cases: Sequence[TestCase], keep: int) -> List[TestCase]:
    if keep >= len(cases):
        return list(cases)
    by_area: Dict[str, List[TestCase]] = {}
    for case in cases:
        by_area.setdefault(case.area or "-", []).append(case)
    picked: List[TestCase] = []
    areas = list(by_area.keys())
    index = 0
    while len(picked) < keep and areas:
        area = areas[index % len(areas)]
        bucket = by_area[area]
        if bucket:
            picked.append(bucket.pop(0))
        if not bucket:
            areas = [item for item in areas if item != area]
            if not areas:
                break
            index = 0
            continue
        index += 1
    return picked


def _fit_catalog(
    cases: Sequence[TestCase],
    system: str,
    suffix: str,
    budget: Optional[int],
) -> Tuple[str, Dict[str, Any]]:
    attempts = (
        ("full", CASE_LINE_LIMIT),
        ("full", 140),
        ("short", 120),
        ("title", 90),
        ("title", 70),
    )
    selected = list(cases)
    catalog = _case_catalog(selected)
    meta: Dict[str, Any] = {
        "detail": "full",
        "line_limit": CASE_LINE_LIMIT,
        "cases_sent": len(selected),
        "cases_total": len(cases),
    }
    if budget is None:
        return catalog, meta

    for detail, line_limit in attempts:
        catalog = _case_catalog(selected, line_limit, detail)
        if _est_tokens(system, catalog, suffix) <= budget:
            meta.update(detail=detail, line_limit=line_limit, cases_sent=len(selected))
            return catalog, meta

    low, high = 8, len(selected)
    best_n = 8
    while low <= high:
        mid = (low + high) // 2
        subset = _diverse_cases(cases, mid)
        catalog = _case_catalog(subset, 70, "title")
        if _est_tokens(system, catalog, suffix) <= budget:
            best_n = mid
            low = mid + 1
        else:
            high = mid - 1
    selected = _diverse_cases(cases, max(1, best_n))
    catalog = _case_catalog(selected, 70, "title")
    meta.update(detail="title", line_limit=70, cases_sent=len(selected), truncated=True)
    logging.info(
        "Judge catalog packed: %s/%s cases, detail=%s, ~%s tokens",
        meta["cases_sent"],
        meta["cases_total"],
        meta["detail"],
        _est_tokens(system, catalog, suffix),
    )
    return catalog, meta


def _beat_user(plan_name: str, catalog: str, chunk: Sequence[SourceIntent], beat_clip: int) -> str:
    return (
        f"Generated plan: {plan_name}\n\n"
        f"TEST CASES (id | area | title | STR | ER):\n{catalog}\n\n"
        f"DESIGN BEATS to score:\n{_beat_lines(chunk, beat_clip)}\n\n"
        "Return JSON: {\"beats\":[{\"id\":\"...\",\"match\":\"semantic|partial|unmatched\","
        "\"case_ids\":[\"id\"],\"note\":\"one short sentence\"}]}\n"
        "Every beat id must appear once. case_ids empty if unmatched."
    )


def _largest_beat_chunk(
    plan_name: str,
    catalog: str,
    remaining: Sequence[SourceIntent],
    system: str,
    budget: Optional[int],
) -> Tuple[List[SourceIntent], int]:
    cap = min(BEAT_BATCH, len(remaining))
    if budget is None:
        return list(remaining[:cap]), 280

    for beat_clip in (200, 140, 90):
        best = 1
        low, high = 1, cap
        while low <= high:
            mid = (low + high) // 2
            user = _beat_user(plan_name, catalog, remaining[:mid], beat_clip)
            if _est_tokens(system, user) <= budget:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        user = _beat_user(plan_name, catalog, remaining[:best], beat_clip)
        if _est_tokens(system, user) <= budget:
            return list(remaining[:best]), beat_clip
    return list(remaining[:1]), 80

def _stock_as_intents(stock_cases: Sequence[TestCase]) -> List[SourceIntent]:
    intents = []
    for case in stock_cases:
        intents.append(
            SourceIntent(
                id=case.id or f"stock_{len(intents) + 1}",
                text=f"[{case.area}] {case.title}" if case.area else case.title,
                origin=case.area or "stock",
            )
        )
    return intents


def _clamp_score(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 3
    return min(5, max(1, number))


def judge_plan(
    plan: GeneratedTestPlan,
    source_intents: Optional[List[SourceIntent]],
    stock_cases: Optional[List[TestCase]],
    config: JudgeConfig,
) -> Dict[str, Any]:
    """
    Returns alignments plus 1-5 LLM metrics. Raises on transport/parse failure.
    """
    reference = list(source_intents or [])
    if not reference and stock_cases:
        reference = _stock_as_intents(stock_cases)

    system = (
        "You are a senior game QA reviewer. Compare a generated test plan against "
        "design beats. Judge MEANING, not wording. "
        "semantic = the case would actually test that beat; "
        "partial = related but missing a key check; "
        "unmatched = not tested. "
        "Reply with JSON only."
    )
    beat_budget = _prompt_budget(config, GROQ_BEAT_MAX_TOKENS)
    reserve_n = min(BEAT_BATCH, max(len(reference), 1))
    beat_placeholder = _beat_lines(reference[:reserve_n], 140) if reference else "(none)"
    catalog_suffix = f"DESIGN BEATS:\n{beat_placeholder}\nReturn JSON beats."
    catalog, pack_meta = _fit_catalog(plan.test_cases, system, catalog_suffix, beat_budget)
    alignments: List[CoverageAlignment] = []

    if reference:
        cursor = 0
        while cursor < len(reference):
            chunk, beat_clip = _largest_beat_chunk(
                plan.model_name,
                catalog,
                reference[cursor:],
                system,
                beat_budget,
            )
            user = _beat_user(plan.model_name, catalog, chunk, beat_clip)
            if beat_budget is not None:
                lines = catalog.splitlines()
                while (
                    lines
                    and len(lines) > 4
                    and _est_tokens(system, user) > beat_budget
                ):
                    lines = lines[: max(4, int(len(lines) * 0.85))]
                    catalog = "\n".join(lines)
                    pack_meta = dict(pack_meta)
                    pack_meta["cases_sent"] = len(lines)
                    pack_meta["truncated"] = True
                    user = _beat_user(plan.model_name, catalog, chunk, beat_clip)
            estimated = _est_tokens(system, user)
            logging.info(
                "Judge beat batch %s-%s (~%s prompt tokens, clip=%s)",
                cursor + 1,
                cursor + len(chunk),
                estimated,
                beat_clip,
            )
            parsed = _chat(config, system, user, max_tokens=GROQ_BEAT_MAX_TOKENS)
            rows = parsed.get("beats") or parsed.get("items") or []
            by_id = {str(row.get("id")): row for row in rows if isinstance(row, dict)}
            for intent in chunk:
                row = by_id.get(intent.id) or {}
                match = str(row.get("match") or "unmatched").lower()
                if match not in {"semantic", "partial", "unmatched"}:
                    match = "unmatched"
                case_ids = [str(cid) for cid in (row.get("case_ids") or []) if cid]
                if match == "unmatched":
                    case_ids = []
                alignments.append(
                    CoverageAlignment(
                        stock_id=intent.id,
                        stock_title=_clip(intent.text, 180),
                        matched_candidate_ids=case_ids,
                        match_type=match,  # type: ignore[arg-type]
                        best_similarity=90 if match == "semantic" else 60 if match == "partial" else 0,
                        rationale=str(row.get("note") or f"LLM marked this beat {match}."),
                    )
                )
            cursor += len(chunk)

    covered = sum(1 for item in alignments if item.match_type in {"semantic", "partial"})
    coverage_ratio = covered / max(len(alignments), 1) if alignments else 0.0
    unmatched_preview = [
        f"{item.stock_id}: {item.stock_title}"
        for item in alignments
        if item.match_type == "unmatched"
    ][:12]
    qual_budget = _prompt_budget(config, GROQ_QUAL_MAX_TOKENS)
    unmatched_block = "\n".join(unmatched_preview) if unmatched_preview else "(none)"
    qual_suffix = f"Unmatched design beats:\n{unmatched_block}\nScore 1-5 JSON."
    qual_catalog, _ = _fit_catalog(plan.test_cases, system, qual_suffix, qual_budget)
    qual_user = (
        f"Generated plan: {plan.model_name} ({len(plan.test_cases)} cases).\n\n"
        f"TEST CASES:\n{qual_catalog}\n\n"
        f"Unmatched design beats:\n"
        + unmatched_block
        + "\n\n"
        "Score 1-5:\n"
        "- gdd_fidelity: cases stay true to the design, little invented scope\n"
        "- tester_readiness: a QA could execute STR and expected results without guessing\n"
        "Return JSON: {\"gdd_fidelity\":n,\"tester_readiness\":n,\"summary\":\"3-5 sentences\"}"
    )
    if qual_budget is not None:
        while _est_tokens(system, qual_user) > qual_budget and len(unmatched_preview) > 0:
            unmatched_preview = unmatched_preview[:-2]
            unmatched_block = "\n".join(unmatched_preview) if unmatched_preview else "(none)"
            qual_user = (
                f"Generated plan: {plan.model_name} ({len(plan.test_cases)} cases).\n\n"
                f"TEST CASES:\n{qual_catalog}\n\n"
                f"Unmatched design beats:\n{unmatched_block}\n\n"
                "Score 1-5:\n"
                "- gdd_fidelity: cases stay true to the design, little invented scope\n"
                "- tester_readiness: a QA could execute STR and expected results without guessing\n"
                "Return JSON: {\"gdd_fidelity\":n,\"tester_readiness\":n,\"summary\":\"3-5 sentences\"}"
            )
    qualitative = _chat(config, system, qual_user, max_tokens=GROQ_QUAL_MAX_TOKENS)

    return {
        "alignments": alignments,
        "coverage_ratio": coverage_ratio,
        "covered": covered,
        "total": len(alignments),
        "gdd_fidelity": _clamp_score(qualitative.get("gdd_fidelity")),
        "tester_readiness": _clamp_score(qualitative.get("tester_readiness")),
        "summary": str(qualitative.get("summary") or "").strip(),
        "backend": config.backend,
        "model": config.model,
        "prompt_pack": pack_meta,
    }


def apply_llm_judgment(evaluation: ModelEvaluation, result: Dict[str, Any]) -> ModelEvaluation:
    covered = result.get("covered") or 0
    total = result.get("total") or 0
    coverage_ratio = result.get("coverage_ratio") or 0.0
    if total:
        evaluation.llm_gdd_coverage = MetricScore(
            score=_ratio_to_score(coverage_ratio),
            reasoning=(
                f"LLM judged {covered}/{total} design beats as covered "
                f"(semantic or partial) using {result.get('backend')} / {result.get('model')}."
            ),
        )
    evaluation.llm_gdd_fidelity = MetricScore(
        score=_clamp_score(result.get("gdd_fidelity")),
        reasoning="LLM score: how faithfully the cases stay on the design (little invented scope).",
    )
    evaluation.llm_tester_readiness = MetricScore(
        score=_clamp_score(result.get("tester_readiness")),
        reasoning="LLM score: whether a tester could run the STR and expected results as written.",
    )
    evaluation.llm_alignments = result.get("alignments") or []
    evaluation.llm_summary = result.get("summary") or None
    diagnostics = dict(evaluation.diagnostics or {})
    diagnostics["llm_backend"] = result.get("backend")
    diagnostics["llm_model"] = result.get("model")
    diagnostics["llm_beats_judged"] = total
    if result.get("prompt_pack"):
        diagnostics["llm_prompt_pack"] = result["prompt_pack"]
    evaluation.diagnostics = diagnostics
    evaluation.recompute_overall_score()
    return evaluation
