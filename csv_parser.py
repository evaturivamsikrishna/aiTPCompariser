"""
Ingest stock and generated Kwalee test-plan CSVs.

Stock (Stock_mission1.1.csv): AREA / FEATURE + TEST CASE. Area is often blank
after the first row of a group and must be forward-filled. No STR / expected result.

Generated (AiTestPlanGenerator_*.csv): optional banner row, then
TestID, Area/Feature, Test Case, Steps to Reproduce, Expected Result, Source.
"""

import logging
import re
from typing import Any, List, Optional

import pandas as pd
from rapidfuzz import fuzz

from schemas import ColumnMapping, GeneratedTestPlan, TestCase

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_FIELD_ALIASES = {
    "id": ["id", "tc id", "tcid", "testid", "test id", "test case id", "case id"],
    "title": ["test case", "title", "name", "description", "scenario", "summary"],
    "area": ["area / feature", "area/feature", "area", "feature", "area feature"],
    "steps": ["steps to reproduce", "steps", "str", "procedure", "actions", "test steps"],
    "expected": ["expected result", "expected results", "expected outcome", "expected"],
}

_HEADER_HINTS = (
    "test case",
    "testid",
    "test id",
    "area",
    "feature",
    "steps to reproduce",
    "expected result",
)

_MIN_MATCH = 78
_STATUS_TITLES = {
    "pass",
    "fail",
    "blocked",
    "untested",
    "untestable",
    "not applicable",
    "in-progress",
    "in progress",
}


def _norm(text: str) -> str:
    text = str(text or "").lower().replace("_", " ").replace("-", " ")
    text = text.replace("/", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _best_header(headers: List[str], aliases: List[str], used: set) -> Optional[str]:
    best_header = None
    best_score = 0
    for header in headers:
        if header in used or not str(header).strip():
            continue
        normalized = _norm(header)
        if not normalized:
            continue
        for alias in aliases:
            score = fuzz.token_set_ratio(normalized, alias)
            if normalized == alias:
                score = 100
            if score > best_score:
                best_score = score
                best_header = header
    if best_header is None or best_score < _MIN_MATCH:
        return None
    return best_header


def get_test_plan_column_mapping(headers: List[str]) -> ColumnMapping:
    used: set = set()
    title_column = _best_header(headers, _FIELD_ALIASES["title"], used)
    if title_column:
        used.add(title_column)
    else:
        for header in headers:
            if _norm(header) and "id" not in _norm(header):
                title_column = header
                used.add(header)
                break
        if not title_column and headers:
            title_column = headers[0]
            used.add(title_column)

    id_column = _best_header(headers, _FIELD_ALIASES["id"], used)
    if id_column:
        used.add(id_column)
    area_column = _best_header(headers, _FIELD_ALIASES["area"], used)
    if area_column:
        used.add(area_column)
    steps_column = _best_header(headers, _FIELD_ALIASES["steps"], used)
    if steps_column:
        used.add(steps_column)
    expected_column = _best_header(headers, _FIELD_ALIASES["expected"], used)

    mapping = ColumnMapping(
        id_column=id_column,
        title_column=title_column,
        area_column=area_column,
        steps_column=steps_column,
        expected_result_column=expected_column,
    )
    logging.info("Mapped columns: %s", mapping.model_dump())
    return mapping


def _header_row_score(values: List[str]) -> int:
    cells = [_norm(v) for v in values if _norm(v)]
    if not cells:
        return 0
    hits = 0
    for cell in cells:
        if any(hint in cell or fuzz.token_set_ratio(cell, hint) >= 90 for hint in _HEADER_HINTS):
            hits += 1
    return hits


def _load_table(file_obj: Any) -> pd.DataFrame:
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    raw = pd.read_csv(file_obj, header=None, dtype=str, keep_default_na=False)
    raw = raw.fillna("")
    header_idx = 0
    best = -1
    scan_limit = min(25, len(raw))
    for idx in range(scan_limit):
        score = _header_row_score(raw.iloc[idx].tolist())
        if score > best:
            best = score
            header_idx = idx
    if best < 1:
        raise ValueError("Could not find a test-plan header row (need TEST CASE / Test Case).")

    headers = [str(v).strip() for v in raw.iloc[header_idx].tolist()]
    # Make empty header names unique so pandas doesn't collapse them
    seen = {}
    clean_headers = []
    for i, name in enumerate(headers):
        if not name:
            name = f"_blank_{i}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        clean_headers.append(name)

    df = raw.iloc[header_idx + 1 :].copy()
    df.columns = clean_headers
    df = df.dropna(how="all")
    keep = [c for c in df.columns if not str(c).startswith("_blank_")]
    df = df[keep]
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _split_steps(raw: str) -> List[str]:
    if not raw:
        return []
    parts = re.split(r"[\n|;]+|(?:\s*\d+[\.\)]\s+)", raw)
    return [p.strip(" -•\t") for p in parts if p and p.strip(" -•\t")]


def ingest_and_parse_test_plan_csv(file_obj: Any, model_name: str) -> GeneratedTestPlan:
    logging.info("Starting CSV ingestion for: %s", getattr(file_obj, "name", file_obj))
    df = _load_table(file_obj)
    if df.empty or len(df.columns) == 0:
        raise ValueError("CSV has no usable rows or columns.")

    mapping = get_test_plan_column_mapping(df.columns.tolist())
    if mapping.area_column and mapping.area_column in df.columns:
        df[mapping.area_column] = df[mapping.area_column].replace("", pd.NA).ffill().fillna("")

    test_cases: List[TestCase] = []
    for index, row in df.iterrows():
        def get_col(col_name: Optional[str]) -> str:
            if not col_name or col_name not in row.index:
                return ""
            value = row.get(col_name, "")
            return str(value).strip() if pd.notna(value) else ""

        title = get_col(mapping.title_column)
        if not title or title.lower() in _STATUS_TITLES:
            continue
        if _norm(title) in {"test case", "area feature"}:
            continue

        test_cases.append(
            TestCase(
                id=get_col(mapping.id_column) or f"row_{index + 1}",
                area=get_col(mapping.area_column) or None,
                title=title,
                steps=_split_steps(get_col(mapping.steps_column)),
                expected_result=get_col(mapping.expected_result_column) or None,
            )
        )

    if not test_cases:
        raise ValueError(
            f"No test cases parsed from '{model_name}'. "
            "Stock files need AREA / FEATURE + TEST CASE. "
            "Generated files need Test Case + Steps to Reproduce + Expected Result."
        )

    plan = GeneratedTestPlan(model_name=model_name, test_cases=test_cases)
    logging.info("Parsed %s test cases for '%s'.", len(test_cases), model_name)
    return plan


def unique_areas(plan: GeneratedTestPlan) -> List[str]:
    seen = []
    for case in plan.test_cases:
        area = (case.area or "").strip()
        if area and area not in seen:
            seen.append(area)
    return seen


def filter_plan_by_areas(plan: GeneratedTestPlan, areas: Optional[List[str]]) -> GeneratedTestPlan:
    if not areas:
        return plan
    wanted = set(areas)
    kept = [case for case in plan.test_cases if (case.area or "") in wanted]
    return GeneratedTestPlan(model_name=plan.model_name, test_cases=kept)
