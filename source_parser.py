"""
Extract testable intents from the document used to generate test cases.

Supports CSV (requirement/user-story style rows) and free-text .txt/.md.
No API keys.
"""

import logging
import re
from typing import Any, List, Optional

import pandas as pd
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class SourceIntent(BaseModel):
    """A unit of product intent from a source spec or an overall-intent summary."""
    id: str
    text: str
    origin: str = Field("source", description="overall_intent or source_document")

_MIN_CHUNK = 24
_MAX_INTENTS = 200
_SKIP_PREFIXES = (
    "table of contents",
    "revision history",
    "confidential",
    "copyright",
    "page ",
)


def _clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def intents_from_overall_text(text: str) -> List[SourceIntent]:
    text = _clean(text or "")
    if not text:
        return []
    parts = _split_prose(text)
    if not parts:
        parts = [text]
    intents = [
        SourceIntent(id=f"intent_{i + 1}", text=part, origin="overall_intent")
        for i, part in enumerate(parts)
    ]
    return intents[:_MAX_INTENTS]


def ingest_source_document(file_obj: Any) -> List[SourceIntent]:
    name = str(getattr(file_obj, "name", "") or "").lower()
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)

    if name.endswith(".csv"):
        intents = _from_csv(file_obj)
    else:
        raw = file_obj.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        if re.search(r"(?im)^mission\s+\d", raw):
            intents = _from_gdd_outline(raw)
        else:
            intents = _from_prose(raw, origin="source_document")

    logging.info("Parsed %s source intents from %s", len(intents), getattr(file_obj, "name", "source"))
    if not intents:
        raise ValueError("Could not extract any requirements/intents from the source document.")
    return intents


def merge_source_intents(
    overall_text: Optional[str], source_file: Any
) -> List[SourceIntent]:
    intents: List[SourceIntent] = []
    overall = _clean(overall_text or "")
    if overall:
        split = intents_from_overall_text(overall)
        intents.extend(split if split else [
            SourceIntent(id="intent_1", text=overall, origin="overall_intent")
        ])
    if source_file is not None:
        intents.extend(ingest_source_document(source_file))
    kept: List[SourceIntent] = []
    for item in intents:
        if item.origin == "overall_intent" or len(item.text) >= _MIN_CHUNK:
            kept.append(item)
        if len(kept) >= _MAX_INTENTS * 2:
            break
    return kept


def _from_gdd_outline(text: str) -> List[SourceIntent]:
    """Parse numbered GDD beats, tagged by Mission heading (e.g. Mission 1.1)."""
    sections = re.split(r"(?im)(?=^mission\s+\d)", text)
    intents: List[SourceIntent] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        heading = section.splitlines()[0].strip().lstrip("\ufeff")
        origin = heading if heading.lower().startswith("mission") else "GDD"
        for line in section.splitlines():
            match = re.match(r"^\s*\d+\.\s+(.*\S.*)$", line)
            if not match:
                continue
            body = _clean(match.group(1))
            if len(body) < 8:
                continue
            label = f"{heading}: {body}" if heading.lower().startswith("mission") else body
            intents.append(
                SourceIntent(id=f"gdd_{len(intents) + 1}", text=label, origin=origin)
            )
            if len(intents) >= _MAX_INTENTS:
                return intents
    return intents


def _from_csv(file_obj: Any) -> List[SourceIntent]:
    df = pd.read_csv(file_obj)
    df = df.dropna(how="all").fillna("")
    df.columns = df.columns.str.strip()
    if df.empty:
        return []

    preferred = [
        col
        for col in df.columns
        if re.search(
            r"user.?story|requirement|description|summary|objective|acceptance|criteria|spec|feature|intent|scenario",
            col,
            re.I,
        )
    ]
    text_cols = preferred or [
        col for col in df.columns if df[col].dtype == object or str(df[col].dtype).startswith("string")
    ]
    if not text_cols:
        text_cols = list(df.columns)

    intents: List[SourceIntent] = []
    for index, row in df.iterrows():
        pieces = []
        for col in text_cols:
            value = str(row.get(col, "")).strip()
            if value and value.lower() not in {"nan", "none"}:
                pieces.append(value)
        blob = " — ".join(pieces) if pieces else ""
        if len(blob) < _MIN_CHUNK:
            continue
        intents.append(
            SourceIntent(id=f"src_row_{index + 1}", text=blob, origin="source_document")
        )
        if len(intents) >= _MAX_INTENTS:
            break
    return intents


def _from_prose(text: str, origin: str) -> List[SourceIntent]:
    text = _clean(text)
    chunks = _split_prose(text)
    intents: List[SourceIntent] = []
    for i, chunk in enumerate(chunks):
        lowered = chunk.lower()
        if any(lowered.startswith(prefix) for prefix in _SKIP_PREFIXES):
            continue
        intents.append(SourceIntent(id=f"src_{i + 1}", text=chunk, origin=origin))
        if len(intents) >= _MAX_INTENTS:
            break
    return intents


def _split_prose(text: str) -> List[str]:
    blocks = re.split(r"\n(?=#{1,6}\s|[-*•]\s|\d+[.)]\s)", text)
    chunks: List[str] = []
    for block in blocks:
        block = _clean(block)
        if not block:
            continue
        if len(block) <= 400:
            if len(block) >= _MIN_CHUNK:
                chunks.append(re.sub(r"^#{1,6}\s+", "", block))
            continue
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", block)
        buf = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(buf) + len(sentence) < 320:
                buf = f"{buf} {sentence}".strip()
            else:
                if len(buf) >= _MIN_CHUNK:
                    chunks.append(buf)
                buf = sentence
        if len(buf) >= _MIN_CHUNK:
            chunks.append(buf)
    return chunks
