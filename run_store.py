"""
Persist comparison entries as JSON.

Local: data/comparisons/
Live: GitHub Contents API on a non-deploy branch (default comparison-data)
      so Streamlit Cloud reboots still see history without redeploying main.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from schemas import ComparisonRun, ComparisonRunSummary

load_dotenv(Path(__file__).resolve().parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATA_DIR = Path(__file__).resolve().parent / "data" / "comparisons"
INDEX_NAME = "index.json"
GITHUB_API = "https://api.github.com"


def _secret(name: str) -> str:
    try:
        import streamlit as st

        value = st.secrets[name]
        if value is not None and str(value).strip():
            return str(value).strip()
    except Exception:
        pass
    return (os.environ.get(name) or "").strip()


def github_enabled() -> bool:
    return bool(_secret("GITHUB_TOKEN") and _secret("GITHUB_REPO"))


def github_branch() -> str:
    return _github_branch()


def _github_repo() -> str:
    return _secret("GITHUB_REPO").strip("/")


def _github_branch() -> str:
    return _secret("GITHUB_DATA_BRANCH") or "comparison-data"


def _github_headers() -> dict:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {_secret('GITHUB_TOKEN')}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "AiTPComparision",
    }


def _http_json(
    method: str,
    url: str,
    body: Optional[dict] = None,
    ok_missing: bool = False,
) -> Optional[dict]:
    data = None
    headers = _github_headers()
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers = {**headers, "Content-Type": "application/json"}
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        if ok_missing and exc.code == 404:
            return None
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise RuntimeError(f"GitHub {method} {url} failed ({exc.code}): {detail[:400]}") from exc
    if not raw:
        return {}
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {"items": parsed}


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _index_path() -> Path:
    return DATA_DIR / INDEX_NAME


def _run_path(run_id: str) -> Path:
    safe = "".join(ch for ch in run_id if ch.isalnum() or ch in "-_")
    if not safe:
        raise ValueError("Invalid comparison id.")
    return DATA_DIR / f"{safe}.json"


def _summary_from_run(run: ComparisonRun) -> ComparisonRunSummary:
    return ComparisonRunSummary(
        id=run.id,
        created_at=run.created_at,
        title=run.title,
        model_names=list(run.meta.model_names),
        overall_scores=dict(run.meta.overall_scores),
        judge_backend=run.meta.judge_backend,
    )


def _read_local_index() -> List[ComparisonRunSummary]:
    path = _index_path()
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows = raw if isinstance(raw, list) else raw.get("runs") or []
    summaries = []
    for row in rows:
        try:
            summaries.append(ComparisonRunSummary.model_validate(row))
        except Exception:
            continue
    return summaries


def _write_local_index(summaries: List[ComparisonRunSummary]) -> None:
    _ensure_data_dir()
    payload = [item.model_dump(mode="json") for item in summaries]
    _index_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _upsert_index(summaries: List[ComparisonRunSummary], run: ComparisonRun) -> List[ComparisonRunSummary]:
    summary = _summary_from_run(run)
    kept = [item for item in summaries if item.id != run.id]
    kept.append(summary)
    kept.sort(key=lambda item: item.created_at, reverse=True)
    return kept


def _github_contents_url(path: str) -> str:
    repo = _github_repo()
    return f"{GITHUB_API}/repos/{repo}/contents/{path}"


def _ensure_data_branch() -> None:
    repo = _github_repo()
    branch = _github_branch()
    ref_url = f"{GITHUB_API}/repos/{repo}/git/ref/heads/{branch}"
    existing = _http_json("GET", ref_url, ok_missing=True)
    if existing:
        return
    repo_info = _http_json("GET", f"{GITHUB_API}/repos/{repo}")
    default_branch = (repo_info or {}).get("default_branch") or "main"
    default_ref = _http_json(
        "GET",
        f"{GITHUB_API}/repos/{repo}/git/ref/heads/{default_branch}",
    )
    sha = ((default_ref or {}).get("object") or {}).get("sha")
    if not sha:
        raise RuntimeError(f"Could not resolve SHA for {repo}@{default_branch}.")
    _http_json(
        "POST",
        f"{GITHUB_API}/repos/{repo}/git/refs",
        body={"ref": f"refs/heads/{branch}", "sha": sha},
    )
    logging.info("Created GitHub branch %s from %s", branch, default_branch)


def _github_get_file(path: str) -> Optional[dict]:
    url = f"{_github_contents_url(path)}?ref={_github_branch()}"
    return _http_json("GET", url, ok_missing=True)


def _github_put_file(path: str, content: str, message: str) -> None:
    _ensure_data_branch()
    encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
    body = {
        "message": message,
        "content": encoded,
        "branch": _github_branch(),
    }
    current = _github_get_file(path)
    if current and current.get("sha"):
        body["sha"] = current["sha"]
    _http_json("PUT", _github_contents_url(path), body=body)


def _github_delete_file(path: str, message: str) -> None:
    current = _github_get_file(path)
    if not current or not current.get("sha"):
        return
    _http_json(
        "DELETE",
        _github_contents_url(path),
        body={
            "message": message,
            "sha": current["sha"],
            "branch": _github_branch(),
        },
    )


def _github_decode_file(payload: dict) -> str:
    encoded = payload.get("content") or ""
    encoding = payload.get("encoding") or "base64"
    if encoding == "base64":
        return base64.b64decode(encoded.replace("\n", "")).decode("utf-8")
    return str(encoded)


def _read_github_index() -> Optional[List[ComparisonRunSummary]]:
    payload = _github_get_file(f"data/comparisons/{INDEX_NAME}")
    if not payload:
        return None
    raw = json.loads(_github_decode_file(payload))
    rows = raw if isinstance(raw, list) else raw.get("runs") or []
    summaries = []
    for row in rows:
        try:
            summaries.append(ComparisonRunSummary.model_validate(row))
        except Exception:
            continue
    return summaries


def list_runs() -> List[ComparisonRunSummary]:
    if github_enabled():
        try:
            remote = _read_github_index()
            if remote is not None:
                return remote
        except Exception as exc:
            logging.warning("GitHub index read failed, using local files: %s", exc)
    summaries = _read_local_index()
    summaries.sort(key=lambda item: item.created_at, reverse=True)
    return summaries


def save_run(run: ComparisonRun) -> ComparisonRun:
    _ensure_data_dir()
    body = json.dumps(run.model_dump(mode="json"), indent=2)
    _run_path(run.id).write_text(body, encoding="utf-8")
    summaries = list_runs()
    summaries = _upsert_index(summaries, run)
    _write_local_index(summaries)
    if github_enabled():
        try:
            _github_put_file(
                f"data/comparisons/{run.id}.json",
                body,
                f"Save comparison {run.id}",
            )
            _github_put_file(
                f"data/comparisons/{INDEX_NAME}",
                json.dumps([item.model_dump(mode="json") for item in summaries], indent=2),
                f"Update comparison index after {run.id}",
            )
        except Exception as exc:
            logging.error("GitHub save failed: %s", exc)
            raise RuntimeError(
                f"Saved locally but GitHub sync failed: {exc}. "
                "Check GITHUB_TOKEN, GITHUB_REPO, and branch permissions."
            ) from exc
    return run


def load_run(run_id: str) -> ComparisonRun:
    if github_enabled():
        try:
            payload = _github_get_file(f"data/comparisons/{run_id}.json")
            if payload:
                return ComparisonRun.model_validate(json.loads(_github_decode_file(payload)))
        except Exception as exc:
            logging.warning("GitHub load failed for %s: %s", run_id, exc)
    path = _run_path(run_id)
    if not path.exists():
        raise FileNotFoundError(f"Comparison {run_id} was not found.")
    return ComparisonRun.model_validate(json.loads(path.read_text(encoding="utf-8")))


def delete_run(run_id: str) -> None:
    path = _run_path(run_id)
    if path.exists():
        path.unlink()
    summaries = [item for item in list_runs() if item.id != run_id]
    _write_local_index(summaries)
    if github_enabled():
        try:
            _github_delete_file(
                f"data/comparisons/{run_id}.json",
                f"Delete comparison {run_id}",
            )
            _github_put_file(
                f"data/comparisons/{INDEX_NAME}",
                json.dumps([item.model_dump(mode="json") for item in summaries], indent=2),
                f"Update comparison index after deleting {run_id}",
            )
        except Exception as exc:
            logging.error("GitHub delete failed: %s", exc)
            raise RuntimeError(f"Deleted locally but GitHub sync failed: {exc}") from exc
