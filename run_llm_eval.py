"""
Run testcases against the local LLM API and persist results incrementally.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_write_json(path: str, payload: Any) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp_path, path)


def _post_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _build_index(existing: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    index: Dict[int, Dict[str, Any]] = {}
    for item in existing:
        case_id = item.get("case_id")
        if isinstance(case_id, int):
            index[case_id] = item
    return index


def run(
    input_path: str,
    output_path: str,
    base_url: str,
    endpoint: str,
    timeout: float,
    sleep_s: float,
    force: bool,
) -> None:
    cases = _load_json(input_path)
    if not isinstance(cases, list):
        raise ValueError("Input testcases must be a JSON list.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    existing: List[Dict[str, Any]] = []
    if os.path.exists(output_path):
        try:
            existing = _load_json(output_path)
            if not isinstance(existing, list):
                existing = []
        except json.JSONDecodeError:
            existing = []

    existing_index = _build_index(existing)
    results: List[Dict[str, Any]] = list(existing)

    total = len(cases)
    completed = 0

    for idx, case in enumerate(cases):
        case_id = idx
        if not force and case_id in existing_index:
            completed += 1
            print(f"{completed}/{total} complete (skipped case_id={case_id})")
            continue

        question = case.get("question")
        expected = case.get("answer")
        payload = {"query": question}

        entry: Dict[str, Any] = {
            "case_id": case_id,
            "question": question,
            "expected_answer": expected,
            "llm_response": None,
            "status": "error",
            "error": None,
        }

        url = base_url.rstrip("/") + "/" + endpoint.lstrip("/")
        try:
            response = _post_json(url, payload, timeout=timeout)
            entry["llm_response"] = response.get("response")
            entry["status"] = "ok"
        except HTTPError as e:
            entry["error"] = f"HTTPError {e.code}: {e.reason}"
        except URLError as e:
            entry["error"] = f"URLError: {e.reason}"
        except Exception as e:
            entry["error"] = f"Exception: {e}"

        if case_id in existing_index:
            for i, item in enumerate(results):
                if item.get("case_id") == case_id:
                    results[i] = entry
                    break
        else:
            results.append(entry)
            existing_index[case_id] = entry

        results_sorted = sorted(
            results,
            key=lambda item: item.get("case_id", 1_000_000_000),
        )
        _safe_write_json(output_path, results_sorted)

        completed += 1
        print(f"{completed}/{total} complete")

        if sleep_s > 0:
            time.sleep(sleep_s)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run testcases against local LLM API and save responses."
    )
    parser.add_argument(
        "--input",
        default="testcases.json",
        help="Path to input testcases JSON.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output JSON.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL for the API.",
    )
    parser.add_argument(
        "--endpoint",
        default="/api/v1/chat",
        help="Endpoint path for chat.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep between requests in seconds.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all cases even if output already has them.",
    )
    return parser.parse_args(argv)


def _default_output_path() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return os.path.join("llmResponces", f"{timestamp}.json")


def main() -> None:
    args = parse_args()
    output_path = args.output or _default_output_path()
    run(
        input_path=args.input,
        output_path=output_path,
        base_url=args.base_url,
        endpoint=args.endpoint,
        timeout=args.timeout,
        sleep_s=args.sleep,
        force=args.force,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
