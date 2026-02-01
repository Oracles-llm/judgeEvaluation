"""
Score LLM responses against expected answers using Vertex AI Gemini.

Output format:
[
  {"case_id": ..., "question": "...", "expected_answer": "...", "llm_response": "...", "score": 0.0},
  ...,
  {"average_score": 0.1234}
]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import random
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple


try:
    from google.auth.transport.requests import AuthorizedSession
    from google.oauth2 import service_account
except Exception:  # pragma: no cover - runtime dependency
    AuthorizedSession = None  # type: ignore[assignment]
    service_account = None  # type: ignore[assignment]


SCOPE = "https://www.googleapis.com/auth/cloud-platform"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_write_json(path: str, payload: Any) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp_path, path)


def _latest_json_file(folder: str) -> Optional[str]:
    if not os.path.isdir(folder):
        return None
    candidates = [
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.lower().endswith(".json")
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _resolve_existing_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return os.path.abspath(path)
    return os.path.join(SCRIPT_DIR, path)


def _extract_average_payload(existing: Any) -> Tuple[List[Dict[str, Any]], bool]:
    if not isinstance(existing, list):
        return [], False
    if not existing:
        return [], False
    last = existing[-1]
    if isinstance(last, dict) and "average_score" in last and len(last) == 1:
        return existing[:-1], True
    return existing, False


def _build_index(existing: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    index: Dict[int, Dict[str, Any]] = {}
    for item in existing:
        case_id = item.get("case_id")
        if isinstance(case_id, int):
            index[case_id] = item
    return index


def _build_prompt(question: str, expected: str, response: str) -> str:
    return (
        "You are an impartial grader. Compare the LLM response to the expected "
        "answer for the given question.\n\n"
        "Scoring:\n"
        "- 1.0 = fully correct and complete.\n"
        "- 0.0 = completely incorrect or unrelated.\n"
        "- Use partial credit for partially correct answers.\n"
        "- Ignore style differences, but penalize contradictions or major omissions.\n\n"
        'Respond with a single-line JSON object: {"score": <number>} and no other text.\n\n'
        f"Question: {question}\n"
        f"Expected Answer: {expected}\n"
        f"LLM Response: {response}\n"
    )


def _build_batch_prompt(items: List[Dict[str, Any]]) -> str:
    lines = [
        "You are an impartial grader. Compare each LLM response to the expected "
        "answer for the given question.",
        "",
        "Scoring:",
        "- 1.0 = fully correct and complete.",
        "- 0.0 = completely incorrect or unrelated.",
        "- Use partial credit for partially correct answers.",
        "- Ignore style differences, but penalize contradictions or major omissions.",
        "",
        "Respond with a single-line JSON array of objects, each like:",
        '{"case_id": <id>, "score": <number>}',
        "Return ONLY this JSON array. Do not add code fences or extra text.",
        "",
        "Cases:",
    ]
    for item in items:
        lines.append(f'case_id: {item["case_id"]}')
        lines.append(f'Question: {item["question"]}')
        lines.append(f'Expected Answer: {item["expected_answer"]}')
        lines.append(f'LLM Response: {item["llm_response"]}')
        lines.append("")
    return "\n".join(lines)


def _parse_score(text: str) -> Optional[float]:
    if not text:
        return None
    cleaned = text.strip()
    # Remove code fences if present.
    cleaned = re.sub(r"^```[a-zA-Z0-9]*\\s*|```$", "", cleaned).strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and "score" in obj:
            return float(obj["score"])
        if isinstance(obj, list) and obj:
            first = obj[0]
            if isinstance(first, dict) and "score" in first:
                return float(first["score"])
            if isinstance(first, (int, float)):
                return float(first)
    except Exception:
        pass
    match = re.search(r'"?score"?\\s*:\\s*([0-9]*\\.?[0-9]+)', cleaned)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _parse_scores(text: str) -> Optional[List[Dict[str, Any]]]:
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```[a-zA-Z0-9]*\\s*|```$", "", cleaned).strip()
    candidates: List[str] = [cleaned]
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidates.append(cleaned[start : end + 1])
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict) and isinstance(obj.get("scores"), list):
                return obj["scores"]
        except Exception:
            continue
    extracted = _extract_score_objects(cleaned)
    if extracted:
        return extracted
    return None


def _extract_score_objects(text: str) -> List[Dict[str, Any]]:
    objects: List[Dict[str, Any]] = []
    start = None
    depth = 0
    in_string = False
    escape = False

    for idx, ch in enumerate(text):
        if start is None:
            if ch == "{":
                start = idx
                depth = 1
                in_string = False
                escape = False
            continue

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_string = False
            continue

        if ch == "\"":
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                snippet = text[start : idx + 1]
                start = None
                try:
                    obj = json.loads(snippet)
                    if isinstance(obj, dict) and "score" in obj:
                        objects.append(obj)
                except Exception:
                    continue
    return objects


class VertexJudge:
    def __init__(
        self,
        service_account_path: str,
        project_id: str,
        location: str,
        model: str,
        timeout: float,
        retries: int,
        min_interval: float,
    ) -> None:
        if service_account is None or AuthorizedSession is None:
            raise RuntimeError(
                "Missing google-auth dependencies. Install with: pip install google-auth"
            )
        creds = service_account.Credentials.from_service_account_file(
            service_account_path,
            scopes=[SCOPE],
        )
        self.session = AuthorizedSession(creds)
        self.timeout = timeout
        self.retries = max(0, int(retries))
        self.min_interval = max(0.0, float(min_interval))
        self._last_request_ts = 0.0
        self.use_json_mode = True
        self.url = (
            f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}"
            f"/locations/{location}/publishers/google/models/{model}:generateContent"
        )

    def score(self, prompt: str) -> str:
        if self.min_interval > 0:
            now = time.monotonic()
            wait_for = self.min_interval - (now - self._last_request_ts)
            if wait_for > 0:
                time.sleep(wait_for)
            self._last_request_ts = time.monotonic()

        def build_payload() -> Dict[str, Any]:
            payload_local: Dict[str, Any] = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": 1024,
                    "topP": 1.0,
                    "topK": 1,
                },
            }
            if self.use_json_mode:
                payload_local["generationConfig"]["responseMimeType"] = "application/json"
            return payload_local

        payload = build_payload()
        attempt = 0
        while True:
            resp = self.session.post(self.url, json=payload, timeout=self.timeout)
            if resp.status_code == 400 and self.use_json_mode:
                # Fall back if responseMimeType is not supported for this model.
                self.use_json_mode = False
                payload = build_payload()
                continue
            if resp.status_code in {429, 500, 503} and attempt < self.retries:
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    time.sleep(int(retry_after))
                else:
                    time.sleep((2**attempt) + random.random())
                attempt += 1
                continue
            resp.raise_for_status()
            break
        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        text = ""
        for part in parts:
            text += part.get("text", "")
        return text.strip()


def _clamp_score(score: float) -> float:
    if score < 0:
        return 0.0
    if score > 1:
        return 1.0
    return float(score)


def _derive_output_path(input_path: str) -> str:
    filename = os.path.basename(input_path)
    return os.path.join("scores", filename)


def run(
    input_path: str,
    output_path: str,
    service_account_path: str,
    project_id: str,
    location: str,
    model: str,
    timeout: float,
    sleep_s: float,
    max_cases: Optional[int],
    batch_size: int,
    retries: int,
    min_interval: float,
    force: bool,
) -> None:
    cases = _load_json(input_path)
    if not isinstance(cases, list):
        raise ValueError("Input file must be a JSON list.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    existing_results: List[Dict[str, Any]] = []
    if os.path.exists(output_path):
        try:
            existing_payload = _load_json(output_path)
            existing_results, _ = _extract_average_payload(existing_payload)
        except json.JSONDecodeError:
            existing_results = []

    existing_index = _build_index(existing_results)
    results: List[Dict[str, Any]] = list(existing_results)

    judge = VertexJudge(
        service_account_path=service_account_path,
        project_id=project_id,
        location=location,
        model=model,
        timeout=timeout,
        retries=retries,
        min_interval=min_interval,
    )

    total = len(cases) if max_cases is None else min(max_cases, len(cases))
    completed = 0

    batch_size = max(1, int(batch_size))
    batch: List[Dict[str, Any]] = []
    batch_case_ids: List[int] = []

    def flush_batch() -> None:
        nonlocal completed
        if not batch:
            return
        prompt = _build_batch_prompt(batch)
        parsed_scores: Dict[int, float] = {}
        try:
            raw = judge.score(prompt)
            parsed_list = _parse_scores(raw)
            if parsed_list is None:
                raise ValueError(f"Could not parse scores from: {raw!r}")
            if all(isinstance(item, dict) and "case_id" in item for item in parsed_list):
                for item in parsed_list:
                    if not isinstance(item, dict):
                        continue
                    cid = item.get("case_id")
                    score = item.get("score")
                    if isinstance(cid, int) and isinstance(score, (int, float)):
                        parsed_scores[cid] = _clamp_score(float(score))
            else:
                if len(parsed_list) == len(batch):
                    for idx, item in enumerate(parsed_list):
                        score: Optional[float] = None
                        if isinstance(item, dict) and "score" in item:
                            if isinstance(item["score"], (int, float)):
                                score = float(item["score"])
                        elif isinstance(item, (int, float)):
                            score = float(item)
                        if score is not None:
                            parsed_scores[batch[idx]["case_id"]] = _clamp_score(score)
        except Exception as exc:
            print(f"batch error: {exc}", file=sys.stderr)

        missing = [item for item in batch if item["case_id"] not in parsed_scores]
        if missing:
            for item in missing:
                case_id = item["case_id"]
                try:
                    raw = judge.score(
                        _build_prompt(
                            item["question"],
                            item["expected_answer"],
                            item["llm_response"],
                        )
                    )
                    parsed = _parse_score(raw)
                    if parsed is None:
                        raise ValueError(f"Could not parse score from: {raw!r}")
                    parsed_scores[case_id] = _clamp_score(parsed)
                except Exception as exc:
                    print(f"case_id={case_id} error: {exc}", file=sys.stderr)
                    parsed_scores[case_id] = 0.0

        for item in batch:
            case_id = item["case_id"]
            entry: Dict[str, Any] = {
                "case_id": case_id,
                "question": item["question"],
                "expected_answer": item["expected_answer"],
                "llm_response": item["llm_response"],
                "score": parsed_scores.get(case_id, 0.0),
            }

            if case_id in existing_index:
                for i, existing_item in enumerate(results):
                    if existing_item.get("case_id") == case_id:
                        results[i] = entry
                        break
            else:
                results.append(entry)
                existing_index[case_id] = entry

            completed += 1
            print(f"{completed}/{total} complete")

        results_sorted = sorted(
            results,
            key=lambda item: item.get("case_id", 1_000_000_000),
        )
        scores = [
            r.get("score")
            for r in results_sorted
            if isinstance(r.get("score"), (int, float))
        ]
        avg = sum(scores) / len(scores) if scores else 0.0
        payload = list(results_sorted) + [{"average_score": avg}]
        _safe_write_json(output_path, payload)

        if sleep_s > 0:
            time.sleep(sleep_s)

    for idx, case in enumerate(cases[:total]):
        case_id = case.get("case_id", idx)
        if not isinstance(case_id, int):
            case_id = idx

        if not force and case_id in existing_index:
            completed += 1
            print(f"{completed}/{total} complete (skipped case_id={case_id})")
            continue

        question = str(case.get("question", "") or "")
        expected = str(case.get("expected_answer", "") or "")
        response = str(case.get("llm_response", "") or "")

        if not response.strip() or not expected.strip():
            entry: Dict[str, Any] = {
                "case_id": case_id,
                "question": question,
                "expected_answer": expected,
                "llm_response": response,
                "score": 0.0,
            }
            if case_id in existing_index:
                for i, existing_item in enumerate(results):
                    if existing_item.get("case_id") == case_id:
                        results[i] = entry
                        break
            else:
                results.append(entry)
                existing_index[case_id] = entry
            completed += 1
            print(f"{completed}/{total} complete")
            continue

        batch.append(
            {
                "case_id": case_id,
                "question": question,
                "expected_answer": expected,
                "llm_response": response,
            }
        )
        batch_case_ids.append(case_id)

        if len(batch) >= batch_size:
            flush_batch()
            batch = []
            batch_case_ids = []

    if batch:
        flush_batch()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score LLM responses using Vertex AI Gemini."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to input responses JSON.",
    )
    parser.add_argument(
        "--file",
        dest="file_name",
        default=None,
        help="File name inside llmResponces to score (e.g. 20260201_2130.json).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output scores JSON (default: scores/<input filename>).",
    )
    parser.add_argument(
        "--service-account",
        default=os.path.join(SCRIPT_DIR, "chatbot-sa.json"),
        help="Path to GCP service account JSON.",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="GCP project ID (defaults to service account project_id).",
    )
    parser.add_argument(
        "--location",
        default="us-central1",
        help="Vertex AI location.",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Vertex AI Gemini model name.",
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
        "--max-cases",
        type=int,
        default=None,
        help="Maximum number of cases to score.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of cases per API request.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry attempts for 429/5xx responses.",
    )
    parser.add_argument(
        "--min-interval",
        type=float,
        default=0.5,
        help="Minimum seconds between API requests.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-score all cases even if output already has them.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.input and args.file_name:
        raise ValueError("Use either --input or --file, not both.")

    input_path = None
    if args.file_name:
        input_path = os.path.join(SCRIPT_DIR, "llmResponces", args.file_name)
    elif args.input:
        input_path = _resolve_existing_path(args.input)
    else:
        raise ValueError(
            "Input file required. Use --file <name> for llmResponces or --input <path>."
        )

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not args.output:
        output_path = _derive_output_path(input_path)
        if not os.path.isabs(output_path):
            output_path = os.path.join(SCRIPT_DIR, output_path)
    else:
        output_path = args.output

    service_account_path = args.service_account
    if not os.path.isabs(service_account_path):
        service_account_path = _resolve_existing_path(service_account_path)

    project_id = args.project
    if not project_id:
        service_data = _load_json(service_account_path)
        project_id = service_data.get("project_id")
    if not project_id:
        raise ValueError("Project ID is required (use --project or service account project_id).")

    run(
        input_path=input_path,
        output_path=output_path,
        service_account_path=service_account_path,
        project_id=project_id,
        location=args.location,
        model=args.model,
        timeout=args.timeout,
        sleep_s=args.sleep,
        max_cases=args.max_cases,
        batch_size=args.batch_size,
        retries=args.retries,
        min_interval=args.min_interval,
        force=args.force,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
