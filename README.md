Score LLM responses with Vertex Gemini

Prereqs
- Python 3.9+
- `google-auth` installed: `pip install google-auth`
- GCP service account JSON at `judgeEvaluation/chatbot-sa.json` (or pass `--service-account`)

Usage (required)
- You must specify which file to score using one of:
  - `--file <name>` for a file inside `judgeEvaluation/llmResponces`
  - `--input <path>` for an explicit path

Examples
```
# From repo root or judgeEvaluation/
python score_llm_responses.py --file 20260201_2130.json

# Explicit path
python score_llm_responses.py --input judgeEvaluation/llmResponces/20260201_2130.json
```

Options
- `--batch-size` (default 10): number of cases per API call
- `--min-interval` (default 0.5): minimum seconds between API requests
- `--retries` (default 3): retry attempts for 429/5xx responses
- `--model` (default `gemini-2.5-flash`)
- `--location` (default `us-central1`)
- `--project` (optional; otherwise read from service account)
- `--output` (default `judgeEvaluation/scores/<input filename>`)
 - `--force`: re-score all cases even if output already has them

Output format
- A JSON list of per-case objects with `case_id`, `question`, `expected_answer`, `llm_response`, `score`
- Final entry `{ "average_score": <number> }`
