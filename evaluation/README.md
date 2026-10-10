# Evaluation suite: Offline Field Assistant

A repeatable test and evaluation harness for the offline RAG assistant (HMMWV manual, hybrid BM25 + embedding retrieval, local Ollama `llama3.2`).

## Run it

From the project root:

```bash
# Everything (first run generates answers with Ollama; later runs use the cache)
HF_HUB_OFFLINE=1 venv/bin/python evaluation/run_evaluation.py

# Fast and free: no language model involved (about 10 seconds)
HF_HUB_OFFLINE=1 venv/bin/python evaluation/run_evaluation.py --no-llm

# Pick tiers
HF_HUB_OFFLINE=1 venv/bin/python evaluation/run_evaluation.py --tiers retrieval,guards

# Force fresh model answers (ignore cache)
HF_HUB_OFFLINE=1 venv/bin/python evaluation/run_evaluation.py --refresh
```

Reports land in `evaluation/reports/<timestamp>/` and are copied to `evaluation/reports/latest/`:

| File | What it is |
|---|---|
| `report.md` / `report.html` | The formal report: summary, methodology, retrieval test, generation test, citation test, safety, API, performance, failures, analysis. Open `report.html` in a browser (Print > Save as PDF for a PDF) |
| `results.json` | Every check, rank and raw/final answer (for diffing between runs) |
| `cases.csv` | One row per check, easy to open in Excel |

## Cost control (nothing here uses a paid API)

* The language model is local Ollama, so there are **no API credits** involved at all.
* Tiers 0-5 never call the model.
* Only the stratified subset marked `"llm": true` in `golden_set.json` is generated (22 of 63 cases, covering every category).
* Every generated answer is **cached** in `evaluation/cache/`, keyed by the model, the prompt code and the exact retrieved context. Re-running costs nothing until you change the prompt or retrieval, and then only the affected cases are regenerated.
* Scoring is **deterministic code**, not an LLM judge: a 3B local judge is unreliable and would double the cost.

## Tiers

| Tier | What it tests | Needs Ollama |
|---|---|:-:|
| `validate` | Every expectation in the golden set really exists in the index (no wrong ground truth) | no |
| `retrieval` | The chunks that would be sent to the model: right troubleshooting group, complete rows, no rows from other problems, right page/phrase for facts. Plus an **ablation**: hybrid vs semantic-only vs keyword-only (Hit@k, MRR) | no |
| `guards` | Post-processing safety code: citation normalisation, invented quotes, fake pages, paraphrased table rows, unsupported bullets | no |
| `robustness` | Empty, huge, unicode, SQL/XSS-looking and null-byte inputs; determinism | no |
| `api` | HTTP server (status codes, JSON shape, bad input, 6 concurrent requests) with the model stubbed out | no |
| `persistence` | Fresh processes load the index offline and agree with each other | no |
| `generation` | Real answers: 3 required sections, numbers present in the excerpts, exact table rows, expected facts, refusal on out-of-scope, resistance to prompt injection | yes (cached) |
| `citation` | Claim-level citation audit: coverage, page validity, claim support (cited page really backs the claim), gold-page hit; raw model vs final | no (reads cache) |
| `stability` | Same prompt twice gives the same answer (temperature 0) | yes |

## Golden set (`golden_set.json`)

63 cases: 25 whole-problem troubleshooting, 2 specific-cause, 16 factual lookups, 8 paraphrase/typo/keyword-only, 2 safety, 7 out-of-scope (including a **coverage-boundary** question about the generator manual that is not indexed), 3 prompt-injection. Plus 18 edge inputs.

To add a case, append to `cases` with an `expect` block:

* `group_prefix` + `rows` / `rows_min` / `rows_max` for troubleshooting-table questions
* `pages` and/or `any_of` (phrases that must appear in the matching chunk) for factual questions
* `answer_any` for phrases the final answer must contain
* `"llm": true` to include it in the generation tier

`validate` will tell you immediately if an expectation does not exist in the index.

## Reading the results

A failing check is not automatically a bug in the assistant. Check whether the expectation is fair, then decide whether to fix retrieval/prompting or adjust the case. The report's **Failures** section prints what was expected and what was retrieved.

## Shareable summary

`python3 evaluation/make_summary.py` rebuilds `EVALUATION_SUMMARY.md` (GitHub-ready), `LINKEDIN_POST.md` and `assets/results_summary.png` from the latest run.
