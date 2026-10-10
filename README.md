# Offline RAG Field Assistant

An **offline** question-answering assistant for equipment manuals. It answers from a 420-page US Army HMMWV operator's manual (TM 9-2320-280-10) using hybrid retrieval and a **local** LLM, and every answer must cite manual pages. No internet and no paid API are needed once the models are downloaded.

![Evaluation results](evaluation/assets/results_summary.png)

## How it works

1. **Ingest** (`src/ingest.py`): extracts the PDF with PyMuPDF, splits text into overlapping chunks, and reads the troubleshooting table as structured rows (problem, cause, action) so each cause stays paired with its action.
2. **Hybrid retrieval** (`src/retrieval.py`): BM25 keyword search plus `all-MiniLM-L6-v2` embeddings in ChromaDB, merged with Reciprocal Rank Fusion. Problem-style questions are routed to the troubleshooting table.
3. **Grounded generation** (`src/rag.py`): a local Ollama `llama3.2` (3B) model answers in three fixed sections with `[Manual page N]` citations.
4. **Guardrails in code**: invented quotes, fake page numbers, paraphrased table rows and unsupported claims are removed or repaired after generation, and answers that ignore the format cannot pass through.
5. **Local web UI** (`src/app.py`): standard-library HTTP server at `http://127.0.0.1:8000`.

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

ollama pull llama3.2                 # local model
HF_HUB_OFFLINE=0 python src/ingest.py   # first run only: downloads the embedding model, builds data/vector_db
python src/app.py                    # open http://127.0.0.1:8000
```

Ask from the command line instead: `python src/rag.py "Why does the engine crank but not start?"`

## Evaluation

The project ships with a repeatable evaluation suite in [`evaluation/`](evaluation/): **63 golden questions, 205 automated checks**, covering retrieval, generation, claim-level citations, prompt-injection and out-of-scope safety, API behaviour, robustness and offline start. Scoring is deterministic code (no LLM judge) and answers are cached, so re-runs are free.

| Test | Result |
|---|---|
| Retrieval (hybrid, Hit@1 / MRR@10) | 0.85 / 0.895 (BM25-only and embeddings-only: 0.77 Hit@1) |
| Generation | 22 / 22 |
| Citation (every claim cited, real pages, page backs claim) | 22 / 22, 100% claim support |
| Prompt-injection / out-of-scope | 3 / 3 and 5 / 5 |
| Overall | **197 / 205** (186 / 205 before the fixes the evaluation led to) |

The evaluation found and led to fixes for a prompt-injection bypass, correct facts being silently dropped, and API errors on malformed input. Known limits: a small golden set, fixes tuned on the same questions they are scored on, and 8 retrieval-ranking checks still failing (rows from neighbouring problems leak into some answers).

- Summary: [`evaluation/EVALUATION_SUMMARY.md`](evaluation/EVALUATION_SUMMARY.md)
- Full report (open in a browser): [`evaluation/reports/latest/report.html`](evaluation/reports/latest/report.html)
- Run it yourself: `HF_HUB_OFFLINE=1 python evaluation/run_evaluation.py` (add `--no-llm` for a free 10-second run)

## Project layout

```
src/          ingest, retrieval, rag (generation + guardrails), web app
tests/        original regression tests
evaluation/   golden set, runner, report builder, reports, charts
data/manuals/ source PDFs
```

## Notes

- The vector database is not committed; rebuild it with `python src/ingest.py`.
- `TM 9-2320-280-10` is a US Army publication. The diesel generator manual is included only to test the assistant's coverage boundary (it is deliberately not indexed); remove it if you are not able to redistribute it.
