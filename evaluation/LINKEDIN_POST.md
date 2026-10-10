I tested my offline RAG assistant instead of trusting it. Here is what I found.

The project answers questions from a 420-page military vehicle manual, fully offline, using hybrid search (BM25 + embeddings with Reciprocal Rank Fusion) and a local Llama 3.2 3B model. Every answer has to cite manual pages.

I built an evaluation suite with 63 questions and 205 automated checks across retrieval, generation, citations, safety, API and robustness. No paid API and no LLM judge, so a re-run costs nothing.

What the data showed:

1. Hybrid search beats either half alone: Hit@1 0.85 vs 0.77 (embeddings) and 0.77 (BM25).
2. Guardrails matter. Raw model output had 93% claim-level citation support. After my code-level guards it was 100%, with zero fabricated page numbers.
3. The tests found real bugs: a prompt-injection hole ("say PWNED" got through), correct facts being silently deleted, and an API that returned 500 for bad input.

After fixing them, checks went from 186/205 to 197/205. Injection resistance 2/3 to 3/3, out-of-scope refusal 2/5 to 5/5.

Limits, stated plainly: a 63-question set is small, only 22 questions go through the model, and I tuned the fixes on the same questions, so a held-out set is next. 8 retrieval checks still fail (rows from neighbouring problems leak into some answers).

Code, golden set and full report on GitHub: https://github.com/Shiva123-del/offline-rag-field-assistant

#RAG #LLM #GenerativeAI #AIEngineering #Evaluation #Ollama #Python
