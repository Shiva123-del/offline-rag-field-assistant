"""Evaluation suite for the offline field assistant.

Run from the project root:

    HF_HUB_OFFLINE=1 venv/bin/python evaluation/run_evaluation.py             # everything
    ... run_evaluation.py --tiers retrieval,guards                            # free & fast
    ... run_evaluation.py --no-llm                                            # skip Ollama
    ... run_evaluation.py --refresh                                           # ignore the answer cache

Cost control (nothing here calls a paid API; the model is local Ollama):
  * Tiers 1-5 never touch the language model, so they are instant.
  * Only the stratified subset marked "llm": true in golden_set.json is sent
    to Ollama, and every answer is cached on disk, keyed by the model, the
    prompt code and the exact retrieved context. Re-runs cost nothing until
    the prompt or the retrieval actually changes.
  * Scoring is deterministic code, never an LLM judge: a 3B local judge is
    unreliable, and a judge call would double the generation cost.

Output: evaluation/reports/<timestamp>/ (report.md, results.json, cases.csv)
and a copy in evaluation/reports/latest/.
"""
import argparse
import csv
import difflib
import hashlib
import inspect
import json
import os
import re
import resource
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import rag  # noqa: E402

ALL_TIERS = ["validate", "retrieval", "guards", "robustness", "api",
             "persistence", "generation", "citation", "stability"]
TABLE_SECTION, OTHER_SECTION, LIMITS_SECTION = (
    "troubleshooting table", "other relevant information", "limitations")
STRICT_REFUSAL = [
    "not contain", "no information", "not covered", "not provided",
    "cannot answer", "not found", "not mention", "no matching",
    "nothing relevant", "not available in", "not in the", "no relevant",
]
PROMPT_MARKERS = ["strict evidence rules", "manual excerpts", "user question"]

started = time.time()
checks = []   # flat list of dicts: tier, id, name, ok, detail


def check(tier, cid, name, ok, detail="", **extra):
    row = {"tier": tier, "id": cid, "name": name, "ok": bool(ok), "detail": detail}
    row.update(extra)
    checks.append(row)
    print(f"{'PASS' if ok else 'FAIL'} [{tier}] {cid} {name[:70]} {detail[:110]}", flush=True)
    return row


def squash(text):
    return " ".join(text.split()).lower()


def load_golden():
    return json.loads((HERE / "golden_set.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Relevance of a retrieved chunk to a case's expectation.
# --------------------------------------------------------------------------
def prefixes(exp):
    value = exp.get("group_prefix")
    return [value] if isinstance(value, str) else list(value or [])


def is_relevant(chunk, exp):
    """Chunk is {text, page, kind, group}. Pages and phrases must both hold."""
    pre = prefixes(exp)
    if pre:
        return chunk["kind"] == "table_row" and any(
            chunk["group"].startswith(p) for p in pre)
    ok = True
    if exp.get("pages"):
        ok = ok and chunk["page"] in exp["pages"]
    if exp.get("any_of"):
        text = squash(chunk["text"])
        ok = ok and any(p.lower() in text for p in exp["any_of"])
    return ok if (exp.get("pages") or exp.get("any_of")) else False


def chunk_dict(retriever, index):
    meta = retriever.metadatas[index]
    return {"text": retriever.documents[index], "page": meta["page"],
            "kind": meta.get("kind", "text"), "group": meta.get("group", "")}


# --------------------------------------------------------------------------
# Tier 0: the ground truth itself must be valid against the index.
# --------------------------------------------------------------------------
def tier_validate(golden, retriever):
    corpus = [chunk_dict(retriever, i) for i in range(len(retriever.documents))]
    for case in golden["cases"]:
        exp = case.get("expect")
        if not exp:
            continue
        found = sum(is_relevant(c, exp) for c in corpus)
        need = exp.get("rows") or exp.get("rows_min") or 1
        ok = found >= (need if prefixes(exp) else 1)
        check("validate", case["id"], "ground truth exists in the index", ok,
              f"{found} matching chunk(s) in corpus")


# --------------------------------------------------------------------------
# Tier 1: retrieval quality + hybrid/semantic/keyword ablation (no LLM).
# --------------------------------------------------------------------------
def first_rank(chunks, exp):
    for rank, chunk in enumerate(chunks, 1):
        if is_relevant(chunk, exp):
            return rank
    return None


def context_verdict(case, context):
    """Pass/fail of the exact chunks that would be handed to the LLM."""
    exp = case["expect"]
    pre = prefixes(exp)
    problems = []
    table = [c for c in context if c["is_table"]]
    if pre:
        wrong = [c for c in table if not any(c["group"].startswith(p) for p in pre)]
        if not table:
            problems.append("no troubleshooting rows retrieved")
        if wrong:
            problems.append(f"{len(wrong)} row(s) from other problem groups")
        if "rows" in exp and len(table) != exp["rows"]:
            problems.append(f"{len(table)} rows, expected {exp['rows']}")
        if "rows_min" in exp and len(table) < exp["rows_min"]:
            problems.append(f"{len(table)} rows < min {exp['rows_min']}")
        if "rows_max" in exp and len(table) > exp["rows_max"]:
            problems.append(f"{len(table)} rows > max {exp['rows_max']}")
        if exp.get("any_of"):
            text = squash(" ".join(c["text"] for c in context))
            if not any(p.lower() in text for p in exp["any_of"]):
                problems.append(f"missing phrase {exp['any_of']}")
    else:
        if not any(is_relevant(c, exp) for c in context):
            problems.append("no retrieved chunk satisfies pages/phrase")
    return problems


def tier_retrieval(golden, retriever, store):
    methods = {"hybrid": [], "semantic": [], "keyword": []}
    per_case = []
    for case in golden["cases"]:
        exp = case.get("expect")
        if not exp:
            continue
        q = case["question"]
        t = time.perf_counter()
        context = rag.retrieve_context(q)
        latency = time.perf_counter() - t
        problems = context_verdict(case, context)
        check("retrieval", case["id"], f"[{case['category']}] {q}", not problems,
              "; ".join(problems) or
              f"{len(context)} chunks, pages {sorted({c['page'] for c in context})}",
              category=case["category"])

        hybrid = [{**c, "text": c["text"]} for c in retriever.search(q, top_k=10)]
        semantic = [chunk_dict(retriever, i) for i in retriever.semantic_ranking(q, 10)]
        keyword = [chunk_dict(retriever, i) for i in retriever.keyword_ranking(q, 10)]
        row = {"id": case["id"], "category": case["category"], "latency_s": latency,
               "question": q, "ok": not problems, "problems": problems,
               "pages": sorted({c["page"] for c in context}),
               "chunks": len(context)}
        for name, ranked in (("hybrid", hybrid), ("semantic", semantic), ("keyword", keyword)):
            rank = first_rank(ranked, exp)
            row[name] = rank
            methods[name].append(rank)
        per_case.append(row)
        store.setdefault("retrieval_latency", []).append(latency)

    summary = {}
    for name, ranks in methods.items():
        n = len(ranks)
        hit = lambda k: sum(1 for r in ranks if r and r <= k) / n if n else 0
        summary[name] = {
            "n": n, "hit@1": hit(1), "hit@3": hit(3), "hit@5": hit(5), "hit@10": hit(10),
            "mrr@10": sum(1 / r for r in ranks if r) / n if n else 0,
        }
    store["ablation"] = summary
    store["retrieval_rows"] = per_case


# --------------------------------------------------------------------------
# Tier 2: deterministic guard / post-processing unit tests (no LLM).
# --------------------------------------------------------------------------
def tier_guards():
    ctx = rag.retrieve_context("Why does the engine crank but not start?")
    pages = {c["page"] for c in ctx}

    fake = ("**Troubleshooting table**\nNo matching table rows.\n**Other relevant information**\n"
            "• The manual says \"always sacrifice a goat before starting\" [Manual page 9999].\n"
            "**Limitations**\nx")
    out, notes = rag.postprocess_answer(fake, ctx)
    check("guards", "G01", "invented quote + fake page removed",
          "goat" not in out.lower() and "9999" not in out, f"notes={len(notes)}")

    flags = rag.check_answer("Engine fails. See Section 4. [Manual page 9999]", ctx)
    check("guards", "G02", "check_answer flags bad cite and section number",
          len(flags) >= 2, f"{len(flags)} flags")
    check("guards", "G03", "check_answer flags missing citations",
          bool(rag.check_answer("Just fill the tank.", ctx)))
    check("guards", "G04", "clean answer raises no flag",
          not rag.check_answer(f"• ok [Manual page {next(iter(pages))}]", ctx))

    norm = {
        "(page 15)": "[Manual page 15]", "(Manual page 15)": "[Manual page 15]",
        "[page 15]": "[Manual page 15]", "(pages 10 and 15)": "[Manual page 10], [Manual page 15]",
        "[Manual page 7]": "[Manual page 7]",
    }
    for src, want in norm.items():
        got = rag.normalize_citations(src)
        check("guards", "G05", f"normalize_citations {src!r}", got == want, f"-> {got!r}")

    para = ("**Troubleshooting table**\n• Cause: the carburettor is dirty -> Action: replace the engine "
            "[Manual page 287]\n**Other relevant information**\nNothing relevant.\n**Limitations**\nx")
    out, _ = rag.postprocess_answer(para, ctx)
    check("guards", "G06", "paraphrased/invented table bullet is replaced by real rows",
          "carburettor" not in out.lower() and "• 3. ENGINE CRANKS" in out)

    ctx_nt = rag.retrieve_context("How do I slave start the vehicle?")
    unsup = "**Other relevant information**\n• Bananas are blue and orbit the moon [Manual page 185].\n**Limitations**\nx"
    out, _ = rag.postprocess_answer(unsup, ctx_nt)
    check("guards", "G07", "unsupported 'other info' bullet dropped", "bananas" not in out.lower())

    keep = ("**Other relevant information**\n• Position slaving vehicle and disabled vehicle close enough "
            "for cable hookup [Manual page 185].\n**Limitations**\nx")
    out, _ = rag.postprocess_answer(keep, ctx_nt)
    check("guards", "G08", "supported 'other info' bullet kept", "cable hookup" in out.lower())

    check("guards", "G09", "invented quote detector ignores real quotes",
          not rag.find_invented_quotes('"' + rag.get_retriever().documents[0][:40] + '"',
                                       [{"text": rag.get_retriever().documents[0]}]))


# --------------------------------------------------------------------------
# Tier 3: robustness on odd inputs + determinism (no LLM).
# --------------------------------------------------------------------------
def tier_robustness(golden):
    for q in golden["edge_inputs"] + ["x" * 5000, "word " * 8000]:
        label = repr(q[:28]) + (f" (len {len(q)})" if len(q) > 28 else "")
        try:
            t = time.perf_counter()
            ctx = rag.retrieve_context(q)
            took = time.perf_counter() - t
            check("robustness", "R01", f"retrieval survives {label}",
                  isinstance(ctx, list) and took < 10, f"{len(ctx)} chunks, {took*1000:.0f} ms")
        except Exception as exc:  # noqa: BLE001
            check("robustness", "R01", f"retrieval survives {label}", False,
                  f"{type(exc).__name__}: {exc}")

    q = "Why is the engine overheating?"
    a = [(c["page"], c["text"][:60]) for c in rag.retrieve_context(q)]
    b = [(c["page"], c["text"][:60]) for c in rag.retrieve_context(q)]
    check("robustness", "R02", "retrieval is deterministic within a process", a == b)
    check("robustness", "R03", "retrieve_context(top_k=3) respects a smaller k",
          len(rag.retrieve_context("how do I check the engine oil level?", top_k=3)) <= 6)


# --------------------------------------------------------------------------
# Tier 4: HTTP API with the LLM stubbed out (tests plumbing at zero cost).
# --------------------------------------------------------------------------
def tier_api():
    import app
    from http.server import ThreadingHTTPServer

    canned = ("**Troubleshooting table**\nNo matching table rows.\n**Other relevant information**\n"
              "Nothing relevant.\n**Limitations**\nTable lists possible causes only.")
    real_generate = rag.generate_answer
    rag.generate_answer = lambda question, context: canned
    srv = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    def call(method, path, body=None, headers=None):
        req = urllib.request.Request(base + path, data=body, method=method, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def post(obj, path="/ask"):
        raw = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        status, body = call("POST", path, raw)
        try:
            return status, json.loads(body or b"{}")
        except json.JSONDecodeError:
            return status, {}

    try:
        s, body = call("GET", "/")
        check("api", "A01", "GET / serves the UI", s == 200 and b"__MANUAL__" not in body
              and b"__EXAMPLES__" not in body, f"{len(body)} bytes, placeholders filled")
        check("api", "A02", "GET /nope -> 404", call("GET", "/nope")[0] == 404)
        check("api", "A03", "POST empty question -> 400", post({"question": ""})[0] == 400)
        check("api", "A04", "POST whitespace question -> 400", post({"question": "   "})[0] == 400)
        check("api", "A05", "POST invalid JSON -> 400", post(b"{not json")[0] == 400)
        check("api", "A06", "POST missing field -> 400", post({})[0] == 400)
        check("api", "A07", "POST /other -> 404", post({}, "/other")[0] == 404)
        s, body = post({"question": "Why is the engine overheating?"})
        check("api", "A08", "valid question -> 200 with answer/notes/warnings/sources",
              s == 200 and all(k in body for k in ("answer", "notes", "warnings", "sources"))
              and body["sources"] and all("page" in x and "text" in x for x in body["sources"]),
              f"status={s}")
        s, body = post({"question": "q" * 3000})
        check("api", "A09", "oversized question handled (truncated to 500)", s == 200, f"status={s}")
        s, _ = post({"question": 12345})
        check("api", "A10", "non-string question -> 4xx, not a 500", 400 <= s < 500,
              f"status={s} (a 500 here blames Ollama for a client mistake)")
        s, _ = post({"question": ["a", "b"]})
        check("api", "A11", "list question -> 4xx, not a 500", 400 <= s < 500, f"status={s}")
        s, _ = post([1, 2, 3])
        check("api", "A12", "JSON array body -> 4xx, not a 500", 400 <= s < 500, f"status={s}")

        results = []
        def worker():
            results.append(post({"question": "Why is the engine overheating?"})[0])
        threads = [threading.Thread(target=worker) for _ in range(6)]
        [t.start() for t in threads]
        [t.join(120) for t in threads]
        check("api", "A13", "6 concurrent requests all succeed (lock works)",
              results == [200] * 6, f"statuses={results}")
    finally:
        rag.generate_answer = real_generate
        srv.shutdown()


# --------------------------------------------------------------------------
# Tier 5: persistence & offline start in fresh processes.
# --------------------------------------------------------------------------
def tier_persistence(retriever):
    code = ("import sys;sys.path.insert(0,'src');import rag;r=rag.get_retriever();"
            "c=rag.retrieve_context('Why does the engine crank but not start?');"
            "print('N',len(r.documents));print('P',sorted({x['page'] for x in c}))")
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    outputs = []
    for run in (1, 2):
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                             text=True, env=env, timeout=300)
        lines = [l for l in out.stdout.splitlines() if l.startswith(("N ", "P "))]
        outputs.append(lines)
        check("persistence", f"D0{run}", f"fresh process #{run} loads the index offline",
              out.returncode == 0 and len(lines) == 2, " | ".join(lines) or out.stderr[-150:])
    check("persistence", "D03", "same chunk count and pages across processes",
          len(outputs) == 2 and outputs[0] == outputs[1] and outputs[0] != [])
    check("persistence", "D04", "index chunk count matches this process",
          bool(outputs and outputs[0]) and outputs[0][0] == f"N {len(retriever.documents)}")


# --------------------------------------------------------------------------
# Tier 6: generation (local Ollama), cached.
# --------------------------------------------------------------------------
CACHE = HERE / "cache"


def prompt_fingerprint():
    material = (inspect.getsource(rag.generate_answer) + rag.CHAT_MODEL + rag.TABLE_LABEL
                + json.dumps(rag.OLLAMA_OPTIONS, sort_keys=True))
    return hashlib.sha256(material.encode()).hexdigest()[:12]


def cached_answer(question, context, refresh):
    """Return (raw_answer, latency_seconds, from_cache)."""
    CACHE.mkdir(exist_ok=True)
    key = hashlib.sha256((prompt_fingerprint() + question
                          + rag.build_context_text(context)).encode()).hexdigest()[:20]
    path = CACHE / f"{key}.json"
    if path.exists() and not refresh:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["raw"], data["latency_s"], True
    t = time.perf_counter()
    raw = rag.generate_answer(question, context)
    latency = time.perf_counter() - t
    path.write_text(json.dumps({"question": question, "raw": raw, "latency_s": latency,
                                "model": rag.CHAT_MODEL, "created": datetime.now().isoformat()},
                               indent=1), encoding="utf-8")
    return raw, latency, False


def ollama_up():
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        return True
    except Exception:  # noqa: BLE001
        return False


def sections_of(answer):
    return {(t or "").lower(): [l for l in lines if l.strip()]
            for t, lines in rag.split_sections(answer) if t}


def bullets(lines):
    return [l for l in lines if re.match(r"^\s*[•*\-]\s+", l)]


def numbers_in(text):
    text = re.sub(r"\[Manual page \d+\]", "", text)
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def score_answer(case, context, raw, final, notes):
    """Deterministic checks. Returns dict of {check_name: (ok, detail)}."""
    exp = case.get("expect", {})
    cat = case["category"]
    res = {}
    secs = sections_of(final)
    raw_secs = sections_of(raw)
    required = {TABLE_SECTION, OTHER_SECTION, LIMITS_SECTION}
    res["format"] = (required <= set(secs), f"sections={sorted(secs)}")

    ctx_pages = {c["page"] for c in context}
    cited = {int(n) for n in re.findall(r"\[Manual page (\d+)\]", final)}
    res["citations_valid"] = (cited <= ctx_pages, f"cited={sorted(cited)} retrieved={sorted(ctx_pages)}")

    table_b = bullets(secs.get(TABLE_SECTION, []))
    other_b = bullets(secs.get(OTHER_SECTION, []))
    body = "\n".join(secs.get(TABLE_SECTION, []) + secs.get(OTHER_SECTION, []))

    # Faithfulness: numbers in the answer body must come from the excerpts.
    ctx_numbers = numbers_in(" ".join(c["text"] for c in context))
    unsupported = sorted(numbers_in(re.sub(r"^\s*[•*\-]\s*\d+\.", "", body, flags=re.M)) - ctx_numbers)
    res["numbers_supported"] = (not unsupported, f"unsupported numbers={unsupported}")

    # Model compliance BEFORE guardrails: how much did code have to repair?
    raw_cites = bool(re.search(r"\[Manual page \d+\]", rag.normalize_citations(raw)))
    res["_raw_format_ok"] = (required <= set(raw_secs), "")
    res["_raw_has_citation"] = (raw_cites, "")
    res["_guard_notes"] = (True, str(len(notes)))

    if cat in ("table_whole", "table_specific") and exp:
        pre = prefixes(exp)
        want = exp.get("rows") or exp.get("rows_min")
        right = [b for b in table_b if any(b.startswith(f"• {p}") for p in pre)]
        ok = len(right) == len(table_b) and bool(table_b)
        if "rows" in exp:
            ok = ok and len(table_b) == exp["rows"]
        res["table_rows"] = (ok, f"{len(table_b)} rows, expected {want}, {len(right)} in right group")
    if exp.get("answer_any"):
        text = squash(final)
        res["answer_content"] = (any(p.lower() in text for p in exp["answer_any"]),
                                 f"needs one of {exp['answer_any']}")

    if cat == "out_of_scope":
        explicit = any(m in squash(" ".join(secs.get(TABLE_SECTION, []) + secs.get(OTHER_SECTION, [])))
                       for m in STRICT_REFUSAL)
        leak = bool(table_b) or any("[Manual page" in b for b in other_b)
        res["refusal"] = (explicit and not leak,
                          f"explicit_refusal={explicit} leaked_cited_claims={leak}")
    if cat == "injection":
        low = squash(final)
        res["injection_safe"] = (
            "pwned" not in low and not any(m in low for m in PROMPT_MARKERS),
            "no PWNED / no prompt leak")
    return res


def tier_generation(golden, args, store):
    cases = [c for c in golden["cases"] if c.get("llm")]
    if args.llm_limit:
        cases = cases[:args.llm_limit]
    if not ollama_up():
        check("generation", "L00", "Ollama reachable", False, "start Ollama or use --no-llm")
        return
    uncached = 0
    rows = []
    warmed = False
    for case in cases:
        q = case["question"]
        context = rag.retrieve_context(q)
        # Warm the model once so the first real latency is not a cold start.
        if not warmed:
            path_probe = hashlib.sha256((prompt_fingerprint() + q
                                         + rag.build_context_text(context)).encode()).hexdigest()[:20]
            if args.refresh or not (CACHE / f"{path_probe}.json").exists():
                try:
                    import ollama
                    ollama.chat(model=rag.CHAT_MODEL, messages=[{"role": "user", "content": "hi"}],
                                options={"num_predict": 1})
                except Exception:  # noqa: BLE001
                    pass
            warmed = True
        raw, latency, hit = cached_answer(q, context, args.refresh)
        uncached += 0 if hit else 1
        final, notes = rag.postprocess_answer(raw, context)
        warnings = rag.check_answer(final, context)
        scores = score_answer(case, context, raw, final, notes)
        public = {k: v for k, v in scores.items() if not k.startswith("_")}
        ok = all(v[0] for v in public.values())
        failed = [f"{k}: {v[1]}" for k, v in public.items() if not v[0]]
        check("generation", case["id"], f"[{case['category']}] {q}", ok,
              "; ".join(failed) or f"{latency:.1f}s{' (cached)' if hit else ''}",
              category=case["category"])
        rows.append({
            "id": case["id"], "category": case["category"], "question": q,
            "latency_s": latency, "from_cache": hit, "ok": ok,
            "checks": {k: v[0] for k, v in public.items()},
            "raw_format_ok": scores["_raw_format_ok"][0],
            "raw_has_citation": scores["_raw_has_citation"][0],
            "guard_notes": int(scores["_guard_notes"][1]),
            "warnings": warnings, "answer": final, "raw_answer": raw,
        })
    store["generation_rows"] = rows
    store["generation_uncached"] = uncached


# --------------------------------------------------------------------------
# Tier 7: citation audit (claim level). Reads the cache, so it costs nothing.
# --------------------------------------------------------------------------
CITE = re.compile(r"\[Manual page (\d+)\]")


def cache_lookup(question, context):
    key = hashlib.sha256((prompt_fingerprint() + question
                          + rag.build_context_text(context)).encode()).hexdigest()[:20]
    path = CACHE / f"{key}.json"
    return json.loads(path.read_text(encoding="utf-8"))["raw"] if path.exists() else None


def citation_stats(answer, context, support_min=0.7):
    """Audit every factual bullet: is it cited, is the page real, does that page back the claim?"""
    secs = sections_of(answer)
    # Any non-empty line is a claim, bulleted or not (small models often drop the bullet).
    refusals = ("no matching table rows", "nothing relevant")
    claims = [l for l in secs.get(TABLE_SECTION, []) + secs.get(OTHER_SECTION, [])
              if l.strip() and not any(l.strip().lower().startswith(r) for r in refusals)]
    page_words = {}
    for c in context:
        page_words.setdefault(c["page"], set()).update(rag._words(c["text"]))
    cited = supported = pages_total = pages_valid = 0
    uncited, unsupported = [], []
    for line in claims:
        pages = [int(n) for n in CITE.findall(line)]
        if not pages:
            uncited.append(line.strip()[:120])
            continue
        cited += 1
        pages_total += len(pages)
        valid = [p for p in pages if p in page_words]
        pages_valid += len(valid)
        words = rag._words(CITE.sub("", re.sub(r"^\s*[•*\-]\s+", "", line)))
        backing = set().union(*(page_words.get(p, set()) for p in pages)) if pages else set()
        score = len(words & backing) / len(words) if words else 1.0
        if score >= support_min and len(valid) == len(pages):
            supported += 1
        else:
            unsupported.append(f"{line.strip()[:110]} (support {score:.2f}, pages {pages})")
    n = len(claims)
    return {
        "claims": n, "cited": cited, "supported": supported,
        "pages_total": pages_total, "pages_valid": pages_valid,
        "coverage": cited / n if n else 1.0,
        "validity": pages_valid / pages_total if pages_total else 1.0,
        "precision": supported / cited if cited else 1.0,
        "uncited": uncited, "unsupported": unsupported,
        "cited_pages": sorted({int(p) for p in CITE.findall(answer)}),
    }


def tier_citation(golden, retriever, store):
    corpus = [chunk_dict(retriever, i) for i in range(len(retriever.documents))]
    rows = []
    for case in golden["cases"]:
        if not case.get("llm"):
            continue
        q, cat = case["question"], case["category"]
        context = rag.retrieve_context(q)
        raw = cache_lookup(q, context)
        if raw is None:
            print(f"SKIP [citation] {case['id']} not in cache (run the generation tier first)")
            continue
        final, _ = rag.postprocess_answer(raw, context)
        fin = citation_stats(final, context)
        before = citation_stats(rag.normalize_citations(raw), context)
        exp = case.get("expect")
        gold = {c["page"] for c in corpus if is_relevant(c, exp)} if exp else set()
        gold_hit = bool(set(fin["cited_pages"]) & gold) if gold else None

        problems = []
        if fin["validity"] < 1:
            problems.append("cites a page that was not retrieved")
        if fin["precision"] < 1:
            problems.append(f"{len(fin['unsupported'])} cited claim(s) not backed by the cited page")
        if fin["uncited"]:
            problems.append(f"{len(fin['uncited'])} claim(s) with no citation")
        if cat == "out_of_scope" and fin["cited"]:
            problems.append("cites the manual for an out-of-scope question")
        if cat not in ("out_of_scope", "injection"):
            if fin["claims"] == 0:
                problems.append("no cited claims at all")
            if gold_hit is False:
                problems.append(f"does not cite any expected page (expected one of {sorted(gold)[:6]})")
        check("citation", case["id"], f"[{cat}] {q}", not problems,
              "; ".join(problems) or f"{fin['cited']}/{fin['claims']} claims cited, "
              f"pages {fin['cited_pages']}", category=cat)
        rows.append({"id": case["id"], "category": cat, "question": q, "ok": not problems,
                     "problems": problems, "final": fin, "raw": before,
                     "gold_pages": sorted(gold)[:10], "gold_hit": gold_hit})
    store["citation_rows"] = rows


def tier_stability(golden, args, store):
    """Same prompt twice, no cache: is the local model repeatable at temperature 0?"""
    if not ollama_up():
        return
    picks = [c for c in golden["cases"] if c["id"] in ("T01", "F01", "O01")][:args.stability_n]
    for case in picks:
        context = rag.retrieve_context(case["question"])
        a = rag.generate_answer(case["question"], context)
        b = rag.generate_answer(case["question"], context)
        ratio = difflib.SequenceMatcher(None, a, b).ratio()
        check("stability", case["id"], f"identical answers on repeat: {case['question']}",
              a == b or ratio > 0.97, f"similarity={ratio:.3f}")


# --------------------------------------------------------------------------
# Report.
# --------------------------------------------------------------------------
def pct(n, d):
    return f"{(100 * n / d):.0f}%" if d else "n/a"


def write_reports(store, args, tiers):
    import report_builder

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = HERE / "reports" / stamp
    out.mkdir(parents=True, exist_ok=True)
    retriever = rag.get_retriever()
    golden = load_golden()
    gen = store.get("generation_rows", [])
    llm_cases = sum(1 for c in golden["cases"] if c.get("llm"))
    meta = {
        "stamp": stamp, "wall": time.time() - started, "model": rag.CHAT_MODEL,
        "model_info": report_builder.ollama_info(rag.CHAT_MODEL),
        "chunks": len(retriever.documents),
        "pages": len({m["page"] for m in retriever.metadatas}),
        "table_rows": sum(1 for m in retriever.metadatas if m.get("kind") == "table_row"),
        "n_cases": len(golden["cases"]), "n_edge": len(golden["edge_inputs"]), "n_llm": llm_cases,
        "machine": f"macOS, {os.cpu_count()} cores, {os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9:.0f} GB RAM",
        "gen_new": store.get("generation_uncached", 0),
        "gen_cached": len(gen) - store.get("generation_uncached", 0),
        "rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
    }
    markdown = report_builder.build_markdown(checks, store, meta)
    (out / "report.md").write_text(markdown, encoding="utf-8")

    (out / "results.json").write_text(json.dumps(
        {"stamp": stamp, "model": rag.CHAT_MODEL, "fingerprint": prompt_fingerprint(),
         "checks": checks, "ablation": store.get("ablation"),
         "retrieval_rows": store.get("retrieval_rows"), "generation": gen,
         "citation": store.get("citation_rows")}, indent=1), encoding="utf-8")
    with (out / "cases.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tier", "id", "name", "result", "detail"])
        for c in checks:
            w.writerow([c["tier"], c["id"], c["name"], "PASS" if c["ok"] else "FAIL", c["detail"]])

    latest = HERE / "reports" / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(out, latest)

    # Charts need matplotlib (system python); then re-render the report with the images.
    try:
        subprocess.run(["python3", str(HERE / "make_summary.py")], capture_output=True, timeout=120)
    except Exception:  # noqa: BLE001
        pass
    markdown = report_builder.build_markdown(checks, store, meta)
    html = report_builder.to_html(markdown, base_dir=out)
    for folder in (out, latest):
        (folder / "report.md").write_text(markdown, encoding="utf-8")
        (folder / "report.html").write_text(html, encoding="utf-8")
    report_builder.to_pdf(str(out / "report.html"), str(out / "report.pdf"))
    total_ok = sum(c["ok"] for c in checks)
    return out, total_ok, len(checks)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiers", default=",".join(ALL_TIERS))
    ap.add_argument("--no-llm", action="store_true", help="skip generation + stability tiers")
    ap.add_argument("--refresh", action="store_true", help="ignore the answer cache")
    ap.add_argument("--llm-limit", type=int, default=0, help="only the first N llm cases")
    ap.add_argument("--stability-n", type=int, default=3)
    args = ap.parse_args()

    tiers = [t for t in args.tiers.split(",") if t]
    if args.no_llm:
        tiers = [t for t in tiers if t not in ("generation", "stability")]  # citation reads the cache

    golden = load_golden()
    print(f"Loading retriever (offline)...  tiers={tiers}", flush=True)
    retriever = rag.get_retriever()
    store = {}
    runners = {
        "validate": lambda: tier_validate(golden, retriever),
        "retrieval": lambda: tier_retrieval(golden, retriever, store),
        "guards": tier_guards,
        "robustness": lambda: tier_robustness(golden),
        "api": tier_api,
        "persistence": lambda: tier_persistence(retriever),
        "generation": lambda: tier_generation(golden, args, store),
        "citation": lambda: tier_citation(golden, retriever, store),
        "stability": lambda: tier_stability(golden, args, store),
    }
    for tier in ALL_TIERS:
        if tier in tiers:
            print(f"\n=== {tier} ===", flush=True)
            runners[tier]()

    out, ok, total = write_reports(store, args, tiers)
    print(f"\n{ok}/{total} checks passed. Report: {out.relative_to(ROOT)}/report.md "
          f"(also evaluation/reports/latest/)")
    sys.exit(0 if ok == total else 1)


if __name__ == "__main__":
    main()
