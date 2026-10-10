import re
import sys

import ollama

from retrieval import (
    HybridRetriever,
    SYNONYMS,
    is_problem_question,
    is_troubleshooting_chunk,
    stem,
    tokenize,
)

CHAT_MODEL = "llama3.2:latest"

# Fewer, better chunks: a small local model copes better with a short context.
TOP_K = 6

# When troubleshooting rows were found, general-text chunks are only
# supporting material; keep a couple so the model is not flooded with them.
MAX_NON_TABLE_WITH_TABLE = 2

# Once the troubleshooting table answers the question, general text is only
# useful as a safety warning. Other passages just add noise.
SAFETY_PATTERN = re.compile(
    r"warning|caution|danger|shock|fire|burn|explos|hot\b|toxic|exhaust|"
    r"carbon monoxide|ground(?:ed|ing)",
    re.IGNORECASE,
)

# Ollama's default context window is small. The prompt (rules + excerpts) can
# overrun it, which silently cuts text off. temperature 0 keeps answers
# repeatable and closer to the evidence.
OLLAMA_OPTIONS = {"num_ctx": 4096, "temperature": 0}

QUESTION = "Why does the engine crank but not start?"

# Label added to troubleshooting-table rows. Ingestion reads the real PDF
# table, so the cause and action in one row belong together.
TABLE_LABEL = "TROUBLESHOOTING TABLE ROW - cause and action paired by the table"

_retriever = None


def get_retriever():
    """Load the embedding model and keyword index once, on first use."""
    global _retriever

    if _retriever is None:
        print("Loading local embedding model and building keyword index...")
        _retriever = HybridRetriever()

    return _retriever


def add_rest_of_table_group(context, retriever, max_groups=1):
    """If troubleshooting rows were found, include their whole problem group.

    A question like "why is it not starting" needs every listed cause for
    that problem, not just the few rows that ranked highest. The groups of
    the best-ranked rows (at most max_groups) are completed, so unrelated
    problems are not mixed in. Returns (context, groups_completed).
    """
    groups = []
    for item in context:
        if item["is_table"] and item["group"] not in groups:
            groups.append(item["group"])
    groups = groups[:max_groups]
    present = {item["text"] for item in context}

    for text, metadata in zip(retriever.documents, retriever.metadatas):
        if (
            metadata.get("kind") == "table_row"
            and metadata.get("group") in groups
            and text not in present
        ):
            context.append({
                "text": text,
                "page": metadata["page"],
                "is_table": True,
                "kind": "table_row",
                "group": metadata["group"]
            })

    return context, groups


def asks_about_whole_problem(question, group, retriever):
    """True if the question is about a problem, not one specific cause.

    "Why is there no electrical output?" wants every row of its problem
    group. "What if the battery is weak?" or "the fuse" names a word that
    belongs to exactly one cause row, so only that row is wanted.
    """
    rows = [
        text.split("| Cause: ", 1)[-1].split(" | Action:")[0]
        for text, metadata in zip(retriever.documents, retriever.metadatas)
        if metadata.get("kind") == "table_row" and metadata.get("group") == group
    ]
    row_terms = [set(tokenize(row)) for row in rows]
    group_terms = set(tokenize(group))

    # Everyday words that only stand for the group's own name
    # (e.g. "electricity" -> "generates") do not name a cause.
    synonym_terms = {
        stem(word)
        for word, extras in SYNONYMS.items()
        if group_terms & {stem(extra) for extra in extras}
    }

    # Words found all over the manual ("generator") name no cause either.
    ignored = group_terms | synonym_terms | retriever.keyword_index.common_terms

    for term in set(tokenize(question)) - ignored:
        if sum(term in terms for terms in row_terms) == 1:
            return False

    return True


def retrieve_context(question, top_k=TOP_K):
    """Hybrid (semantic + keyword) retrieval of manual chunks.

    For questions that describe a problem, chunks from the troubleshooting
    table are placed first, so the model sees the cause/action evidence
    before general text. Order is otherwise the fused ranking.
    """
    retriever = get_retriever()
    matches = retriever.search(question, top_k=top_k)

    context = [
        {
            "text": match["text"],
            "page": match["page"],
            "is_table": match["kind"] == "table_row",
            "kind": match["kind"],
            "group": match["group"]
        }
        for match in matches
    ]

    first_row = next((item for item in context if item["is_table"]), None)
    whole_problem = bool(
        first_row
        and asks_about_whole_problem(question, first_row["group"], retriever)
    )

    whole_groups = []
    if whole_problem:
        context, whole_groups = add_rest_of_table_group(context, retriever)
        # Stray rows of other problems only confuse a small model.
        context = [
            item for item in context
            if not item["is_table"] or item["group"] in whole_groups
        ]

    # Drop the weakest general-text chunks once table rows are present.
    if first_row is not None:
        table = [item for item in context if item["is_table"]]
        other = [item for item in context if not item["is_table"]]
        safety = [i for i in other if SAFETY_PATTERN.search(i["text"])]
        context = table + safety[:MAX_NON_TABLE_WITH_TABLE]

    for item in context:
        item["whole_groups"] = whole_groups

    if is_problem_question(question):
        # sorted() is stable: the fused ranking is kept inside each group.
        context.sort(key=lambda item: not item["is_table"])

    return context


# One group of page numbers, e.g. "15", "10, 11" or "10, 11 and 15".
_PAGE_GROUP = r"pages?\s+(\d+(?:\s*(?:,|and|&)\s*\d+)*)"


def _as_exact_citations(match):
    numbers = re.findall(r"\d+", match.group(1))
    return ", ".join(f"[Manual page {number}]" for number in numbers)


def normalize_citations(answer):
    """Rewrite loose citations into the exact [Manual page N] format.

    "(page 15)", "(Manual page 15)", "[page 15]" and "(pages 10 and 15)"
    all become "[Manual page 15]" (one tag per page). Already-correct
    citations are left unchanged.
    """
    answer = re.sub(
        rf"\(\s*(?:Manual\s+)?{_PAGE_GROUP}\s*\)",
        _as_exact_citations,
        answer,
        flags=re.IGNORECASE,
    )
    answer = re.sub(
        rf"\[\s*(?:Manual\s+)?{_PAGE_GROUP}\s*\]",
        _as_exact_citations,
        answer,
        flags=re.IGNORECASE,
    )
    return answer


def find_unsupported_citations(answer, context):
    """Return cited page numbers that were not in the retrieved context."""
    retrieved_pages = {item["page"] for item in context}
    cited_pages = set()

    for group in re.findall(_PAGE_GROUP, answer, flags=re.IGNORECASE):
        cited_pages.update(int(number) for number in re.findall(r"\d+", group))

    return sorted(cited_pages - retrieved_pages)


def _plain(text):
    """Lowercase, straighten quotes and collapse spaces, for comparisons."""
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("’", "'").replace("‘", "'")
    return " ".join(text.lower().split())


def find_invented_quotes(answer, context, minimum_length=20):
    """Return quoted phrases in the answer that no excerpt contains."""
    excerpts = _plain(" ".join(item["text"] for item in context))
    invented = []

    for quote in re.findall(r'[“"]([^“”"\n]+)[”"]', answer):
        if len(quote) >= minimum_length and _plain(quote) not in excerpts:
            invented.append(quote.strip())

    return invented


TABLE_SECTION = "troubleshooting table"
OTHER_SECTION = "other relevant information"

# Words too common to prove that a sentence is supported by the manual.
_FILLER_WORDS = {
    "that", "this", "with", "from", "have", "will", "should", "must",
    "make", "sure", "your", "they", "them", "then", "than", "also",
    "when", "while", "before", "after", "ensure", "using", "used",
}

_BULLET = re.compile(r"^\s*[•*\-]\s+")


def _words(text):
    """Set of meaningful lowercase words (4+ letters, no filler)."""
    return {
        word for word in re.findall(r"[a-z0-9]+", _plain(text))
        if len(word) >= 4 and word not in _FILLER_WORDS
    }


def split_sections(answer):
    """Split an answer into [(title, [lines])] at its **bold** headings.

    Text before the first heading is kept under the title None.
    """
    sections = [(None, [])]

    for line in answer.split("\n"):
        heading = re.match(r"^\s*\*\*(.+?)\*\*\s*:?\s*$", line)
        if heading:
            sections.append((heading.group(1).strip(), []))
        else:
            sections[-1][1].append(line)

    return sections


def parse_table_rows(context):
    """Turn stored table-row chunks back into exact cause/action data."""
    rows = []

    for item in context:
        if not item.get("is_table") or "Cause:" not in item["text"]:
            continue

        parts = [part.strip() for part in item["text"].split(" | ")]
        fields = {
            key: value
            for part in parts
            if ": " in part
            for key, value in [part.split(": ", 1)]
        }

        if "Cause" in fields and "Action" in fields:
            rows.append({
                "problem": fields.get("Problem", ""),
                "cause": fields["Cause"],
                "action": fields["Action"],
                "page": item["page"]
            })

    return rows


def rebuild_table_section(lines, context):
    """Rewrite the table bullets using the manual's exact row text.

    The model only decides WHICH rows apply. Each bullet is matched to a
    stored row, and the row's own wording and page are printed. Bullets
    that match no row (paraphrased or invented) are dropped.
    Returns (new_lines, notes).
    """
    rows = parse_table_rows(context)
    notes, used, rebuilt = [], set(), []
    primary = rows[0]["problem"] if rows else ""

    for line in lines:
        if not _BULLET.match(line):
            continue

        line_words = _words(line)
        best_index, best_score = None, 0.0

        for index, row in enumerate(rows):
            cause_words = _words(row["cause"])
            action_words = _words(row["action"])
            if not cause_words or not action_words:
                continue

            score = (
                len(cause_words & line_words) / len(cause_words)
                + len(action_words & line_words) / len(action_words)
            ) / 2
            # Several problems can share identical rows ("Other causes" ->
            # "Notify unit maintenance"); prefer the best-ranked problem.
            if row["problem"] == primary:
                score += 0.01

            if score > best_score:
                best_index, best_score = index, score

        if best_index is None or best_score < 0.6:
            notes.append(
                "Removed a table bullet that matches no row of the "
                f"manual's table: {line.strip()}"
            )
        elif best_index not in used:
            used.add(best_index)
            row = rows[best_index]
            rebuilt.append(
                f"• {row['problem']}: Cause: {row['cause']} -> "
                f"Action: {row['action']} [Manual page {row['page']}]"
            )

    # A whole-problem question needs every row of its group(s), even if the
    # small model forgot some. Rows come from the manual, not the model.
    whole_groups = {
        group for item in context for group in item.get("whole_groups", [])
    }
    for index, row in enumerate(rows):
        if index not in used and row["problem"] in whole_groups:
            rebuilt.append(
                f"• {row['problem']}: Cause: {row['cause']} -> "
                f"Action: {row['action']} [Manual page {row['page']}]"
            )

    # "Other causes" is the catch-all last step of each list in the manual.
    rebuilt.sort(key=lambda line: "Cause: Other causes" in line)

    if not rebuilt:
        rebuilt = ["No matching table rows."]

    return rebuilt, notes


def keep_supported_bullets(lines, context, minimum_support=0.7):
    """Keep only bullets whose key words appear in the retrieved excerpts.

    This catches fluent sentences that the manual does not actually say.
    Returns (kept_lines, notes).
    """
    excerpt_words = _words(" ".join(item["text"] for item in context))
    kept, notes = [], []

    for line in lines:
        if not _BULLET.match(line):
            if line.strip():
                kept.append(line)
            continue

        words = _words(_BULLET.sub("", re.sub(r"\[Manual page \d+\]", "", line)))
        support = len(words & excerpt_words) / len(words) if words else 1.0

        if support >= minimum_support:
            # Cite the page of the excerpt that best matches this line, in
            # case the model attached the wrong page number.
            best = max(context, key=lambda item: len(words & _words(item["text"])))
            line = re.sub(r"\s*\[Manual page \d+\]", "", line).rstrip()
            kept.append(f"{line} [Manual page {best['page']}]")
        else:
            notes.append(
                "Removed a line that the retrieved excerpts do not clearly "
                f"support: {line.strip()}"
            )

    return kept, notes


def remove_invented_quote_lines(answer, context):
    """Drop answer lines that quote text the manual does not contain.

    Returns (cleaned_answer, removed_lines).
    """
    kept, removed = [], []

    for line in answer.split("\n"):
        if find_invented_quotes(line, context):
            removed.append(line.strip())
        else:
            kept.append(line)

    return "\n".join(kept), removed


def postprocess_answer(answer, context):
    """Repair what code can repair. Returns (answer, notes)."""
    answer = normalize_citations(answer)
    answer, removed = remove_invented_quote_lines(answer, context)

    notes = [
        "Removed a line because it quoted text that is not in the "
        f"retrieved excerpts: {line}"
        for line in removed
    ]

    output = []
    for title, lines in split_sections(answer):
        key = (title or "").lower()

        if key == TABLE_SECTION:
            lines, section_notes = rebuild_table_section(lines, context)
            notes.extend(section_notes)
        elif key == OTHER_SECTION:
            lines, section_notes = keep_supported_bullets(lines, context)
            if not lines:
                lines = ["Nothing relevant."]
            notes.extend(section_notes)

        if title is not None:
            output.append(f"**{title}**")
        output.extend(lines)

    return "\n".join(output), notes


def check_answer(answer, context):
    """Return a list of human-readable problems found in the answer."""
    problems = []

    if not re.search(r"\[Manual page \d+\]", answer):
        problems.append(
            "The answer contains no [Manual page N] citations, so its "
            "claims cannot be checked against the manual."
        )

    unsupported = find_unsupported_citations(answer, context)
    if unsupported:
        problems.append(
            "The answer cites page(s) "
            f"{', '.join(map(str, unsupported))}, which were not in the "
            "retrieved excerpts. Treat those citations as unverified."
        )

    invented = find_invented_quotes(answer, context)
    if invented:
        problems.append(
            "These quoted phrases are not in the retrieved excerpts, so "
            "they may be invented: " + "; ".join(f'"{quote}"' for quote in invented)
        )

    if re.search(r"\bsections?\s+\d", answer, flags=re.IGNORECASE):
        problems.append(
            "The answer refers to section numbers. The model was told not "
            "to invent them, so treat them as unverified."
        )

    return problems


def build_context_text(context):
    """Format the excerpts, labelling troubleshooting-table chunks."""
    parts = []

    for item in context:
        header = f"[Manual page {item['page']}]"
        if item.get("is_table"):
            header += f" ({TABLE_LABEL})"
        elif item.get("kind") == "maintenance_row":
            header += " (MAINTENANCE SCHEDULE - how often each item is serviced)"
        parts.append(f"\n--- {header} ---\n{item['text']}\n")

    return "".join(parts)


def generate_answer(question, context):
    context_text = build_context_text(context)

    prompt = f"""
You are an offline technical assistant that answers ONLY from a supplied
equipment manual. Accuracy matters more than length. A short, correct
answer is better than a long answer that guesses.

USER QUESTION:
{question}

MANUAL EXCERPTS:
{context_text}

STRICT EVIDENCE RULES:

1. Use only facts explicitly written in the excerpts above.
2. Do not invent causes, procedures, specifications or section numbers.
   Never write "Section N". Refer to pages only.
3. CITATION FORMAT. Every factual statement must end with a citation in
   exactly this form: [Manual page 15]. Not "(page 15)", not "page 15",
   not "[page 15]". Use only page numbers that appear in the excerpt
   headers above.
4. Excerpts labelled "{TABLE_LABEL}"
   are rows of the manual's troubleshooting table, written as
   "Problem | Cause | Action". The cause and action in ONE row belong
   together. Never combine a cause from one row with an action from
   another row.
   * Use only rows whose Problem matches the question. Different Problems
     (for example an engine that will not crank and one that cranks but
     will not start) must not be mixed.
   * If the question asks about a whole problem (for example "why is it
     not starting"), list EVERY row of the matching Problem group. If it
     asks about one specific cause (for example "weak battery"), list
     only the row(s) for that cause.
   * One cause cell can hold two causes joined by " / " that share one
     action. Keep them together as written.
   * Never put words in quotation marks unless you copied them exactly
     from an excerpt. Do not invent safety warnings.
   Excerpts labelled "MAINTENANCE SCHEDULE" give how often an item is
   checked, cleaned or replaced. Use them for "how often" questions and
   put them under "Other relevant information", exactly as written.
5. Keep the starting procedure separate from the troubleshooting table.
   Do not call a starting precaution the cause of a failure.
6. Do not recommend disassembly, adjustment or replacement unless the
   manual clearly says so.
7. Mention relevant safety warnings from the excerpts.
8. If the excerpts do not contain the answer, say so plainly. Do not
   fill the gap with general knowledge.

Answer in exactly these three parts:

**Troubleshooting table**
Use ONLY the labelled table rows that match the question. One bullet per
row, in this form:
"Cause: ... -> Action: ... [Manual page N]"
Write "No matching table rows." if there are none.

**Other relevant information**
Short bullets from the non-table excerpts (procedures, safety warnings),
each ending with a [Manual page N] citation. Write "Nothing relevant."
if there is nothing.

**Limitations**
What the excerpts do not establish. Say that the table lists possible
causes and does not diagnose this specific vehicle or machine. Recommend
authorized technical support for repairs.

Do not claim the problem is diagnosed or fixed.
"""

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options=OLLAMA_OPTIONS
    )

    return response["message"]["content"]


if __name__ == "__main__":
    # Optional: python src/rag.py "your question here"
    question = " ".join(sys.argv[1:]) or QUESTION

    print("\nRetrieving manual evidence...")
    context = retrieve_context(question)

    pages = sorted(set(item["page"] for item in context))
    print(f"Retrieved {len(context)} chunks.")
    print("Manual pages:", ", ".join(map(str, pages)))

    print("\nGenerating evidence-based answer with Ollama...")
    # The small local model ignores some rules, so enforce them in code.
    answer, notes = postprocess_answer(
        generate_answer(question, context), context
    )

    print("\nQUESTION:")
    print(question)

    print("\nANSWER:")
    print(answer)

    for note in notes:
        print(f"\nNOTE: {note}")

    for problem in check_answer(answer, context):
        print(f"\nWARNING: {problem}")
