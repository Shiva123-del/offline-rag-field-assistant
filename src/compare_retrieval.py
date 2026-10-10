"""Compare semantic-only retrieval with hybrid (semantic + keyword) retrieval.

Uses the HybridRetriever in retrieval.py. For each test question it checks
whether the manual's troubleshooting page is among the top matches, for
plain semantic search and for hybrid search.

Usage:
    python src/compare_retrieval.py          # compact results + summary
    python src/compare_retrieval.py --full   # also print the full chunk text
"""

import argparse

from retrieval import HybridRetriever

TOP_K = 5

# The manual's troubleshooting table (cause -> action) is on this page.
TROUBLESHOOTING_PAGE = 15

QUESTIONS = [
    "Why is the generator not starting?",
    "What should I check if the diesel engine will not start?",
    "What are the causes of generator starting failure?",
    "The generator has a weak battery. What does the manual say?",
    "What does the troubleshooting table say about insufficient fuel?",
]

SEPARATOR = "=" * 70


def print_hybrid_results(question: str, matches: list[dict], full: bool) -> None:
    """Print one question and its hybrid-ranked chunks."""
    print(f"\n{SEPARATOR}")
    print(f"QUESTION: {question}")
    print(SEPARATOR)

    for rank, match in enumerate(matches, start=1):
        print(
            f"\nRESULT {rank} | Page {match['page']} | {match['kind']} "
            f"| semantic rank {match['semantic_rank']} "
            f"| keyword rank {match['keyword_rank']} "
            f"| score {match['score']:.4f}"
        )
        text = " ".join(match["text"].split())
        print(text if full else text[:160] + ("..." if len(text) > 160 else ""))


def pages_of(matches: list[dict]) -> list[int]:
    return [match["page"] for match in matches]


def print_summary(rows: list[tuple[str, list[int], list[int]]]) -> None:
    """Show, per question, whether page 15 made the top results."""
    print(f"\n{SEPARATOR}")
    print(f"SUMMARY: is troubleshooting page {TROUBLESHOOTING_PAGE} in the top {TOP_K}?")
    print(SEPARATOR)

    semantic_hits = hybrid_hits = 0

    for question, semantic_pages, hybrid_pages in rows:
        in_semantic = TROUBLESHOOTING_PAGE in semantic_pages
        in_hybrid = TROUBLESHOOTING_PAGE in hybrid_pages
        semantic_hits += in_semantic
        hybrid_hits += in_hybrid

        print(f"\n{question}")
        print(f"  semantic only: {'YES' if in_semantic else 'NO ':3} pages {semantic_pages}")
        print(f"  hybrid       : {'YES' if in_hybrid else 'NO ':3} pages {hybrid_pages}")

    total = len(rows)
    print(f"\nSemantic only: {semantic_hits}/{total}   Hybrid: {hybrid_hits}/{total}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full", action="store_true", help="print full chunk text")
    args = parser.parse_args()

    print("Loading local embedding model and building keyword index...")
    retriever = HybridRetriever()

    summary = []
    for question in QUESTIONS:
        hybrid = retriever.search(question, top_k=TOP_K)
        semantic = retriever.search_semantic_only(question, top_k=TOP_K)

        print_hybrid_results(question, hybrid, args.full)
        summary.append((question, pages_of(semantic), pages_of(hybrid)))

    print_summary(summary)


if __name__ == "__main__":
    main()
