"""Hybrid retrieval over the manual: semantic search + keyword search.

* Semantic search (local sentence-transformers embeddings in ChromaDB)
  finds passages with a similar meaning.
* Keyword search (BM25) catches exact technical terms such as "battery",
  "fuel" or "start" that embeddings can miss.
* The two ranked lists are merged with Reciprocal Rank Fusion (RRF), so a
  chunk that ranks well in either list, or in both, comes out on top.

Everything runs locally. No new dependencies are needed.
"""

import math
import os
import re
from collections import Counter
from pathlib import Path

# This is an offline assistant: use the embedding model already stored on
# this computer and do not contact Hugging Face (no waiting, no errors when
# the internet is off). To download the model for the first time, run once
# with:  HF_HUB_OFFLINE=0 python src/ingest.py
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_DB_PATH = str(PROJECT_ROOT / "data" / "vector_db")
COLLECTION_NAME = "field_manuals"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# How many candidates each search contributes before fusion.
CANDIDATES = 20
# Standard RRF constant; larger values flatten the effect of rank.
RRF_K = 60

# Question words that carry no search value. Words such as "not" are kept
# on purpose: "not starting" is the key phrase in this manual.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "has", "have", "how", "i", "if", "in", "is", "it", "me",
    "my", "of", "on", "or", "say", "says", "should", "the", "this", "to",
    "was", "what", "when", "where", "which", "why", "will", "with", "manual",
}

_SUFFIXES = ("ing", "ed", "es", "s")

# Terms found in more than this share of all chunks (e.g. "generator" in a
# generator manual) say nothing about which chunk is relevant, so they are
# ignored in keyword search.
MAX_TERM_FRACTION = 0.5

# Everyday words for a problem -> the wording this kind of manual uses.
# Used only to widen the *keyword* query; semantic search is untouched.
# Extend it freely for other manuals.
SYNONYMS = {
    "weak": ["low", "lower"],
    "dead": ["low", "lower"],
    "flat": ["low", "lower", "air", "pressure"],
    "failure": ["malfunction"],
    "fault": ["malfunction"],
    "problem": ["malfunction"],
    "cause": ["malfunction"],
    "fix": ["action"],
    "solution": ["action"],
    # Everyday words -> the wording of vehicle operator manuals.
    "stall": ["stops"],
    "stalls": ["stops"],
    "wont": ["fails"],
    "overheating": ["overheats"],
    "hot": ["overheats"],
    "noisy": ["noise"],
    "puncture": ["air", "pressure"],
    "hummer": ["vehicle"],
    "humvee": ["vehicle"],
    "hmmwv": ["vehicle"],
    "truck": ["vehicle"],
}


def stem(word: str) -> str:
    """Very light stemming so "starting", "starts" and "start" match."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            root = word[: -len(suffix)]
            # "brakes" -> "brake" (via "s"), but "switches" -> "switch".
            if suffix == "es" and not root.endswith(("s", "x", "z", "ch", "sh")):
                continue
            return root
    return word


def tokenize(text: str) -> list[str]:
    """Lowercase, split into words, drop stopwords and stem."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [stem(word) for word in words if word not in STOPWORDS]


_STEMMED_SYNONYMS = {
    stem(word): [stem(extra) for extra in extras]
    for word, extras in SYNONYMS.items()
}


# A question containing one of these words is describing a problem, so the
# answer most likely sits in the manual's troubleshooting section.
PROBLEM_WORDS = {
    "not", "cannot", "can't", "won't", "wont", "fail", "fails", "failed",
    "failure", "problem", "issue", "trouble", "why", "stuck", "stops",
    "weak", "dead", "flat", "broken", "damaged", "faulty", "low",
    "no", "without", "wont", "overheat", "overheats", "overheating",
    "stall", "stalls", "leak", "leaks", "leaking", "noise", "noisy",
    "smoke", "smoking", "wobble", "wobbles", "drag", "drags", "dragging",
    "slow", "slowly", "rough", "wanders", "hard", "crank", "cranks",
    "misfire", "misfires", "slips", "slippage", "stops", "wrong",
}
TROUBLESHOOTING_TERMS = [stem(t) for t in ("troubleshooting", "malfunction", "action")]


def is_problem_question(question: str) -> bool:
    """True if the question describes a fault ("not starting", "why", ...)."""
    words = set(re.findall(r"[a-z']+", question.lower()))
    return bool(words & PROBLEM_WORDS)


def is_troubleshooting_chunk(text: str) -> bool:
    """True if the chunk looks like part of the cause/action table."""
    lowered = text.lower()
    return "causes of malfunction" in lowered or "troubleshooting" in lowered


def expand_query(question: str) -> list[str]:
    """Tokenize the question and add the manual's own wording.

    * Everyday problem words are mapped to the manual's wording (SYNONYMS).
    * Questions that describe a problem also search for troubleshooting
      vocabulary, so the cause/action table is found even when the question
      only uses very common words such as "start".
    """
    terms = tokenize(question)
    expanded = list(terms)
    for term in terms:
        expanded.extend(_STEMMED_SYNONYMS.get(term, []))

    if is_problem_question(question):
        expanded.extend(TROUBLESHOOTING_TERMS)

    return expanded


class BM25Index:
    """Small, dependency-free BM25 keyword index."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.term_counts = [Counter(tokenize(doc)) for doc in documents]
        self.lengths = [sum(counts.values()) for counts in self.term_counts]
        self.avg_length = sum(self.lengths) / max(len(self.lengths), 1)

        document_frequency = Counter()
        for counts in self.term_counts:
            document_frequency.update(counts.keys())

        total = len(documents)
        self.idf = {
            term: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }
        self.common_terms = {
            term
            for term, freq in document_frequency.items()
            if freq / total > MAX_TERM_FRACTION
        }

    def query_terms(self, question: str) -> list[str]:
        """Expand the question, then drop terms that appear almost everywhere."""
        terms = expand_query(question)
        informative = [t for t in terms if t not in self.common_terms]
        # If every term is common, fall back to the full list.
        return informative or terms

    def scores(self, question: str) -> list[float]:
        """Return a BM25 score for every document."""
        query_terms = self.query_terms(question)
        results = []

        for counts, length in zip(self.term_counts, self.lengths):
            score = 0.0
            for term in query_terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                norm = 1 - self.b + self.b * length / self.avg_length
                score += self.idf[term] * (
                    frequency * (self.k1 + 1) / (frequency + self.k1 * norm)
                )
            results.append(score)

        return results


class HybridRetriever:
    """Semantic + keyword retrieval with Reciprocal Rank Fusion."""

    def __init__(
        self,
        db_path: str = VECTOR_DB_PATH,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.model = SentenceTransformer(embedding_model)

        client = chromadb.PersistentClient(path=db_path)
        self.collection = client.get_collection(name=collection_name)

        stored = self.collection.get(include=["documents", "metadatas"])
        self.ids = stored["ids"]
        self.documents = stored["documents"]
        self.metadatas = stored["metadatas"]
        self.keyword_index = BM25Index(self.documents)

    def semantic_ranking(self, question: str, count: int) -> list[int]:
        """Indexes of the chunks closest in meaning to the question."""
        embedding = self.model.encode(question).tolist()
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(count, len(self.ids)),
            include=[],
        )
        position = {chunk_id: i for i, chunk_id in enumerate(self.ids)}
        return [position[chunk_id] for chunk_id in results["ids"][0]]

    def keyword_ranking(self, question: str, count: int) -> list[int]:
        """Indexes of the best keyword matches (zero-score chunks dropped)."""
        scores = self.keyword_index.scores(question)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [i for i in order[:count] if scores[i] > 0]

    def search(self, question: str, top_k: int = 5) -> list[dict]:
        """Return the top_k chunks after fusing both rankings.

        Each result has: page, text, score, semantic_rank, keyword_rank
        (the ranks are 1-based, or None if that search did not return it).
        """
        semantic = self.semantic_ranking(question, CANDIDATES)
        keyword = self.keyword_ranking(question, CANDIDATES)

        semantic_rank = {index: rank for rank, index in enumerate(semantic, 1)}
        keyword_rank = {index: rank for rank, index in enumerate(keyword, 1)}

        fused = {}
        for ranks in (semantic_rank, keyword_rank):
            for index, rank in ranks.items():
                fused[index] = fused.get(index, 0.0) + 1.0 / (RRF_K + rank)

        best = sorted(fused, key=fused.get, reverse=True)[:top_k]

        # Plain fusion can bury a chunk that is the clear winner in only one
        # search (e.g. an exact keyword match whose table text embeds badly).
        # Always keep the single best result from each search.
        reserved = [ranking[0] for ranking in (keyword, semantic) if ranking]
        for index in reserved:
            if index in best:
                continue
            # Replace the lowest-ranked result that is not itself reserved.
            for position in range(len(best) - 1, -1, -1):
                if best[position] not in reserved:
                    best[position] = index
                    break

        return [
            {
                "page": self.metadatas[index]["page"],
                "text": self.documents[index],
                "kind": self.metadatas[index].get("kind", "text"),
                "group": self.metadatas[index].get("group", ""),
                "score": fused[index],
                "semantic_rank": semantic_rank.get(index),
                "keyword_rank": keyword_rank.get(index),
            }
            for index in best
        ]

    def search_semantic_only(self, question: str, top_k: int = 5) -> list[dict]:
        """Plain semantic search, kept for before/after comparisons."""
        return [
            {
                "page": self.metadatas[index]["page"],
                "text": self.documents[index],
            }
            for index in self.semantic_ranking(question, top_k)
        ]
