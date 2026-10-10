
import re
from collections import defaultdict

import chromadb
from sentence_transformers import SentenceTransformer

VECTOR_DB_PATH = "data/vector_db"
COLLECTION_NAME = "field_manuals"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
RRF_K = 60


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:
    def __init__(self):
        print("Loading local embedding model...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        print("Connecting to existing vector database...")
        client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        self.collection = client.get_collection(COLLECTION_NAME)

        stored = self.collection.get(
            include=["documents", "metadatas"]
        )

        self.ids = stored["ids"]
        self.documents = stored["documents"]
        self.metadatas = stored["metadatas"]

        if not self.documents:
            raise ValueError("The vector database contains no documents.")

        print(f"Loaded {len(self.documents)} existing chunks.")

        self.bm25_documents = [
            tokenize(document) for document in self.documents
        ]

        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError(
                "Install the missing dependency using: "
                "python -m pip install rank-bm25"
            ) from exc

        self.bm25 = BM25Okapi(self.bm25_documents)
        self.index_by_id = {
            chunk_id: index
            for index, chunk_id in enumerate(self.ids)
        }

    def search(self, question, top_k=TOP_K):
        scores = defaultdict(float)

        # Semantic retrieval from the existing ChromaDB collection.
        semantic = self.collection.query(
            query_embeddings=[
                self.model.encode(question).tolist()
            ],
            n_results=min(15, len(self.documents)),
            include=["documents", "metadatas", "distances"],
        )

        for rank, chunk_id in enumerate(semantic["ids"][0], start=1):
            scores[chunk_id] += 1 / (RRF_K + rank)

        # Keyword retrieval using BM25.
        keyword_scores = self.bm25.get_scores(tokenize(question))
        keyword_order = sorted(
            range(len(keyword_scores)),
            key=lambda i: keyword_scores[i],
            reverse=True,
        )

        for rank, index in enumerate(keyword_order[:15], start=1):
            if keyword_scores[index] > 0:
                scores[self.ids[index]] += 1 / (RRF_K + rank)

        question_words = set(tokenize(question))
        battery_question = bool(
            question_words
            & {"battery", "batteries", "weak", "dead", "flat", "charging"}
        )

        troubleshooting_question = bool(
            question_words
            & {
                "troubleshooting", "malfunction", "failure", "failed",
                "fault", "faulty", "problem", "problems", "issue",
                "issues", "starting", "start", "started", "engine",
                "generator", "diesel", "work", "working", "cause",
                "causes", "output", "electricity", "electrical",
            }
        )

        def priority(chunk_id):
            index = self.index_by_id[chunk_id]
            document = self.documents[index].lower()
            section = self.metadatas[index].get("section", "general")

            if section == "troubleshooting":
                if battery_question and (
                    "battery" in document
                    or "batteries" in document
                    or "charge the battery" in document
                ):
                    return 0

                if troubleshooting_question:
                    return 1

            return 2

        ranked_ids = sorted(
            scores,
            key=lambda chunk_id: (
                priority(chunk_id),
                -scores[chunk_id],
            ),
        )

        results = []

        for chunk_id in ranked_ids[:top_k]:
            index = self.index_by_id[chunk_id]
            results.append({
                "document": self.documents[index],
                "metadata": self.metadatas[index],
                "score": scores[chunk_id],
            })

        return results


def main():
    retriever = HybridRetriever()

    questions = [
        "Why is generator not starting?",
        "What should I check if diesel engine will not start?",
        "weak battery",
        "Why is the generator producing no electrical output?",
    ]

    for question in questions:
        print("\n" + "=" * 70)
        print(f"QUESTION: {question}")
        print("=" * 70)

        for rank, result in enumerate(
            retriever.search(question), start=1
        ):
            metadata = result["metadata"]

            print(
                f"\nResult {rank} | Page {metadata.get('page')} "
                f"| Section: {metadata.get('section', 'unknown')}"
            )
            print(result["document"][:900])
            print(f"Hybrid score: {result['score']:.5f}")


if __name__ == "__main__":
    main()
