from dataclasses import dataclass
from typing import List
from time import perf_counter

from langchain_core.documents import Document


@dataclass
class RetrievedCandidate:
    document: Document
    similarity_score: float
    rank: int


@dataclass
class RetrievalResult:
    candidates: List[RetrievedCandidate]
    retrieval_ms: float


class MedicalRetriever:

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> RetrievalResult:

        start_time = perf_counter()

        results = (
            self.vectorstore
            .similarity_search_with_score(
                query=query,
                k=top_k
            )
        )

        retrieval_ms = (
            perf_counter() - start_time
        ) * 1000

        candidates = []

        for rank, (document, score) in enumerate(
            results,
            start=1
        ):
            candidates.append(
                RetrievedCandidate(
                    document=document,
                    similarity_score=float(score),
                    rank=rank
                )
            )

        return RetrievalResult(
            candidates=candidates,
            retrieval_ms=retrieval_ms
        )