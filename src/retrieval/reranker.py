from dataclasses import dataclass
from time import perf_counter
from typing import List
from sentence_transformers import CrossEncoder
from src.retrieval.retriever import RetrievedCandidate


@dataclass
class RerankedCandidate:
    document: object
    retrieval_score: float
    reranker_score: float
    original_rank: int
    reranked_rank: int


@dataclass
class RerankingResult:
    candidates: List[RerankedCandidate]
    reranking_ms: float


class MedicalReranker:
    def __init__(self):
        # Scores how relevant each retrieved chunk is to the user's query.
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedCandidate],
        top_n: int = 3
    ) -> RerankingResult:

        start_time = perf_counter()

        if not candidates:
            return RerankingResult([], 0.0)

        # CrossEncoder expects (query, document) pairs.
        reranker_scores = self.model.predict([
            (query, candidate.document.page_content)
            for candidate in candidates
        ])

        scored_candidates = [
            (candidate, float(score))
            for candidate, score in zip(candidates, reranker_scores)
        ]

        # Sort by reranker score (highest first).
        scored_candidates.sort(key=lambda item: item[1], reverse=True)

        reranked_candidates = []

        for reranked_rank, (candidate, reranker_score) in enumerate(
            scored_candidates[:top_n],
            start=1
        ):
            reranked_candidates.append(
                RerankedCandidate(
                    document=candidate.document,
                    retrieval_score=candidate.similarity_score,
                    reranker_score=reranker_score,
                    original_rank=candidate.rank,
                    reranked_rank=reranked_rank
                )
            )

        reranking_ms = (perf_counter() - start_time) * 1000

        return RerankingResult(
            candidates=reranked_candidates,
            reranking_ms=reranking_ms
        )