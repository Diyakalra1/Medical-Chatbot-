from dataclasses import dataclass
from typing import List
from time import perf_counter

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
        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L6-v2"
        )

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedCandidate],
        top_n: int = 3
    ) -> RerankingResult:

        start_time = perf_counter()

        if not candidates:
            return RerankingResult(
                candidates=[],
                reranking_ms=0.0
            )

        query_document_pairs = [
            (
                query,
                candidate.document.page_content
            )
            for candidate in candidates
        ]

        scores = self.model.predict(
            query_document_pairs
        )

        scored_candidates = []

        for candidate, score in zip(
            candidates,
            scores
        ):
            scored_candidates.append(
                (
                    candidate,
                    float(score)
                )
            )

        scored_candidates.sort(
            key=lambda item: item[1],
            reverse=True
        )

        selected_candidates = scored_candidates[:top_n]

        reranked_candidates = []

        for reranked_rank, (
            candidate,
            reranker_score
        ) in enumerate(
            selected_candidates,
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

        reranking_ms = (
            perf_counter() - start_time
        ) * 1000

        return RerankingResult(
            candidates=reranked_candidates,
            reranking_ms=reranking_ms
        )