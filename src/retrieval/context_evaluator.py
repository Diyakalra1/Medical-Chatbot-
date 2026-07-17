from dataclasses import dataclass
from typing import List
from src.retrieval.reranker import RerankedCandidate


@dataclass
class ContextEvaluationResult:
    selected_candidates: List[RerankedCandidate]
    evidence_score: float
    should_generate: bool
    decision_reason: str


class ContextEvaluator:
    def __init__(
        self,
        dense_threshold: float = 0.50,
        reranker_threshold: float = 0.0,
        hybrid_reranker_threshold: float = -4.0
    ):
        self.dense_threshold = dense_threshold
        self.reranker_threshold = reranker_threshold
        self.hybrid_reranker_threshold = hybrid_reranker_threshold

    def evaluate(
        self,
        candidates: List[RerankedCandidate]
    ) -> ContextEvaluationResult:

        if not candidates:
            return ContextEvaluationResult(
                [], 0.0, False, "No reranked evidence available"
            )

        top_candidate = candidates[0]
        strong_reranker_evidence = ( top_candidate.reranker_score >= self.reranker_threshold)
        strong_hybrid_evidence = (top_candidate.retrieval_score >= self.dense_threshold and top_candidate.reranker_score >= self.hybrid_reranker_threshold)
        should_generate = (strong_reranker_evidence or strong_hybrid_evidence)

        if strong_reranker_evidence:
            evidence_score = top_candidate.reranker_score
            decision_reason = "Strong cross-encoder relevance evidence"

        elif strong_hybrid_evidence:
            evidence_score = top_candidate.retrieval_score
            decision_reason = (
                "Dense retrieval and reranker jointly passed hybrid evidence thresholds"
            )

        else:
            evidence_score = 0.0
            decision_reason = (
                "Retrieved evidence was insufficient for grounded generation"
            )

        return ContextEvaluationResult(
            candidates if should_generate else [],
            evidence_score,
            should_generate,
            decision_reason
        )