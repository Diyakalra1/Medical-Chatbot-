from enum import Enum
from dataclasses import dataclass
from time import perf_counter
import re


class QueryIntent(str, Enum):
    CONVERSATION = "conversation"
    MEDICAL = "medical"
    HIGH_RISK = "high_risk"


@dataclass
class QueryRoute:
    intent: QueryIntent
    reason: str
    routing_method: str
    routing_ms: float


class QueryRouter:

    def __init__(self):
        self.high_risk_patterns = [
            "cannot breathe",
            "can't breathe",
            "difficulty breathing",
            "not breathing",
            "severe chest pain",
            "uncontrolled bleeding",
            "bleeding heavily",
            "overdosed",
            "overdose",
            "unconscious",
            "sudden paralysis",
            "severe allergic reaction",
            "cannot move one side",
            "can't move one side",
            "cannot move my arm",
            "cannot move my leg",
            "can't move my arm",
            "can't move my leg",
            "one side of my body",
            "face drooping",
            "face is drooping",
            "sudden weakness",
            "sudden numbness",
            "slurred speech",
            "cannot speak clearly",
            "can't speak clearly"
        ]

        self.conversation_patterns = [
            "hello",
            "hi",
            "hey",
            "who are you",
            "what are you",
            "what can you do",
            "thank you",
            "thanks",
            "good morning",
            "good afternoon",
            "good evening"
        ]

    def route(self, query: str) -> QueryRoute:
        start_time = perf_counter()
        query = query.strip().lower()

        match = self._match_pattern(query, self.high_risk_patterns)
        if match:
            return QueryRoute(
                intent=QueryIntent.HIGH_RISK,
                reason=f"Matched high-risk pattern: {match}",
                routing_method="local_rule",
                routing_ms=(perf_counter() - start_time) * 1000
            )

        match = self._match_pattern(query, self.conversation_patterns)
        if match:
            return QueryRoute(
                intent=QueryIntent.CONVERSATION,
                reason=f"Matched conversation pattern: {match}",
                routing_method="local_rule",
                routing_ms=(perf_counter() - start_time) * 1000
            )

        return QueryRoute(
            intent=QueryIntent.MEDICAL,
            reason="Forwarded to medical evidence pipeline",
            routing_method="evidence_pipeline",
            routing_ms=(perf_counter() - start_time) * 1000
        )

    def _match_pattern(self, query: str, patterns: list[str]):
        for pattern in patterns:
            if re.search(rf"\b{re.escape(pattern)}\b", query, re.IGNORECASE):
                return pattern
        return None