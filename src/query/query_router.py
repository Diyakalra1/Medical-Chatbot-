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

        normalized_query = query.strip().lower()

        high_risk_match = self._match_pattern(
            normalized_query,
            self.high_risk_patterns
        )

        if high_risk_match:

            return self._build_route(
                QueryIntent.HIGH_RISK,
                f"Matched high-risk pattern: {high_risk_match}",
                "local_rule",
                start_time
            )

        conversation_match = self._match_pattern(
            normalized_query,
            self.conversation_patterns
        )

        if conversation_match:

            return self._build_route(
                QueryIntent.CONVERSATION,
                f"Matched conversation pattern: {conversation_match}",
                "local_rule",
                start_time
            )

        return self._build_route(
            QueryIntent.MEDICAL,
            "Forwarded to medical evidence pipeline",
            "evidence_pipeline",
            start_time
        )

    def _match_pattern(
        self,
        query: str,
        patterns: list[str]
    ):

        for pattern in patterns:

            pattern_regex = (
                r"\b"
                + re.escape(pattern)
                + r"\b"
            )

            if re.search(
                pattern_regex,
                query,
                re.IGNORECASE
            ):
                return pattern

        return None

    def _build_route(
        self,
        intent: QueryIntent,
        reason: str,
        routing_method: str,
        start_time: float
    ) -> QueryRoute:

        routing_ms = (
            perf_counter() - start_time
        ) * 1000

        return QueryRoute(
            intent=intent,
            reason=reason,
            routing_method=routing_method,
            routing_ms=routing_ms
        )