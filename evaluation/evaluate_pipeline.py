from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

import os
import csv
import statistics

from src.helper import download_huggingface_embeddings

from src.query.query_router import (
    QueryRouter,
    QueryIntent
)

from src.retrieval.retriever import MedicalRetriever
from src.retrieval.reranker import MedicalReranker
from src.retrieval.context_evaluator import ContextEvaluator

from evaluation.test_queries import TEST_QUERIES


load_dotenv()


PINECONE_API_KEY = os.getenv(
    "PINECONE_API_KEY"
)

if not PINECONE_API_KEY:
    raise ValueError(
        "PINECONE_API_KEY is missing"
    )


pc = Pinecone(
    api_key=PINECONE_API_KEY
)


index = pc.Index(
    "medicalchatbot"
)


embedding = (
    download_huggingface_embeddings()
)


vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedding
)


router = QueryRouter()

retriever = MedicalRetriever(
    vectorstore
)

reranker = MedicalReranker()

context_evaluator = ContextEvaluator()


def evaluate_query(query):

    route = router.route(
        query
    )

    if route.intent == QueryIntent.CONVERSATION:

        return {
            "actual": "CONVERSATION",
            "routing_ms": route.routing_ms,
            "retrieval_ms": 0.0,
            "reranking_ms": 0.0,
            "total_pipeline_ms": route.routing_ms,
            "evidence_score": None,
            "top_reranker_score": None,
            "top_original_rank": None,
            "max_dense_score": None,
            "top_candidate_dense_score": None
        }

    if route.intent == QueryIntent.HIGH_RISK:

        return {
            "actual": "HIGH_RISK",
            "routing_ms": route.routing_ms,
            "retrieval_ms": 0.0,
            "reranking_ms": 0.0,
            "total_pipeline_ms": route.routing_ms,
            "evidence_score": None,
            "top_reranker_score": None,
            "top_original_rank": None,
            "max_dense_score": None,
            "top_candidate_dense_score": None
        }

    retrieval_result = retriever.retrieve(
        query=query,
        top_k=10
    )

    max_dense_score = None

    if retrieval_result.candidates:

        max_dense_score = max(
            candidate.similarity_score
            for candidate
            in retrieval_result.candidates
        )

    reranking_result = reranker.rerank(
        query=query,
        candidates=retrieval_result.candidates,
        top_n=3
    )

    context_result = context_evaluator.evaluate(
        reranking_result.candidates
    )

    actual = (
        "GENERATE"
        if context_result.should_generate
        else "ABSTAIN"
    )

    top_reranker_score = None
    top_original_rank = None
    top_candidate_dense_score = None

    if reranking_result.candidates:

        top_candidate = (
            reranking_result.candidates[0]
        )

        top_reranker_score = (
            top_candidate.reranker_score
        )

        top_original_rank = (
            top_candidate.original_rank
        )

        top_candidate_dense_score = (
            top_candidate.retrieval_score
        )

    total_pipeline_ms = (
        route.routing_ms
        + retrieval_result.retrieval_ms
        + reranking_result.reranking_ms
    )

    return {
        "actual": actual,
        "routing_ms": route.routing_ms,
        "retrieval_ms": (
            retrieval_result.retrieval_ms
        ),
        "reranking_ms": (
            reranking_result.reranking_ms
        ),
        "total_pipeline_ms": (
            total_pipeline_ms
        ),
        "evidence_score": (
            context_result.evidence_score
        ),
        "top_reranker_score": (
            top_reranker_score
        ),
        "top_original_rank": (
            top_original_rank
        ),
        "max_dense_score": (
            max_dense_score
        ),
        "top_candidate_dense_score": (
            top_candidate_dense_score
        )
    }


def percentile(values, percentile_value):

    sorted_values = sorted(values)

    index = (
        len(sorted_values) - 1
    ) * percentile_value

    lower_index = int(index)

    upper_index = min(
        lower_index + 1,
        len(sorted_values) - 1
    )

    fraction = (
        index - lower_index
    )

    return (
        sorted_values[lower_index]
        + (
            sorted_values[upper_index]
            - sorted_values[lower_index]
        )
        * fraction
    )


def print_latency_summary(
    name,
    values
):

    print(
        f"\n{name}"
    )

    print(
        "Mean:",
        round(
            statistics.mean(values),
            2
        ),
        "ms"
    )

    print(
        "P50:",
        round(
            statistics.median(values),
            2
        ),
        "ms"
    )

    print(
        "P95:",
        round(
            percentile(
                values,
                0.95
            ),
            2
        ),
        "ms"
    )


def run_boundary_evaluation():

    results = []

    correct = 0

    print(
        "\nMEDASSIST PIPELINE EVALUATION"
    )

    print(
        "=" * 70
    )

    for index, test in enumerate(
        TEST_QUERIES,
        start=1
    ):

        query = test["query"]
        expected = test["expected"]

        result = evaluate_query(
            query
        )

        actual = result["actual"]

        passed = (
            actual == expected
        )

        if passed:
            correct += 1

        status = (
            "PASS"
            if passed
            else "FAIL"
        )

        print(
            f"\n[{index}] {status}"
        )

        print(
            "Query:",
            query
        )

        print(
            "Expected:",
            expected
        )

        print(
            "Actual:",
            actual
        )

        print(
            "Max Dense Score:",
            result["max_dense_score"]
        )

        print(
            "Top Candidate Dense Score:",
            result[
                "top_candidate_dense_score"
            ]
        )

        print(
            "Top Reranker Score:",
            result["top_reranker_score"]
        )

        print(
            "Evidence Score:",
            result["evidence_score"]
        )

        results.append({
            "query": query,
            "category": test["category"],
            "expected": expected,
            "actual": actual,
            "passed": passed,
            "routing_ms": round(
                result["routing_ms"],
                2
            ),
            "retrieval_ms": round(
                result["retrieval_ms"],
                2
            ),
            "reranking_ms": round(
                result["reranking_ms"],
                2
            ),
            "total_pipeline_ms": round(
                result["total_pipeline_ms"],
                2
            ),
            "top_original_rank": (
                result["top_original_rank"]
            ),
            "max_dense_score": (
                result["max_dense_score"]
            ),
            "top_candidate_dense_score": (
                result[
                    "top_candidate_dense_score"
                ]
            ),
            "top_reranker_score": (
                result["top_reranker_score"]
            ),
            "evidence_score": (
                result["evidence_score"]
            )
        })

    agreement = (
        correct / len(TEST_QUERIES)
    ) * 100

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BOUNDARY EVALUATION RESULTS"
    )

    print(
        "Passed:",
        correct,
        "/",
        len(TEST_QUERIES)
    )

    print(
        "Decision Agreement:",
        round(
            agreement,
            2
        ),
        "%"
    )

    output_file = (
        "evaluation/evaluation_results.csv"
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    print(
        "\nResults saved to:",
        output_file
    )


def run_latency_benchmark():

    supported_queries = [
        test["query"]
        for test in TEST_QUERIES
        if test["category"]
        == "supported_medical"
    ]

    benchmark_results = []

    runs_per_query = 3

    print(
        "\n\nMEDASSIST SUPPORTED PATH BENCHMARK"
    )

    print(
        "=" * 70
    )

    for query_index, query in enumerate(
        supported_queries,
        start=1
    ):

        print(
            f"\n[{query_index}] {query}"
        )

        for run_number in range(
            1,
            runs_per_query + 1
        ):

            result = evaluate_query(
                query
            )

            print(
                f"Run {run_number} | "
                f"Retrieval "
                f"{result['retrieval_ms']:.2f} ms | "
                f"Reranking "
                f"{result['reranking_ms']:.2f} ms | "
                f"Total "
                f"{result['total_pipeline_ms']:.2f} ms | "
                f"Original Rank "
                f"{result['top_original_rank']} → 1"
            )

            benchmark_results.append({
                "query": query,
                "run": run_number,
                "routing_ms": round(
                    result["routing_ms"],
                    2
                ),
                "retrieval_ms": round(
                    result["retrieval_ms"],
                    2
                ),
                "reranking_ms": round(
                    result["reranking_ms"],
                    2
                ),
                "total_pipeline_ms": round(
                    result["total_pipeline_ms"],
                    2
                ),
                "top_original_rank": (
                    result["top_original_rank"]
                ),
                "max_dense_score": (
                    result["max_dense_score"]
                ),
                "top_candidate_dense_score": (
                    result[
                        "top_candidate_dense_score"
                    ]
                ),
                "top_reranker_score": (
                    result["top_reranker_score"]
                ),
                "evidence_score": (
                    result["evidence_score"]
                ),
                "decision": (
                    result["actual"]
                )
            })

    retrieval_latencies = [
        result["retrieval_ms"]
        for result in benchmark_results
    ]

    reranking_latencies = [
        result["reranking_ms"]
        for result in benchmark_results
    ]

    total_latencies = [
        result["total_pipeline_ms"]
        for result in benchmark_results
    ]

    promoted_candidates = [
        result
        for result in benchmark_results
        if (
            result["top_original_rank"]
            is not None
            and result["top_original_rank"] > 1
        )
    ]

    promotion_rate = (
        len(promoted_candidates)
        / len(benchmark_results)
    ) * 100

    generated_results = [
        result
        for result in benchmark_results
        if result["decision"] == "GENERATE"
    ]

    generation_rate = (
        len(generated_results)
        / len(benchmark_results)
    ) * 100

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SUPPORTED PATH SUMMARY"
    )

    print_latency_summary(
        "Retrieval Latency",
        retrieval_latencies
    )

    print_latency_summary(
        "Reranking Latency",
        reranking_latencies
    )

    print_latency_summary(
        "Total Evidence Pipeline Latency",
        total_latencies
    )

    print(
        "\nReranker Promotion Rate:",
        round(
            promotion_rate,
            2
        ),
        "%"
    )

    print(
        "Supported Query Generation Rate:",
        round(
            generation_rate,
            2
        ),
        "%"
    )

    output_file = (
        "evaluation/latency_results.csv"
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                benchmark_results[0].keys()
            )
        )

        writer.writeheader()

        writer.writerows(
            benchmark_results
        )

    print(
        "\nLatency results saved to:",
        output_file
    )


if __name__ == "__main__":

    run_boundary_evaluation()

    run_latency_benchmark()