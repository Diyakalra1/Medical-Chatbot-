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

def run_100_query_benchmark():

    results = []

    correct = 0

    llm_calls = 0

    category_stats = {}

    print("\nMEDASSIST 100 QUERY BENCHMARK")
    print("=" * 80)

    for index, test in enumerate(TEST_QUERIES, start=1):

        query = test["query"]
        category = test["category"]
        expected = test["expected"]

        result = evaluate_query(query)

        actual = result["actual"]

        passed = actual == expected

        if passed:
            correct += 1

        if actual == "GENERATE":
            llm_calls += 1

        if category not in category_stats:
            category_stats[category] = {
                "correct": 0,
                "total": 0
            }

        category_stats[category]["total"] += 1

        if passed:
            category_stats[category]["correct"] += 1

        status = "PASS" if passed else "FAIL"

        print(f"[{index:03}] {status} | {category} | {query}")

        results.append({
            "query": query,
            "category": category,
            "expected": expected,
            "actual": actual,
            "passed": passed,
            "routing_ms": round(result["routing_ms"],2),
            "retrieval_ms": round(result["retrieval_ms"],2),
            "reranking_ms": round(result["reranking_ms"],2),
            "total_pipeline_ms": round(result["total_pipeline_ms"],2),
            "top_original_rank": result["top_original_rank"],
            "max_dense_score": result["max_dense_score"],
            "top_candidate_dense_score": result["top_candidate_dense_score"],
            "top_reranker_score": result["top_reranker_score"],
            "evidence_score": result["evidence_score"]
        })

    retrieval_latencies = [
        x["retrieval_ms"]
        for x in results
    ]

    reranking_latencies = [
        x["reranking_ms"]
        for x in results
    ]

    total_latencies = [
        x["total_pipeline_ms"]
        for x in results
    ]

    promoted = [
        x
        for x in results
        if (
            x["top_original_rank"] is not None
            and x["top_original_rank"] > 1
        )
    ]

    promotion_rate = (
        len(promoted) / len(results)
    ) * 100

    agreement = (
        correct / len(TEST_QUERIES)
    ) * 100

    pass_rate = (
        llm_calls / len(TEST_QUERIES)
    ) * 100

    abstain_rate = (
        len([
            x for x in results
            if x["actual"] == "ABSTAIN"
        ])
        / len(TEST_QUERIES)
    ) * 100

    conversation_rate = (
        len([
            x for x in results
            if x["actual"] == "CONVERSATION"
        ])
        / len(TEST_QUERIES)
    ) * 100

    high_risk_rate = (
        len([
            x for x in results
            if x["actual"] == "HIGH_RISK"
        ])
        / len(TEST_QUERIES)
    ) * 100

    llm_saved = (
        (
            len(TEST_QUERIES) - llm_calls
        )
        / len(TEST_QUERIES)
    ) * 100

    print("\n")
    print("=" * 80)
    print("MEDASSIST 100 QUERY REPORT")
    print("=" * 80)

    print("\nOverall")
    print("-------------------------------")
    print("Queries:", len(TEST_QUERIES))
    print("Decision Agreement:", round(agreement,2), "%")

    print("\nLatency")
    print("-------------------------------")

    print_latency_summary(
        "Retrieval",
        retrieval_latencies
    )

    print_latency_summary(
        "Reranking",
        reranking_latencies
    )

    print_latency_summary(
        "Total Pipeline",
        total_latencies
    )

    print("\nPipeline")
    print("-------------------------------")
    print("LLM Calls:", llm_calls)
    print("LLM Calls Saved:", round(llm_saved,2), "%")
    print("Generate Rate:", round(pass_rate,2), "%")
    print("Abstain Rate:", round(abstain_rate,2), "%")
    print("Conversation Rate:", round(conversation_rate,2), "%")
    print("High Risk Rate:", round(high_risk_rate,2), "%")
    print("Promotion Rate:", round(promotion_rate,2), "%")

    print("\nCategory Accuracy")
    print("-------------------------------")

    for category in category_stats:

        accuracy = (
            category_stats[category]["correct"]
            / category_stats[category]["total"]
        ) * 100

        print(
            f"{category:30}"
            f"{accuracy:.2f}%"
            f" ({category_stats[category]['correct']}/{category_stats[category]['total']})"
        )

    csv_file = "evaluation/100_query_results.csv"

    with open(
        csv_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    report_file = "evaluation/100_query_benchmark_report.txt"

    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as report:

        report.write("MEDASSIST 100 QUERY BENCHMARK\n")
        report.write("=" * 60 + "\n\n")

        report.write(f"Total Queries : {len(TEST_QUERIES)}\n")
        report.write(f"Decision Agreement : {agreement:.2f}%\n\n")

        report.write("Latency\n")
        report.write("-----------------------------\n")

        report.write(
            f"Average Retrieval : {statistics.mean(retrieval_latencies):.2f} ms\n"
        )

        report.write(
            f"P50 Retrieval : {statistics.median(retrieval_latencies):.2f} ms\n"
        )

        report.write(
            f"P95 Retrieval : {percentile(retrieval_latencies,0.95):.2f} ms\n\n"
        )

        report.write(
            f"Average Reranking : {statistics.mean(reranking_latencies):.2f} ms\n"
        )

        report.write(
            f"P50 Reranking : {statistics.median(reranking_latencies):.2f} ms\n"
        )

        report.write(
            f"P95 Reranking : {percentile(reranking_latencies,0.95):.2f} ms\n\n"
        )

        report.write(
            f"Average Total Pipeline : {statistics.mean(total_latencies):.2f} ms\n"
        )

        report.write(
            f"P50 Total Pipeline : {statistics.median(total_latencies):.2f} ms\n"
        )

        report.write(
            f"P95 Total Pipeline : {percentile(total_latencies,0.95):.2f} ms\n\n"
        )

        report.write("Pipeline\n")
        report.write("-----------------------------\n")
        report.write(f"LLM Calls : {llm_calls}\n")
        report.write(f"LLM Calls Saved : {llm_saved:.2f}%\n")
        report.write(f"Generate Rate : {pass_rate:.2f}%\n")
        report.write(f"Abstain Rate : {abstain_rate:.2f}%\n")
        report.write(f"Conversation Rate : {conversation_rate:.2f}%\n")
        report.write(f"High Risk Rate : {high_risk_rate:.2f}%\n")
        report.write(f"Promotion Rate : {promotion_rate:.2f}%\n\n")

        report.write("Category Accuracy\n")
        report.write("-----------------------------\n")

        for category in category_stats:

            accuracy = (
                category_stats[category]["correct"]
                / category_stats[category]["total"]
            ) * 100

            report.write(
                f"{category:30}"
                f"{accuracy:.2f}% "
                f"({category_stats[category]['correct']}/{category_stats[category]['total']})\n"
            )

    print("\nCSV Saved :", csv_file)
    print("Report Saved :", report_file)


if __name__ == "__main__":

    # run_boundary_evaluation()

    # run_latency_benchmark()

    run_100_query_benchmark()