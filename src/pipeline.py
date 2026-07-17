from src.query.query_router import QueryIntent

def retrieve_answers(
    query,
    query_router,
    medical_retriever,
    medical_reranker,
    context_evaluator,
    models_ready,
    handle_conversation,
    build_high_risk_response,
    generate_answer,
    client,
    model_name,
):
    route = query_router.route(query)

    print("\n--- ROUTING TRACE ---")
    print("Query Intent:", route.intent.value)
    print("Routing Method:", route.routing_method)
    print("Routing Latency:", round(route.routing_ms, 2), "ms")
    print("Routing Reason:", route.reason)

    base_trace = {
        "intent": route.intent.value.upper(),
        "routing_method": route.routing_method,
        "routing_ms": round(route.routing_ms, 2),
        "routing_reason": route.reason,
    }

    if route.intent == QueryIntent.CONVERSATION:
        return {
            "answer": handle_conversation(query),
            "trace": {
                **base_trace,
                "pipeline_path": "CONVERSATION",
                "decision": "CONVERSATION",
            },
        }

    if route.intent == QueryIntent.HIGH_RISK:
        return {
            "answer": build_high_risk_response(),
            "trace": {
                **base_trace,
                "pipeline_path": "HIGH_RISK",
                "decision": "HIGH_RISK",
            },
        }

    if not models_ready:
        return {
            "answer": (
                "MedAssist is preparing the medical evidence pipeline. "
                "Please try again in a moment."
            ),
            "trace": {
                **base_trace,
                "pipeline_path": "MODEL_WARMUP",
                "decision": "MODEL_LOADING",
            },
        }

    retrieval_result = medical_retriever.retrieve(query=query, top_k=10)

    print("\n--- RETRIEVAL TRACE ---")
    print("Retrieval Latency:", round(retrieval_result.retrieval_ms, 2), "ms")

    for candidate in retrieval_result.candidates:
        print(f"Rank {candidate.rank} | Score {candidate.similarity_score:.4f}")

    if not retrieval_result.candidates:
        return {
            "answer": (
                "The medical knowledge base does not contain enough "
                "information to answer this query."
            ),
            "trace": {
                **base_trace,
                "pipeline_path": "EVIDENCE_PIPELINE",
                "documents_retrieved": 0,
                "retrieval_ms": round(retrieval_result.retrieval_ms, 2),
                "decision": "ABSTAIN",
            },
        }

    reranking_result = medical_reranker.rerank(
        query=query,
        candidates=retrieval_result.candidates,
        top_n=3,
    )

    print("\n--- RERANKING TRACE ---")
    print("Reranking Latency:", round(reranking_result.reranking_ms, 2), "ms")

    for candidate in reranking_result.candidates:
        print(
            f"New Rank {candidate.reranked_rank} | "
            f"Original Rank {candidate.original_rank} | "
            f"Retrieval Score {candidate.retrieval_score:.4f} | "
            f"Reranker Score {candidate.reranker_score:.4f}"
        )

    context_result = context_evaluator.evaluate(reranking_result.candidates)

    pipeline_decision = (
        "GENERATE" if context_result.should_generate else "ABSTAIN"
    )

    print("\n--- CONTEXT EVALUATION TRACE ---")
    print("Context Score:", round(context_result.evidence_score, 4))
    print("Decision:", pipeline_decision)
    print("Reason:", context_result.decision_reason)

    top_candidate = (
        reranking_result.candidates[0]
        if reranking_result.candidates
        else None
    )

    trace = {
        **base_trace,
        "pipeline_path": "EVIDENCE_PIPELINE",
        "documents_retrieved": len(retrieval_result.candidates),
        "retrieval_ms": round(retrieval_result.retrieval_ms, 2),
        "reranking_ms": round(reranking_result.reranking_ms, 2),
        "evidence_score": round(context_result.evidence_score, 4),
        "decision": pipeline_decision,
        "decision_reason": context_result.decision_reason,
        "total_evidence_latency_ms": round(
            retrieval_result.retrieval_ms + reranking_result.reranking_ms,
            2,
        ),
    }

    if top_candidate:
        trace["top_original_rank"] = top_candidate.original_rank
        trace["top_reranked_rank"] = top_candidate.reranked_rank
        trace["top_retrieval_score"] = round(top_candidate.retrieval_score, 4)
        trace["top_reranker_score"] = round(top_candidate.reranker_score, 4)

    if not context_result.should_generate:
        return {
            "answer": (
                "I could not find sufficiently relevant information in my "
                "medical knowledge base to provide a grounded response to "
                "this question. Please consult a qualified healthcare "
                "professional for appropriate guidance."
            ),
            "trace": trace,
        }

    documents = [
        candidate.document
        for candidate in context_result.selected_candidates
    ]

    grounded_result = generate_answer(query, documents, client,model_name)

    trace["grounding_decision"] = grounded_result["grounding_decision"]

    if grounded_result["grounding_decision"] == "UNSUPPORTED":
        trace["decision"] = "GROUNDING_REJECTED"

    elif grounded_result["grounding_decision"] == "INVALID_RESPONSE":
        trace["decision"] = "GROUNDING_ERROR"

    return {
        "answer": grounded_result["answer"],
        "trace": trace,
    }