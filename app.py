from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from src.helper import download_huggingface_embeddings
from src.retrieval.context_evaluator import ContextEvaluator
from src.retrieval.reranker import MedicalReranker
from src.retrieval.retriever import MedicalRetriever
from src.query.query_router import QueryRouter, QueryIntent

import os
import threading


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing")


MODEL_NAME = "gemini-3-flash-preview"


medical_retriever = None
medical_reranker = None
context_evaluator = None

models_ready = False
model_loading = False
model_error = None

model_lock = threading.Lock()


client = genai.Client(
    api_key=GEMINI_API_KEY
)

query_router = QueryRouter()


def initialize_medical_pipeline():

    global medical_retriever
    global medical_reranker
    global context_evaluator

    global models_ready
    global model_loading
    global model_error

    with model_lock:

        if models_ready or model_loading:
            return

        model_loading = True

    print("\n--- MODEL INITIALIZATION ---")
    print("Starting MedAssist pipeline warm-up...")

    try:

        pc = Pinecone(
            api_key=PINECONE_API_KEY
        )

        print("Pinecone client initialized")

        index = pc.Index(
            "medicalchatbot"
        )

        print("Pinecone index connected")

        embedding = (
            download_huggingface_embeddings()
        )

        print("Embedding model loaded")

        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embedding
        )

        medical_retriever = MedicalRetriever(
            vectorstore
        )

        print("Medical retriever initialized")

        medical_reranker = MedicalReranker()

        print("Medical reranker loaded")

        context_evaluator = ContextEvaluator()

        print("Context evaluator initialized")

        models_ready = True
        model_error = None

        print(
            "MedAssist pipeline READY"
        )

    except Exception as error:

        model_error = str(error)

        print(
            "MODEL INITIALIZATION ERROR:",
            error
        )

    finally:

        model_loading = False


def start_model_warmup():

    warmup_thread = threading.Thread(
        target=initialize_medical_pipeline,
        daemon=True
    )

    warmup_thread.start()


def handle_conversation(query):

    normalized_query = query.strip().lower()

    if (
        "who are you" in normalized_query
        or "what are you" in normalized_query
    ):
        return (
            "I'm MedAssist, a grounded health information assistant. "
            "I use a medical knowledge base to provide evidence-supported "
            "general health information."
        )

    if "what can you do" in normalized_query:
        return (
            "I can help explain medical conditions, symptoms, causes, "
            "and general health information using evidence retrieved from "
            "my medical knowledge base."
        )

    if "thank" in normalized_query:
        return (
            "You're welcome. Feel free to ask another "
            "health-related question."
        )

    return (
        "Hello! I'm MedAssist. I can help you understand general health "
        "and medical topics using a grounded medical knowledge base. "
        "What would you like to know?"
    )


def build_high_risk_response():

    return """
This may describe a potentially serious medical situation.

Please seek immediate medical attention or contact local emergency
services. Do not rely on MedAssist for urgent medical situations.

If possible, ask someone nearby for assistance and seek professional
medical care immediately.
"""


def build_context(documents):

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1
    ):

        source = document.metadata.get(
            "source",
            "Medical Knowledge Base"
        )

        page = document.metadata.get(
            "page",
            "Unknown"
        )

        context_parts.append(
            f"""
SOURCE {index}
Document: {source}
Page: {page}

{document.page_content}
"""
        )

    return "\n".join(context_parts)


def generate_answer(query, documents):

    context = build_context(documents)

    prompt = f"""
You are MedAssist, a grounded health information assistant.

You are NOT a doctor and you must NOT diagnose diseases.

Your task has TWO stages.

STAGE 1: GROUNDING DECISION

Determine whether the MEDICAL CONTEXT contains sufficient evidence
to address the specific USER QUERY.

You must evaluate support for the actual question or claim.

Topic overlap alone is NOT sufficient.

For example:

If the user asks:

"Can crystals cure asthma?"

and the context only discusses asthma symptoms, causes, or standard
medical management but provides no evidence about crystals curing asthma,
the query is UNSUPPORTED.

If the user asks:

"What causes asthma?"

and the context explains asthma causes or triggers,
the query is SUPPORTED.

If the user says:

"My doctor told me I have diabetes. Help me understand the condition."

and the context explains diabetes,
the query is SUPPORTED.

IMPORTANT GROUNDING RULES:

1. Use ONLY the provided medical context.
2. Do not use your own medical knowledge.
3. Topic similarity does not mean the user's claim is supported.
4. Treatments, cures, causes, precautions, complications, and symptoms
   must be explicitly supported by the context.
5. If the context cannot address the central question, classify the
   query as UNSUPPORTED.
6. Never infer that a treatment works merely because the disease is
   discussed in the context.
7. A diagnosis stated by the user may be treated as background context.
   Do not independently verify or confirm the diagnosis.

STAGE 2: GROUNDED ANSWER

If the query is SUPPORTED, answer using ONLY the medical context.

If the query is UNSUPPORTED, do not attempt to answer the medical claim.

USER QUERY:

{query}

MEDICAL CONTEXT:

{context}

Return your response using EXACTLY this format:

GROUNDING_DECISION: SUPPORTED

ANSWER:
GENERAL INFORMATION
Explain relevant general medical information supported by the context.

WHAT MAY BE RELEVANT
Explain which details from the user's query are relevant.
Do not diagnose.

GENERAL GUIDANCE
Provide only general guidance supported by the context.

WHEN TO SEEK MEDICAL CARE
Explain warning signs or escalation guidance supported by the context.

SOURCES
List the source numbers used.

OR

GROUNDING_DECISION: UNSUPPORTED

ANSWER:
The retrieved medical evidence does not contain sufficient information
to support or address the specific claim or question. I cannot provide
a grounded answer from the available medical knowledge base.

Do not include any text before GROUNDING_DECISION.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return parse_grounded_response(
        response.text
    )


def parse_grounded_response(response_text):

    cleaned_response = response_text.strip()

    first_line = (
        cleaned_response
        .splitlines()[0]
        .strip()
        .upper()
    )

    if first_line == "GROUNDING_DECISION: SUPPORTED":
        grounding_decision = "SUPPORTED"

    elif first_line == "GROUNDING_DECISION: UNSUPPORTED":
        grounding_decision = "UNSUPPORTED"

    else:
        grounding_decision = "INVALID_RESPONSE"

    print("\n--- GROUNDING TRACE ---")
    print(
        "Grounding Decision:",
        grounding_decision
    )

    if grounding_decision == "INVALID_RESPONSE":

        print(
            "Reason: Model did not follow "
            "grounding response contract"
        )

        return {
            "answer": (
                "I could not verify sufficient support "
                "from the retrieved medical evidence."
            ),
            "grounding_decision": grounding_decision
        }

    if grounding_decision == "UNSUPPORTED":

        print(
            "Reason: Retrieved context does not "
            "support the specific query or claim"
        )

        return {
            "answer": (
                "The retrieved medical evidence does not "
                "contain sufficient information to support "
                "or address this specific claim or question. "
                "I cannot provide a grounded answer from the "
                "available medical knowledge base."
            ),
            "grounding_decision": grounding_decision
        }

    print(
        "Reason: Retrieved context supports "
        "the specific query"
    )

    answer_marker = "ANSWER:"

    marker_index = cleaned_response.find(
        answer_marker
    )

    if marker_index == -1:

        return {
            "answer": (
                "I could not construct a grounded response "
                "from the retrieved medical evidence."
            ),
            "grounding_decision": "INVALID_RESPONSE"
        }

    answer = cleaned_response[
        marker_index + len(answer_marker):
    ].strip()

    return {
        "answer": answer,
        "grounding_decision": grounding_decision
    }


def retrieve_answers(query):

    route = query_router.route(query)

    print("\n--- ROUTING TRACE ---")
    print("Query Intent:", route.intent.value)
    print("Routing Method:", route.routing_method)
    print(
        "Routing Latency:",
        round(route.routing_ms, 2),
        "ms"
    )
    print("Routing Reason:", route.reason)

    base_trace = {
        "intent": route.intent.value.upper(),
        "routing_method": route.routing_method,
        "routing_ms": round(
            route.routing_ms,
            2
        ),
        "routing_reason": route.reason
    }

    if route.intent == QueryIntent.CONVERSATION:

        return {
            "answer": handle_conversation(query),
            "trace": {
                **base_trace,
                "pipeline_path": "CONVERSATION",
                "decision": "CONVERSATION"
            }
        }

    if route.intent == QueryIntent.HIGH_RISK:

        return {
            "answer": build_high_risk_response(),
            "trace": {
                **base_trace,
                "pipeline_path": "HIGH_RISK",
                "decision": "HIGH_RISK"
            }
        }

    if not models_ready:

        return {
            "answer": (
                "MedAssist is preparing the medical "
                "evidence pipeline. Please try again "
                "in a moment."
            ),
            "trace": {
                **base_trace,
                "pipeline_path": "MODEL_WARMUP",
                "decision": "MODEL_LOADING"
            }
        }

    retrieval_result = medical_retriever.retrieve(
        query=query,
        top_k=10
    )

    print("\n--- RETRIEVAL TRACE ---")

    print(
        "Retrieval Latency:",
        round(
            retrieval_result.retrieval_ms,
            2
        ),
        "ms"
    )

    for candidate in retrieval_result.candidates:

        print(
            f"Rank {candidate.rank} | "
            f"Score "
            f"{candidate.similarity_score:.4f}"
        )

    if not retrieval_result.candidates:

        return {
            "answer": (
                "The medical knowledge base does not "
                "contain enough information to answer "
                "this query."
            ),
            "trace": {
                **base_trace,
                "pipeline_path": "EVIDENCE_PIPELINE",
                "documents_retrieved": 0,
                "retrieval_ms": round(
                    retrieval_result.retrieval_ms,
                    2
                ),
                "decision": "ABSTAIN"
            }
        }

    reranking_result = medical_reranker.rerank(
        query=query,
        candidates=retrieval_result.candidates,
        top_n=3
    )

    print("\n--- RERANKING TRACE ---")

    print(
        "Reranking Latency:",
        round(
            reranking_result.reranking_ms,
            2
        ),
        "ms"
    )

    for candidate in reranking_result.candidates:

        print(
            f"New Rank {candidate.reranked_rank} | "
            f"Original Rank {candidate.original_rank} | "
            f"Retrieval Score "
            f"{candidate.retrieval_score:.4f} | "
            f"Reranker Score "
            f"{candidate.reranker_score:.4f}"
        )

    context_result = context_evaluator.evaluate(
        reranking_result.candidates
    )

    pipeline_decision = (
        "GENERATE"
        if context_result.should_generate
        else "ABSTAIN"
    )

    print("\n--- CONTEXT EVALUATION TRACE ---")

    print(
        "Context Score:",
        round(
            context_result.evidence_score,
            4
        )
    )

    print(
        "Decision:",
        pipeline_decision
    )

    print(
        "Reason:",
        context_result.decision_reason
    )

    top_candidate = None

    if reranking_result.candidates:
        top_candidate = (
            reranking_result.candidates[0]
        )

    trace = {
        **base_trace,
        "pipeline_path": "EVIDENCE_PIPELINE",
        "documents_retrieved": len(
            retrieval_result.candidates
        ),
        "retrieval_ms": round(
            retrieval_result.retrieval_ms,
            2
        ),
        "reranking_ms": round(
            reranking_result.reranking_ms,
            2
        ),
        "evidence_score": round(
            context_result.evidence_score,
            4
        ),
        "decision": pipeline_decision,
        "decision_reason": (
            context_result.decision_reason
        ),
        "total_evidence_latency_ms": round(
            retrieval_result.retrieval_ms
            + reranking_result.reranking_ms,
            2
        )
    }

    if top_candidate:

        trace["top_original_rank"] = (
            top_candidate.original_rank
        )

        trace["top_reranked_rank"] = (
            top_candidate.reranked_rank
        )

        trace["top_retrieval_score"] = round(
            top_candidate.retrieval_score,
            4
        )

        trace["top_reranker_score"] = round(
            top_candidate.reranker_score,
            4
        )

    if not context_result.should_generate:

        return {
            "answer": (
                "I could not find sufficiently relevant "
                "information in my medical knowledge base "
                "to provide a grounded response to this "
                "question. Please consult a qualified "
                "healthcare professional for appropriate "
                "guidance."
            ),
            "trace": trace
        }

    documents = [
        candidate.document
        for candidate
        in context_result.selected_candidates
    ]

    grounded_result = generate_answer(
        query,
        documents
    )

    trace["grounding_decision"] = (
        grounded_result["grounding_decision"]
    )

    if (
        grounded_result["grounding_decision"]
        == "UNSUPPORTED"
    ):
        trace["decision"] = "GROUNDING_REJECTED"

    elif (
        grounded_result["grounding_decision"]
        == "INVALID_RESPONSE"
    ):
        trace["decision"] = "GROUNDING_ERROR"

    return {
        "answer": grounded_result["answer"],
        "trace": trace
    }


@app.route("/")
def index():

    return render_template(
        "chat.html"
    )


@app.route("/ready")
def ready():

    if models_ready:

        return jsonify({
            "ready": True,
            "status": "READY"
        })

    if model_error:

        return jsonify({
            "ready": False,
            "status": "ERROR",
            "error": model_error
        }), 500

    return jsonify({
        "ready": False,
        "status": "LOADING"
    })


@app.route(
    "/get",
    methods=["POST"]
)
def chat():

    message = request.form.get(
        "msg",
        ""
    ).strip()

    if not message:

        return jsonify({
            "answer": (
                "Please enter a health-related question."
            ),
            "trace": None
        })

    print("Input:", message)

    try:

        result = retrieve_answers(
            message
        )

        print(
            "Response:",
            result["answer"]
        )

        return jsonify(
            result
        )

    except Exception as error:

        print(
            "Error:",
            error
        )

        return jsonify({
            "answer": (
                "MedAssist encountered an error while "
                "processing your query."
            ),
            "trace": {
                "decision": "PIPELINE_ERROR"
            }
        }), 500


start_model_warmup()


if __name__ == "__main__":

    app.run(
        debug=True,
        use_reloader=False
    )